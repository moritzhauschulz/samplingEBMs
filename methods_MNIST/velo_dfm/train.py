import os
import copy
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion

from velo_dfm.model import ResNetFlow

def load_and_sample(args):
    model = ResNetFlow(64, args)

    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    try:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        print(f'successfully loaded model...')
    except FileNotFoundError as e:
        print('Checkpoint not found.')
        sys.exit(1)

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
            
    def get_independent_sample(loader, args=args):
        (x, _) = next(iter(loader))
        return preprocess(x)


    if args.source == 'data':
        train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
        xt = get_independent_sample(test_loader).long().to(args.device) 
        plot(f'./source_{time.time()}.png', xt.float())
    else:
        xt = None

    model.to(args.device)
    model_samples = gen_samples(model, args, xt=xt)
    out = f'./samples_{time.time()}.png'
    plot(out, torch.tensor(model_samples).float())
    print(f'saved to {out}')

def gen_samples(model, args, batch_size=None, t=0.0, xt=None, print_stats=True):
    model.eval()
    S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
    D = args.discrete_dim
    # Variables, B, D for batch size and number of dimensions respectively
    B = batch_size if batch_size is not None else args.batch_size
    if not xt is None:
        B = xt.shape[0]

    if args.source == 'mask':
        M = S - 1
    # Initialize xt with the mask index value if not provided
    if xt is None:
        if args.source == 'mask':
            xt = M * torch.ones((B, D), dtype=torch.long).to(args.device)
        else:
            xt = torch.randint(0, S, (B, D)).to(args.device)

    t = 0.0  # Initial time

    while t < 1.0:
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 
        if args.scheduler_type == 'linear':
            a1 = 1 / (1 - t)
            b = -a1
            #adaptive dt
            if t>0:
                dt = min(args.delta_t, (1 - t))
            else:
                dt = args.delta_t
        elif args.scheduler_type == 'quadratic':
            a1 = (2 * t) / (1 - t ** 2)
            b = -a1
            #adaptive dt
            if t>0:
                dt = min(args.delta_t, (1 - t ** 2)/(2 * t))
            else:
                dt = args.delta_t
        elif args.scheduler_type == 'quadratic_noise':
            a1 = t * (2 - t)/(1 - t)
            a2 = 1 - t
            b = -1 /(1 - t)
            #adaptive dt 
            if t>0:
                dt = min(args.delta_t, 1 - t)
            else:
                dt = args.delta_t

        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            x1_logits, noise_logits = model(xt, t_)
            x1_logits = F.softmax(x1_logits, dim=-1)
            if args.scheduler_type == 'quadratic_noise':
                noise_logits = F.softmax(noise_logits, dim=-1)
                ut = a1 * x1_logits + a2 * noise_logits + b * delta_xt #a3 is zero...
            else:
                ut = a1 * x1_logits + b * delta_xt #a3 is zero...
        
        step_probs = delta_xt + (ut * dt)

        if args.impute_self_connections:
            step_probs = step_probs.clamp(max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 

        t += dt

        step_probs = step_probs.clamp(min=0) #can I avoid this?


        if t < 1.0 or not args.source == 'mask':
            xt = Categorical(step_probs).sample() #(B,D)
        else:
            if print_stats:
                if torch.any(xt == M):
                    num_masked_entries = torch.sum(xt == M).item()
                    print(f"Share of masked entries (over all entries) in the final but one tensor: {num_masked_entries/ (B * D)}")
                    print(f"Forcing mask values into range...")
                print(f'Share of samples with non-zero probability for at least one mask: {(step_probs[:,:,M].sum(dim=-1)>0.001).sum()/B}')
            step_probs[:, :, M] = 0
            step_probs_sum = step_probs.sum(dim=-1, keepdim=True)
            zero_sum_mask = step_probs_sum == 0
            if zero_sum_mask.any():
                step_probs[zero_sum_mask.expand(-1, -1, S).bool() & (torch.arange(S).to(args.device) < M).unsqueeze(0).unsqueeze(0).expand(B, D, S)] = 1/M
            # print(step_probs[zero_sum_mask.expand(-1, -1, S)])
            xt = Categorical(step_probs).sample() # (B, D)
            if torch.any(xt == M):
                num_masked_entries = torch.sum(xt == M).item()
                print(f"Forcing failed. Number of masked entries in the final tensor: {num_masked_entries}")

    return xt.detach().cpu().numpy()


def compute_loss(model, xt, x1, t, args, x_noise=None):
    B, D = args.batch_size, args.discrete_dim
    S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
    if args.source == 'mask':
        M = S - 1

    x1_logits, noise_logits = model(xt, t)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1)
    if args.scheduler_noise:
        noise_loss = F.cross_entropy(noise_logits.transpose(1,2), x_noise, reduction='none').sum(dim=-1)
        loss = loss + noise_loss

    x_hat = torch.argmax(x1_logits, dim=-1)
    acc = (x_hat == x1).float().mean().item()

    return loss.mean(dim=0), acc

def main_loop(args, verbose=False):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    my_print = args.my_print

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    # load omniglot
    if args.source == 'omniglot':
        og_args = copy.deepcopy(args)
        og_args.dataset_name == 'omniglot'
        og_train_loader, og_val_loader, og_test_loader, args = vamp_utils.load_dataset(args)
        #can use the same plot function...

    def preprocess(data, args=args):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
            
    def get_independent_sample(loader, args=args):
        (x, _) = next(iter(loader))
        return preprocess(x, args)

    if args.source == 'omniglot':
        source_train_loader = copy.deepcopy(og_train_loader)
    else:
        source_train_loader = copy.deepcopy(train_loader)
        
    # make model
    model = ResNetFlow(64, args)
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # move to cuda
    model.to(args.device)
    ema_model.to(args.device)

    start_time = time.time()
    cum_eval_time = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        pbar = tqdm(train_loader) if verbose else train_loader

        for it, ((x, _), (x_source, _)) in enumerate(zip(pbar, source_train_loader)):
            
            x1 = preprocess(x).long().to(args.device)            

            (B, D) = x1.shape

            S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
            if args.source == 'mask':
                M = S - 1
                x0 = torch.ones((B,D)).to(args.device).long() * M
            elif args.source == 'uniform':
                x0 = torch.randint(0, S, (B, D)).to(args.device)
            elif args.source == 'data':
                x0 = preprocess(x_source).long().to(args.device)
            elif args.source == 'omniglot':
                x0 = preprocess(x_source, args=og_args).long().to(args.device)     

            if args.scheduler_type == 'quadratic_noise':
                x_noise = torch.randint(0, S, (B, D)).to(args.device)

            t = torch.rand((B,)).to(args.device)

            if args.scheduler_type == 'linear':
                kappa1 = t
            elif args.scheduler_type == 'quadratic':
                kappa1 = torch.square(t)
            elif args.scheduler_type == 'quadratic_noise':
                kappa1 = torch.square(t)
                kappa2 = t - torch.square(t)

            xt = x1.clone()
            mask0 = torch.rand((B,D)).to(args.device) < (1 - kappa1[:, None])
            xt[mask0] = x0[mask0]
            if args.scheduler_type == 'quadratic_noise':
                mask_noise = torch.rand((B,D)).to(args.device) < (kappa2/(1 - kappa1))[:, None]
                mask_noise = mask_noise & mask0
                xt[mask_noise] = x_noise[mask_noise]
                loss, acc = compute_loss(model, xt, x1, t, args, x_noise)
            else:
                loss, acc = compute_loss(model, xt, x1, t, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}, Acc {acc}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            torch.save(model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')

            log_entry = {'epoch':None,'timestamp':None}
            
            log_entry['loss'] = loss.item()
            log_entry['acc'] = acc
            torch.save(ema_model.state_dict(), f'{args.ckpt_path}/ema_model_{epoch}.pt')

            if args.source == 'data':
                xt = get_independent_sample(test_loader).long().to(args.device) 
                plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
            elif args.source == 'omniglot':
                xt = get_independent_sample(og_test_loader, args=og_args).long().to(args.device) 
                plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
            else:
                xt = None

            samples = gen_samples(model, args, batch_size=args.batch_size, xt=xt)
            plot(f'{args.sample_path}/samples_{epoch}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_model, args, batch_size=100, xt=xt)
            plot(f'{args.sample_path}/ema_samples_{epoch}.png', torch.tensor(ema_samples).float())
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)
    log_completion(args.methods, args.dataset_name, args)

