import os
import sys
import copy
import torch
import utils.mlp as mlp
import numpy as np
import torchvision
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils.eval import plot_weight_histogram

import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion

from velo_mask_edfs.model import ResNetFlow
from velo_mask_edfs.model import EBM
from velo_mask_edfs.model import Dataq


def gen_samples(model, args, batch_size=None, t=0.0, xt=None):
    model.eval()
    S, D = args.vocab_size_with_mask, args.discrete_dim

    # Variables, B, D for batch size and number of dimensions respectively
    B = batch_size if batch_size is not None else args.batch_size

    M = S - 1

    # Initialize xt with the mask index value if not provided
    if xt is None:
        xt = M * torch.ones((B, D), dtype=torch.long).to(args.device)


    dt = args.delta_t  # Time step
    t = 0.0  # Initial time

    while t < 1.0:
        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            ut = model(xt, t_)
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 

        step_probs = delta_xt + (ut * dt)

        if args.impute_self_connections:
            step_probs = step_probs.clamp(max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 

        t += dt

        step_probs = step_probs.clamp(min=0)


        if t < 1.0:
            xt = Categorical(step_probs).sample() #(B,D)
        else:
            print(f'final t at {t}')
            if torch.any(xt == M):
                num_masked_entries = torch.sum(xt == M).item()
                print(f"Number of masked entries in the final but one tensor: {num_masked_entries}")
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

def compute_loss(ebm_model, temp, dfs_model, q_dist, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size_with_mask
    M = S - 1

    delta_xt = torch.zeros((B,D,S)).to(args.device)
    delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 

    delta_x1 = torch.zeros((B,D,S)).to(args.device)
    delta_x1 = delta_x1.scatter_(-1, x1[:, :, None], 1.0)

    t_ = t.unsqueeze(-1).unsqueeze(-1).expand((B,D,S))

    ut = dfs_model(xt, t)  * (args.loss_weight * (1 - t[:, None, None]) + (1 - args.loss_weight))
    ut_target =  1/(1-t_) * (delta_x1 - delta_xt) * (args.loss_weight * (1 - t[:, None, None]) + (1 - args.loss_weight))

    loss = (ut - ut_target).square()
    if args.impute_self_connections:
        loss.scatter_(-1, xt[:, :, None], 0.0)
    loss = loss.sum(dim=(1,2))

    log_prob = -ebm_model(x1.float()) / temp
    # print(f'max is {log_prob.max().item()}')
    # print(f'min is {log_prob.min().item()}')
    log_q_density = q_dist.get_last_log_likelihood()
    log_weights = log_prob - log_q_density
    if args.norm_by_sum and not args.norm_by_max:
        log_norm = torch.logsumexp(log_weights, dim=-1) #make this a moving average?
    elif args.norm_by_max and not args.norm_by_sum:
        log_norm = torch.max(log_weights)
    else:
        raise NotImplementedError('Must either normalize by sum or max to avoid inf.')
    # print(f'\n log norm is {log_norm} \n')
    weights = (log_weights - log_norm).exp()
    loss = weights * loss #the math is wrong?
    loss = loss.mean(dim=0)

    return loss, weights #can we have accuracy here?


def main_loop(args, verbose=False):
    my_print = args.my_print

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    #load data for q distribution
    train_set, _, _ = vamp_utils.load_static_mnist(args, return_datasets=True)


    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = (init_batch.mean(0) * (1. - 2 * eps) + eps).to(args.device)
    q_dist = Dataq(args, train_set, bernoulli_mean=init_mean)

    # make dfs model
    dfs_model = ResNetFlow(64, args)
    ema_dfs_model = copy.deepcopy(dfs_model)
    
    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # make ebm model
    if args.ebm_model.startswith("mlp-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.mlp_ebm(np.prod(args.input_size), nint)
    elif args.ebm_model.startswith("resnet-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    elif args.ebm_model.startswith("cnn-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.MNISTConvNet(nint)
    else:
        raise ValueError("invalid ebm_model definition")

    ebm_model = EBM(net, init_mean).to(args.device)
    try:
        d = torch.load(args.pretrained_ebm)
        ebm_model.load_state_dict(d['ema_model'])
        print(f'successfully loaded EBM...')
    except FileNotFoundError as e:
        print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
        sys.exit(1)
    ebm_model.eval()

    # move to cuda
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)
    ebm_model.to(args.device)

    #set temperature
    temp = args.start_temp
    temp_decay = args.end_temp/args.start_temp ** (1/args.num_epochs)

    start_time = time.time()
    cum_eval_time = 0

    for epoch in range(args.num_epochs):
        
        dfs_model.train()
        pbar = tqdm(q_dist.loader) if verbose else q_dist.loader
    
        for it, (x, _) in enumerate(pbar):
            # x, _ = q_dist.loader
            empirical_samples = preprocess(x).long().to(args.device)
            x1 = q_dist.sample(empirical_samples).to(args.device).long()

            (B, D), S = x1.size(), args.vocab_size_with_mask
            M = S - 1
            x0 = torch.ones((B,D)).to(args.device).long() * M

            t = torch.rand((B,)).to(args.device)
            xt = x1.clone()
            mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
            xt[mask] = x0[mask]

            loss, weights = compute_loss(ebm_model, temp, dfs_model, q_dist, xt, x1, t, args)
            dfs_optimizer.zero_grad()
            loss.backward()
            dfs_optimizer.step()

            # update ema_model
            for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            eval_start_time = time.time()

            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')

            log_entry = {'epoch':None,'timestamp':None}
            
            log_entry['loss'] = loss.item()
            torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{epoch}.pt')

            plot(f'{args.sample_path}/q_samples_{epoch}_first_ten_types_{q_dist.get_last_is_empirical()[0:10].cpu().numpy()}.png', torch.tensor(x1).float())
            samples = gen_samples(dfs_model, args, 100)
            plot(f'{args.sample_path}/samples_{epoch}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_dfs_model, args, 100)
            plot(f'{args.sample_path}/ema_samples_{epoch}.png', torch.tensor(ema_samples).float())
            eval_end_time = time.time()
            output_dir = f'{args.plot_path}/weights_histogram_{epoch}.png'
            if not os.path.exists(output_dir):
                plot_weight_histogram(weights, output_dir=output_dir)
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)
        
        temp = temp * temp_decay
    log_completion(args.methods, args.dataset_name, args)

