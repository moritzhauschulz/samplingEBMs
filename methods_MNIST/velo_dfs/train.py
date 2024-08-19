import os
import copy
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import utils.utils as utils
from utils.utils import get_batch_data
from utils.utils import plot as toy_plot
from utils.utils import get_x0
import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion
from utils.eval import get_eval_timestamp
from utils.eval import exp_hamming_mmd
from utils.eval import rbf_mmd
from utils.toy_data_lib import get_db


from utils.model import ResNetFlow
from utils.model import MLPModel

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
        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            ut = model(xt, t_)
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 
        if args.scheduler_type == 'linear':

            #adaptive dt
            if t>0:
                dt = min(args.delta_t, (1 - t))
            else:
                dt = args.delta_t
        elif args.scheduler_type == 'quadratic':

            #adaptive dt
            if t>0:
                dt = min(args.delta_t, (1 - t ** 2)/(2 * t))
            else:
                dt = args.delta_t
        elif args.scheduler_type == 'quadratic_noise':
            #adaptive dt 
            if t>0:
                dt = min(args.delta_t, 1 - t)
            else:
                dt = args.delta_t

        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            ut, _ = model(xt, t_)



        step_probs = delta_xt + (ut * dt)

        if args.impute_self_connections:
            step_probs = step_probs.clamp(max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 

        t += dt

        step_probs = step_probs.clamp(min=0)


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

def compute_loss(model,B,D,S,t,x1,x0,args):
    if args.source == 'mask':
        M = S - 1

    if args.scheduler_type == 'quadratic_noise':
        x_noise = torch.randint(0, S, (B, D)).to(args.device)
    else:
        x_noise = None
    
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

    delta_xt = torch.zeros((B,D,S)).to(args.device)
    delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 

    delta_x1 = torch.zeros((B,D,S)).to(args.device)
    delta_x1 = delta_x1.scatter_(-1, x1[:, :, None], 1.0)

    w_noise = torch.ones((B,D,S)).to(args.device) * 1/S

    t_ = t.unsqueeze(-1).unsqueeze(-1).expand((B,D,S))

    ut, _ = model(xt, t)
    ut = ut * (args.loss_weight * (1 - t[:, None, None]) + (1 - args.loss_weight))

    if args.scheduler_type == 'linear':
        a1 = 1 / (1 - t)
        b = -a1
    elif args.scheduler_type == 'quadratic':
        a1 = (2 * t) / (1 - t ** 2)
        b = -a1
    elif args.scheduler_type == 'quadratic_noise':
        a1 = t * (2 - t)/(1 - t)
        a2 = 1 - t
        b = -1 /(1 - t)
    
    if args.scheduler_type == 'quadratic_noise':
        ut_target = a1[:, None, None] * delta_x1 + a2[:, None, None] * w_noise + b[:, None, None] * delta_xt #a3 is zero...
    else:
        ut_target = a1[:, None, None] * delta_x1 + b[:, None, None] * delta_xt

    ut_target =  ut_target * (args.loss_weight * (1 - t[:, None, None]) + (1 - args.loss_weight))

    loss = (ut - ut_target).square()
    if args.impute_self_connections:
        loss.scatter_(-1, xt[:, :, None], 0.0)
    loss = loss.sum(dim=(1,2)).mean(dim=0)

    return loss

def main_loop(args, verbose=False):

    #set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.is_toy:
        main_loop_toy(args, verbose)
    else:
        main_loop_real(args, verbose)
    log_completion(args.methods, args.dataset_name, args)

def main_loop_real(args, verbose=False):

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                    p, normalize=True, nrow=int(x.size(0) ** .5))
    # load omniglot
    if args.source == 'omniglot':
        og_args = copy.deepcopy(args)
        og_args.dataset_name == 'omniglot'
        og_train_loader, og_val_loader, og_test_loader, og_args = vamp_utils.load_dataset(og_args)
        source_train_loader = copy.deepcopy(og_train_loader)
        #can use the same plot function...
    else:
        source_train_loader = copy.deepcopy(train_loader)

    def preprocess(data, args=args):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
            
    def get_independent_sample(loader, args=args):
        (x, _) = next(iter(loader))
        return preprocess(x, args)

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
            t = torch.rand((B,)).to(args.device)

            if args.source == 'data':
                x0 = preprocess(x_source).long().to(args.device)
            elif args.source == 'omniglot':
                x0 = preprocess(x_source, args=og_args).long().to(args.device)
            else:
                x0 = get_x0(B,D,S,args)

            loss = compute_loss(model,B,D,S,t,x1,x0,args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            #save models
            torch.save(model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')
            torch.save(ema_model.state_dict(), f'{args.ckpt_path}/ema_model_{epoch}.pt')

            #save samples
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

            #save log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['loss'] = loss.item()
            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, epoch, timestamp)

def main_loop_toy(args, verbose=False):

    # load data
    db = get_db(args)

    plot = lambda p, x: toy_plot(p, x, args)

    # make model
    model = MLPModel(args).to(args.device)
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # move to cuda
    model.to(args.device)
    ema_model.to(args.device)

    start_time = time.time()
    cum_eval_time = 0

    pbar = tqdm(range(1,args.num_epochs + 1)) if verbose else range(1,args.num_epochs + 1)
    for epoch in pbar:
        model.train()

        x1 = torch.from_numpy(get_batch_data(db, args)).to(args.device)          

        (B, D) = x1.shape
        S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
        
        t = torch.rand((B,)).to(args.device)

        if args.source == 'data':
            x0 = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
        else:
            x0 = get_x0(B,D,S,args)

        loss = compute_loss(model,B,D,S,t,x1,x0,args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update ema_model
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

        if verbose:
            pbar.set_description(f'Epoch {epoch} Loss {loss.item()}')

        if (epoch % args.plot_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            #save samples
            if args.source == 'data':
                xt = torch.from_numpy(get_batch_data(db, args, batch_size = 2500)).to(args.device)
                plot(f'{args.sample_path}/source_{epoch}.png', xt)
            else:
                xt = None
            samples = gen_samples(model, args, batch_size = 2500, xt=xt)
            plot(f'{args.sample_path}/samples_{epoch}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_model, args, batch_size=2500, xt=xt)
            plot(f'{args.sample_path}/ema_samples_{epoch}.png', torch.tensor(ema_samples).float())
            _, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)
        if (epoch % args.eval_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            #save models
            torch.save(model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')
            torch.save(ema_model.state_dict(), f'{args.ckpt_path}/ema_model_{epoch}.pt')

            #compute mmds
            exp_hamming_mmd_list = []
            rbf_mmd_list = []
            for _ in range(10):
                if args.source == 'data':
                    xt = torch.from_numpy(get_batch_data(db, args, batch_size = 2500)).to(args.device)
                    plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
                else:
                    xt = None
                x = torch.from_numpy(gen_samples(model, args, batch_size = 4000, xt=xt)).to('cpu')
                y = get_batch_data(db, args, batch_size=4000)
                y = torch.from_numpy(np.float32(y)).to('cpu')
                hamming_mmd, bandwidth = exp_hamming_mmd(x,y,args)
                euclidean_mmd, sigma = rbf_mmd(x,y,args)
                exp_hamming_mmd_list.append(hamming_mmd)
                rbf_mmd_list.append(euclidean_mmd)
            hamming_mmd = sum(exp_hamming_mmd_list)/10
            euclidean_mmd = sum(rbf_mmd_list)/10

            #log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['sampler_hamming_mmd'], log_entry['bandwidth'] = hamming_mmd.item(), bandwidth.item()
            log_entry['sampler_euclidean_mmd'], log_entry['sigma'] = euclidean_mmd.item(), sigma.item()
            log_entry['loss'] = loss.item()
            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)
            log(args, log_entry, epoch, timestamp)

