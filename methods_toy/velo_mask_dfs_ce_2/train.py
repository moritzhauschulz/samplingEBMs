import sys
import os
import json
import torch
import shutil
import random
import numpy as np
from tqdm import tqdm
import torch.distributions as dists
from torch.distributions.categorical import Categorical
import pickle
import time
import torch.nn.functional as F

from utils import utils
from velo_mask_dfs_ce_2.model import MLPScore, EBM
from velo_mask_dfs_ce_2.model import MLPModel as MLPFlow
from utils import sampler
from utils.eval import ebm_evaluation
from utils.eval import sampler_evaluation
from utils.eval import sampler_ebm_evaluation
from utils.utils import get_batch_data
from utils.eval import log
from utils.eval import log_completion
from utils.eval import make_plots

def make_sampler(model_path, args):
    sampler = MLPFlow(args).to(args.device)
    try:
        sampler.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print('Specify a valid model checkpoint to load as and try again.')
        sys.exit(1)
    sampler.eval()
    return sampler


def gen_samples(model, args, batch_size=None, t=0.0, xt=None):
    model.eval()
    S, D = args.vocab_size_with_mask, args.discrete_dim

    # Variables, B, D for batch size and number of dimensions respectively
    B = batch_size if batch_size is not None else args.batch_size

    M = S - 1

    # Initialize xt with the mask index value if not provided
    if xt is None:
        xt = M * torch.ones((B, D), dtype=torch.long).to(args.device)
    
    forced_unmask = torch.zeros((B,)).to(args.device)

    dt = args.delta_t  # Time step
    t = 0.0  # Initial time

    while t < 1.0:
        t_ = t * torch.ones((B,)).to(args.device)

        with torch.no_grad():
            x1_logits = F.softmax(model(xt, t_), dim=-1)
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 
        ut = 1/(1-t) * (x1_logits - delta_xt)

        step_probs = delta_xt + (ut * dt)
        # step_probs = step_probs.scatter_(-1, xt[:, :, None], 0.0)
        # step_probs = step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0))

        t += dt

        if t < 1.0:
            xt = Categorical(step_probs).sample() #(B,D)
        else:
            print(f'final t at {t}')
            if torch.any(xt == M):
                num_masked_entries = torch.sum(xt == M).item()
                print(f"Number of masked entries in the final but one tensor: {num_masked_entries}")
                print(f"Forcing mask values into range...")
            forced_unmask[step_probs[:,:,M].sum(dim=-1)>0.001] = 1
            print(f'share forced {forced_unmask.sum()/B}')
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

    return xt.detach().cpu().numpy(), forced_unmask

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def compute_loss(ebm_model, dfs_model, q_dist, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size_with_mask
    M = S - 1

    t_expanded = t[:, None, None].expand(-1, D, S)

    x1_logits = dfs_model(xt, t)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1)

    energy = torch.exp(-ebm_model(x1.float()))
    q_density = q_dist.log_prob(x1.float()).sum(dim=-1).exp()

    loss = (energy/q_density * loss).mean(dim=0)

    return loss

def main_loop(db, args, verbose=False):
    assert args.vocab_size == 2, 'Only support binary data'

    samples = get_batch_data(db, args, batch_size=10000)
    mean = np.mean(samples, axis=0)
    q_dist = torch.distributions.Bernoulli(probs=torch.from_numpy(mean).to(args.device) * (1. - 2 * 1e-2) + 1e-2)
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    #ebm_model = EBM(net, torch.from_numpy(mean)).to(args.device)
    
    ebm_model = EBM(net).to(args.device)
    # idx = input("Please enter the experiment number for the energy discrepancy ebm to use (e.g. 0): ")
    # check = input("Please enter checkpoint number to load (normally 100000): ")
    try:
        ebm_model.load_state_dict(torch.load(f'./{args.pretrained_ebm}'))
    except FileNotFoundError as e:
        print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
        sys.exit(1)
    ebm_model.eval()
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/initial_heat.png', args)

    dfs_model = MLPFlow(args).to(args.device)
    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.dfs_lr)

    pbar = tqdm(range(1,args.num_epochs + 1))

    start_time = time.time()
    cum_eval_time = 0

    for epoch in pbar:
        dfs_model.train()
        dfs_pbar = tqdm(range(args.surrogate_iter_per_epoch)) if verbose else range(args.surrogate_iter_per_epoch)
        
        for it in dfs_pbar:
           
            #TO BE IMPLEMENTED
            #get (x0, x1) coupling
            x1 = q_dist.sample((args.batch_size,)).long()
            (B, D), S = x1.size(), args.vocab_size_with_mask
            M = S - 1
            x0 = torch.ones((B,D)).to(args.device).long() * M

            #sample t
            t = torch.rand((B,)).to(args.device)

            #sample xt conditional on x0, x1 (here: convex interpolant with linear scheduler)
            xt = x1.clone()
            mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
            xt[mask] = x0[mask]
            
            #compute energy conditional velocity matching loss
            loss = compute_loss(ebm_model, dfs_model, q_dist, xt, x1, t, args)

            dfs_optimizer.zero_grad()
            loss.backward()
            dfs_optimizer.step()

            if verbose:
                dfs_pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.eval_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()
            log_entry = {'epoch':None,'timestamp':None}

            ebm_model.eval()
            log_entry['sampler_hamming_mmd'], log_entry['bandwidth'], log_entry['sampler_euclidean_mmd'], log_entry['sigma'] = sampler_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x)[0]))
            log_entry['sampler_ebm_hamming_mmd'], log_entry['bandwidth'],log_entry['sampler_ebm_euclidean_mmd'], log_entry['sigma'] = sampler_ebm_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x)[0]), ebm_model)
            log_entry['loss'] = loss.item()
        
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)
        
        if (epoch % args.plot_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            samples, forced_unmask = gen_samples(dfs_model, args, batch_size=2500)
            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{args.sample_path}/sample_{epoch}.png', im_size=4.1, im_fmt='png',highlighted=forced_unmask.bool().cpu().numpy())

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time

        pbar.set_description(f'Epoch {epoch}')


    make_plots(args.log_path)
    log_completion(args.methods, args.data_name, args)


