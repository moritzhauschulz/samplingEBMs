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

from utils import utils
from methods.cd_runi_inter.model import MLPScore, EBM
from methods.cd_runi_inter.model import MLPModel as MLPFlow
from methods.dataq_dfs.model import MixtureModel
from utils import sampler
from utils.eval import ebm_evaluation
from utils.eval import sampler_evaluation
from utils.eval import sampler_ebm_evaluation
from utils.utils import get_batch_data
from utils.eval import log
from utils.eval import log_completion
from utils.eval import make_plots

def gen_samples(model, args, batch_size=None, t=0.0, xt=None):
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    dt = args.delta_t
    if batch_size is None:
        batch_size = args.batch_size
    if xt is None:
        xt = torch.randint(0, S, (batch_size, D)).to(args.device)

    while t < 1.0:
        t_ = t * torch.ones((batch_size,)).to(args.device)
        with torch.no_grad():
            step_probs = model(xt, t_) * dt

        step_probs = step_probs.clamp(max=1.0)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)) 

        xt = Categorical(step_probs).sample() # (B, D)

        t += dt

    return xt.detach().cpu().numpy()

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
    (B, D), S = x1.size(), args.vocab_size

    R_star = torch.zeros((B, D, S)).to(args.device)
    R_star = R_star.scatter_(-1, x1[:, :, None], 1.0) / (1 - t[:, None, None])
    R_star[xt == x1] = 0.0

    
    R_DB_1 = torch.zeros((B, D, S)).to(args.device)
    R_DB_1[xt == x1] = 1 * args.eta
    R_DB_2 = torch.zeros((B, D, S)).to(args.device)
    R_DB_2 = R_DB_2.scatter_(-1, x1[:, :, None], 1.0) * args.eta * ((S*t + 1 - t) / (1-t))[:, None, None]
    R_DB = R_DB_1 + R_DB_2

    R_true = (R_star + R_DB) * (1 - t[:, None, None])
    R_est = dfs_model(xt, t) * (1 - t[:, None, None])
    loss = (R_est - R_true).square()
    loss.scatter_(-1, xt[:, :, None], 0.0)
    loss = loss.sum(dim=(1,2))

    energy = torch.exp(-ebm_model(x1.float()))
    q_density = q_dist.likelihood(x1.float())
    loss = (energy / q_density * loss).mean(dim=0)
    
    return loss

def main_loop(db, args, verbose=False):
    assert args.vocab_size == 2, 'Only support binary data'

    samples = get_batch_data(db, args, batch_size=10000)
    mean = np.mean(samples, axis=0)
    q_dist = MixtureModel(samples, mean, args.mixture, device=args.device)

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
            # x1_q = q_dist.sample((args.batch_size,)).long()
            # x1_p = torch.from_numpy(get_batch_data(db, args)).to(args.device)
            # mask = (torch.rand((args.batch_size,)).to(args.device) < 0.5).int().unsqueeze(1)
            # x1 = mask * x1_q + (1 - mask) * x1_p

            x1 = q_dist.sample(args.batch_size).long() 

            (B, D), S = x1.size(), args.vocab_size
            t = torch.rand((B,)).to(args.device)
            xt = x1.clone()
            uniform_noise = torch.randint(0, S, (B, D)).to(args.device)
            corrupt_mask = torch.rand((B, D)).to(args.device) < (1 - t[:, None])
            xt[corrupt_mask] = uniform_noise[corrupt_mask]
            
            loss = compute_loss(ebm_model, dfs_model, q_dist, xt, x1, t, args) #basically fit dfs_model to ebm model
            
            dfs_optimizer.zero_grad()
            loss.backward()
            dfs_optimizer.step()

            if verbose:
                dfs_pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.eval_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()
            log_entry = {'epoch':None,'timestamp':None}

            ebm_model.eval()
            log_entry['sampler_mmd'] = sampler_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x)))
            log_entry['sampler_ebm_mmd'] = sampler_ebm_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x)), ebm_model)
        
            samples = gen_samples(dfs_model, args, batch_size=args.batch_size * 10)
            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{args.sample_path}/sample_{epoch}.png', im_size=4.1, im_fmt='png')

            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)

        pbar.set_description(f'Epoch {epoch}')

    
    make_plots(args.log_path)
    log_completion(args.methods, args.data_name, args)


