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
from mask_dfs_ce_2.model import MLPScore, EBM
from mask_dfs_ce_2.model import MLPModel as MLPFlow
from mask_dfs_ce_2.model import alt_MLPModel as alt_MLPFlow

from utils import sampler
from utils.eval import ebm_evaluation
from utils.eval import sampler_evaluation
from utils.eval import sampler_ebm_evaluation
from utils.utils import get_batch_data
from utils.eval import log
from utils.eval import log_completion
from utils.eval import make_plots

def make_sampler(model_path, args):
    sampler = alt_MLPFlow(args).to(args.device)
    try:
        sampler.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print('Specify a valid model checkpoint to load as and try again.')
        sys.exit(1)
    sampler.eval()
    return sampler

def gen_samples(model, args, batch_size=None, t=0.0, xt=None, eta=0):
    model.eval()
    S, D = args.vocab_size_with_mask, args.discrete_dim

    # Variables, B, D for batch size and number of dimensions respectively
    B = batch_size if batch_size is not None else args.batch_size

    # Level of stochasticity  
    M = S - 1

    # Initialize xt with the mask index value if not provided
    if xt is None:
        xt = M * torch.ones((B, D), dtype=torch.long).to(args.device)

    dt = args.delta_t  # Time step
    t = 0.0  # Initial time

    while t < 1.0:
        t_ = t * torch.ones((batch_size,)).to(args.device)
        with torch.no_grad():
            R = F.softmax(model(xt, t_), dim = -1) #NOTE: no softmax
        from_M_mask = (xt == M).unsqueeze(-1).expand_as(R)
        to_M_mask = torch.zeros_like(R).bool().to(args.device)
        to_M_mask[:,:,M] = True

        #set diagonal to negative of sum of remaining entries
        R[from_M_mask & ~to_M_mask] *= (1 + eta * t) * 1 /(1-t)
        R[to_M_mask & ~from_M_mask] *= eta
        R = R.scatter_(-1, xt[:,:,None], 0.0)
        R.scatter_(-1, xt[:, :, None], (-1 * R.sum(dim=-1, keepdim=True)))

        R_dt = R * dt

        diagonal_ones = torch.zeros_like(R_dt).to(args.device)
        diagonal_ones = diagonal_ones.scatter_(-1, xt[:, :, None], 1.0)
        step_probs = (diagonal_ones + R_dt).clamp(min=0, max=1) #clamping to avoid out of range values

        t += dt

        if t < 1.0:  # Don’t re-mask on the final step
            step_probs_sum = step_probs.sum(dim=-1, keepdim=True)
            non_zero_mask = (step_probs_sum != 0).expand_as(step_probs)
            step_probs[non_zero_mask] = step_probs[non_zero_mask] / step_probs_sum.expand_as(step_probs)[non_zero_mask]
            if torch.any(non_zero_mask == False):
                print(f'Careful: Had to introduce {torch.sum(~non_zero_mask)} new unifrom probabilities due to zero sum – could not normalize')
                step_probs[~non_zero_mask] = 1/S
            xt = Categorical(step_probs).sample() # (B, D)
        else:
            if torch.any(xt == M):
                num_masked_entries = torch.sum(xt == M).item()
                print(f"Share of masked entries in the final tensor: {num_masked_entries / (B * D)}")
                print(f"Forcing mask values into range...")
            step_probs[:, :, M] = 0
            step_probs_sum = step_probs.sum(dim=-1, keepdim=True)
            non_zero_mask = (step_probs_sum != 0).expand_as(step_probs)
            step_probs[non_zero_mask] = step_probs[non_zero_mask] / step_probs_sum.expand_as(step_probs)[non_zero_mask]
            zero_sum_mask = step_probs_sum == 0
            if zero_sum_mask.any():
                print(f'Careful: Had to introduce {torch.sum(~non_zero_mask)} new unifrom probabilities in final step due to zero sum – could not normalize')
                step_probs[zero_sum_mask.expand(-1, -1, S) & (torch.arange(S).to(args.device) < 2).unsqueeze(0).unsqueeze(0)] = 0.5

            xt = Categorical(step_probs).sample() # (B, D)

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
    (B, D), S = x1.size(), args.vocab_size_with_mask
    M = S - 1

    x1_logits = dfs_model(xt,t)
    with torch.no_grad():
        energy = torch.exp(-ebm_model(x1.float()))
    q_density = q_dist.log_prob(x1.float()).sum(dim=-1).exp()
    weights = energy/q_density

    x1[xt != M] = -1
    star_loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none', ignore_index = -1) #where x_t=M
    M_target = torch.ones((B, D), dtype=torch.long).to(args.device) * M
    M_target[xt == M] = -1
    db_loss = F.cross_entropy(x1_logits.transpose(1,2), M_target, reduction='none', ignore_index = -1).float() #where x_t!=M
    loss = (star_loss + db_loss) * weights.unsqueeze(-1).expand((B,D))

    return loss.mean()

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

    dfs_model = alt_MLPFlow(args).to(args.device)
    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.dfs_lr)

    pbar = tqdm(range(1,args.num_epochs + 1)) if verbose else range(1,args.num_epochs + 1)

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

            x1 = q_dist.sample((args.batch_size,)).long() #remember that there is no data available under the assumptions

            (B, D), S = x1.size(), args.vocab_size_with_mask
            t = torch.rand((B,)).to(args.device)
            xt = x1.clone()
            xt[torch.rand((B,D)).to(args.device) < (1 - t[:, None])] = S - 1
            
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

            for eta in args.eta_list:
                log_entry[f'sampler_mmd_eta-{eta}'], log_entry['bandwidth'] = sampler_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x,eta=eta)))
                log_entry[f'sampler_ebm_mmd_eta-{eta}'], log_entry['bandwidth'] = sampler_ebm_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x, eta=eta)), ebm_model)
                log_entry['loss'] = loss.item()


            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)
        
        if (epoch % args.plot_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            for eta in args.eta_list:
                samples = gen_samples(dfs_model, args, batch_size=2500, eta=eta)
                if args.vocab_size == 2:
                    float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
                else:
                    float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
                utils.plot_samples(float_samples, f'{args.sample_path}/sample_{epoch}_eta-{eta}.png', im_size=4.1, im_fmt='png')

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time

        if verbose:
            pbar.set_description(f'Epoch {epoch}')
        elif (epoch % args.plot_every == 0) or (epoch == args.num_epochs):
            print(f'Epoch is {epoch} with loss at {loss}')


    make_plots(args.log_path)
    log_completion(args.methods, args.data_name, args)


