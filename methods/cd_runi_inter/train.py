import sys
import os
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.distributions as dists
from torch.distributions.categorical import Categorical
import pickle

from utils import utils
from methods.cd_runi_inter.model import MLPScore, EBM
from methods.cd_runi_inter.model import MLPModel as MLPFlow
from utils import sampler

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def gen_samples(model, args, batch_size=None):
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    t = 0.0
    dt = args.delta_t
    if batch_size is None:
        batch_size = args.batch_size
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

def compute_loss(ebm_model, dfs_model, q_dist, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size

    R_star = torch.zeros((B, D, S)).to(args.device)
    R_star = R_star.scatter_(-1, x1[:, :, None], 1.0) / (1 - t[:, None, None])
    R_star[xt == x1] = 0.0

    eta = 1.0
    R_DB_1 = torch.zeros((B, D, S)).to(args.device)
    R_DB_1[xt == x1] = 1 * eta
    R_DB_2 = torch.zeros((B, D, S)).to(args.device)
    R_DB_2 = R_DB_2.scatter_(-1, x1[:, :, None], 1.0) * eta * ((S*t + 1 - t) / (1-t))[:, None, None]
    R_DB = R_DB_1 + R_DB_2

    R_true = (R_star + R_DB) * (1 - t[:, None, None])
    R_est = dfs_model(xt, t) * (1 - t[:, None, None])
    loss = (R_est - R_true).square()
    loss.scatter_(-1, xt[:, :, None], 0.0)
    loss = loss.sum(dim=(1,2))

    energy = torch.exp(-ebm_model(x1.float()))
    q_density = q_dist.log_prob(x1.float()).sum(dim=-1).exp()
    loss = (energy / q_density * loss).mean(dim=0)
    
    return loss

def main_loop(db, args, verbose=False):

    samples = get_batch_data(db, args, batch_size=10000)
    mean = np.mean(samples, axis=0)
    q_dist = torch.distributions.Bernoulli(probs=torch.from_numpy(mean).to(args.device) * (1. - 2 * 1e-2) + 1e-2)
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    
    dfs_model = MLPFlow(args).to(args.device)
    #ebm_model = EBM(net, torch.from_numpy(mean)).to(args.device)
    ebm_model = EBM(net).to(args.device)
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.save_dir}/heat.pdf', args)

    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=1e-4)
    ebm_optimizer = torch.optim.Adam(ebm_model.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):

        #dfs training

        dfs_model.train()
        ebm_model.eval()

        dfs_pbar = tqdm(range(args.surrogate_iter_per_epoch)) if verbose else range(args.surrogate_iter_per_epoch)
        
        for it in dfs_pbar:
            x1 = q_dist.sample((args.batch_size,)).long().to(args.device) #remember that there is no data available under the assumptions

            (B, D), S = x1.size(), args.vocab_size
            t = torch.rand((B,)).to(args.device)
            xt = x1.clone()
            uniform_noise = torch.randint(0, S, (B, D)).to(args.device)
            corrupt_mask = torch.rand((B, D)).to(args.device) < (1 - t[:, None])
            xt[corrupt_mask] = uniform_noise[corrupt_mask]
            
            loss = compute_loss(ebm_model, dfs_model, q_dist, xt, x1, t, args) #basically fit dfs_model to ebm_model
            
            dfs_optimizer.zero_grad()
            loss.backward()
            dfs_optimizer.step()

            if verbose:
                dfs_pbar.set_description(f'Epoch {epoch} Iter {it} DFS Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}afs_model_{epoch}.pt')

            samples = gen_samples(dfs_model, args, batch_size=args.batch_size * 10)
            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{args.sample_path}dfs_sample_{epoch}.png', im_size=4.1, im_fmt='png')

        
        #ebm training
        dfs_model.eval()
        ebm_model.train()

        ebm_pbar = tqdm(range(args.ebm_iter_per_epoch)) if verbose else range(args.ebm_iter_per_epoch)

        for it in ebm_pbar:
            data_samples = get_batch_data(db, args)
            data_samples = torch.from_numpy(np.float32(data_samples)).to(args.device)
            
            model_samples = torch.from_numpy(gen_samples(dfs_model, args)).to(args.device)

            data_nrg = ebm_model(data_samples)
            model_nrg = ebm_model(model_samples)

            reg_loss = args.cd_alpha * (data_nrg ** 2 + model_nrg ** 2)
            cd_loss = data_nrg - model_nrg
            loss = (reg_loss + cd_loss).logsumexp(dim=-1).mean()

            ebm_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ebm_model.parameters(), max_norm=5)
            ebm_optimizer.step()     

            if verbose:
                ebm_pbar.set_description(f'Epoch {epoch} Iter {it} EBM Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):

            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}ebm_model_{epoch}.pt')

            if args.vocab_size == 2:
                utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}ebm_heat_{epoch}.pdf', args)
                utils.plot_sampler(ebm_model, f'{args.sample_path}ebm_samples_{epoch}.png', args)

 
        




            

