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
    q_density = q_dist.log_prob(x1.float()).sum(dim=-1).exp()
    loss = (energy / q_density * loss).mean(dim=0)
    
    return loss

def main_loop(db, args, verbose=False):
    assert args.vocab_size == 2, 'Only support binary data'

    samples = get_batch_data(db, args, batch_size=10000)
    mean = np.mean(samples, axis=0)
    q_dist = torch.distributions.Bernoulli(probs=torch.from_numpy(mean).to(args.device) * (1. - 2 * 1e-2) + 1e-2)
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    
    dfs_model = MLPFlow(args).to(args.device)
    #ebm_model = EBM(net, torch.from_numpy(mean)).to(args.device)
    ebm_model = EBM(net).to(args.device)
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/initial_heat.png', args)

    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.dfs_lr)
    ebm_optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.ebm_lr)

    pbar = tqdm(range(1,args.num_epochs + 1))

    start_time = time.time()
    cum_eval_time = 0

    for epoch in pbar:

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

        
        #ebm training
        dfs_model.eval()
        ebm_model.train()

        ebm_pbar = tqdm(range(args.ebm_iter_per_epoch)) if verbose else range(args.ebm_iter_per_epoch)

        for it in ebm_pbar:
            data_samples = get_batch_data(db, args)
            data_samples = torch.from_numpy(np.float32(data_samples)).to(args.device)
            
            #sample from

            if args.rand_k or args.lin_k or (args.K > 0):
                if args.rand_k:
                    K = random.randrange(0,1/args.delta_t) + 1
                elif args.lin_k:
                    K = min(1/args.delta_t, int(1/args.delta_t * float(epoch + 1) / args.warmup_k))
                    K = max(K, 1)
                elif args.K > 0:
                    K = args.K
                else:
                    raise ValueError
                
                #delete this...
                if epoch % 500 == 1:
                    print(f'K is at {K}')
                    print(f'Corresponding t is {K * args.delta_t}')

                #go back via p(x_{1-K}|x_1)
                (B, D), S = data_samples.size(), args.vocab_size
                xt = data_samples.clone().long()
                uniform_noise = torch.randint(0, S, (B, D)).to(args.device)
                corrupt_mask = torch.rand((B, D)).to(args.device) < (K * args.delta_t)
                xt[corrupt_mask] = uniform_noise[corrupt_mask]

                #go forth via DFS starting at x_t
                model_samples = torch.from_numpy(gen_samples(dfs_model, args, t=K * args.delta_t, xt=xt)).to(args.device)
            else:
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


        if (epoch) % args.eval_every == 0 or epoch == args.num_epochs:
            eval_start_time = time.time()
            log_entry = {'epoch':None,'timestamp':None}

            if epoch < args.num_epochs:
                ais_samples = args.intermediate_ais_samples
                ais_num_steps = args.intermediate_ais_num_steps
            else: 
                ais_samples =  args.final_ais_samples
                ais_num_steps = args.final_ais_num_steps

            ebm_model.eval()
            log_entry['ebm_nll'], log_entry['ebm_mmd'] = ebm_evaluation(args, db, ebm_model, batch_size=4000, ais_samples=ais_samples, ais_num_steps=ais_num_steps) #batch_size=4000, ais_samples=1000000, ais_num_intermediate=100
            log_entry['sampler_mmd'] = sampler_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x)))
            log_entry['sampler_ebm_mmd'] = sampler_ebm_evaluation(args, db, lambda x: torch.from_numpy(gen_samples(dfs_model, args, batch_size=x)), ebm_model)
            
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}dfs_model_{epoch}.pt')
            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}ebm_model_{epoch}.pt')

            # if not os.path.exists(args, log):
            #     initialize_log_file()

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)

        if (epoch) % args.plot_every == 0 or epoch == args.num_epochs:
            eval_start_time = time.time()

            if args.vocab_size == 2:
                utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}ebm_heat_{epoch}.png', args)
                utils.plot_sampler(ebm_model, f'{args.sample_path}ebm_samples_{epoch}.png', args)
            
            samples = gen_samples(dfs_model, args, batch_size=2500)
            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{args.sample_path}dfs_sample_{epoch}.png', im_size=4.1, im_fmt='png')

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time

            

        pbar.set_description(f'Epoch {epoch}')

    make_plots(args.log_path)
    log_completion(args.methods, args.data_name, args)


 
        




            

