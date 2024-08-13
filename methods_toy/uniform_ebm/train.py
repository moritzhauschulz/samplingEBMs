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
from cd_runi_inter.model import MLPScore, EBM
from cd_runi_inter.model import MLPModel as MLPFlow

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

def main_loop(db, args, verbose=False):
    assert args.vocab_size == 2, 'Only support binary data'

    samples = get_batch_data(db, args, batch_size=10000)
    mean = np.mean(samples, axis=0)
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    #ebm_model = EBM(net, torch.from_numpy(mean)).to(args.device)
    ebm_model = EBM(net).to(args.device)
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/initial_heat.png', args)

    ebm_optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.ebm_lr)

    pbar = tqdm(range(1,args.num_epochs + 1))

    start_time = time.time()
    cum_eval_time = 0

    for epoch in pbar:
        
        #ebm training
        ebm_model.train()

        ebm_pbar = tqdm(range(args.ebm_iter_per_epoch)) if verbose else range(args.ebm_iter_per_epoch)

        for it in ebm_pbar:
            data_samples = get_batch_data(db, args)
            data_samples = torch.from_numpy(np.float32(data_samples)).to(args.device)
            
            #sample from uniform
            shape = data_samples.shape
            random_tensor = torch.rand(shape)
            model_samples = (random_tensor > 0.5).int().to(args.device)

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
            log_entry['ebm_nll'], log_entry['ebm_mmd'], log_entry['bandwidth'] = ebm_evaluation(args, db, ebm_model, batch_size=4000, ais_samples=ais_samples, ais_num_steps=ais_num_steps) #batch_size=4000, ais_samples=1000000, ais_num_intermediate=100

            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}ebm_model_{epoch}.pt')

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
            
            random_tensor = torch.rand([2500, args.discrete_dim])
            samples = (random_tensor > 0.5).int().to(args.device).detach().cpu().numpy()



            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{args.sample_path}uni_sample_{epoch}.png', im_size=4.1, im_fmt='png')

            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}ebm_model_{epoch}.pt')

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time

        pbar.set_description(f'Epoch {epoch}')

    make_plots(args.log_path)
    log_completion(args.methods, args.data_name, args)


 
        




            

