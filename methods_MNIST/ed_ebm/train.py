import os
import torch
import shutil
import time
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

from utils import utils
from utils.eval import ebm_evaluation, log_completion, log
from utils.toy_data_lib import get_db
from utils import eval
from ed_ebm.model import MLPScore, EBM
from ed_ebm.ed_categorical import ed_categorical

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def energy_discrepancy_bernoulli(energy_net, samples, epsilon=0.1, m_particles=32, w_stable=1.):
    device = samples.device
    bs, dim = samples.shape

    noise_dist = dists.Bernoulli(probs=epsilon * torch.ones((dim,)).to(device))
    beri = (noise_dist.sample((bs,)) + samples) % 2.    # [bs, dim]
    pert_data = (noise_dist.sample((bs * m_particles,)).view(bs, m_particles, dim) + beri.unsqueeze(1)) % 2.    # [bs, m_particles, dim]
    # print((samples.unsqueeze(1) - pert_data).abs().mean())

    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(pert_data.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss

def main_loop(args, verbose=False):
    start_time = time.time()

    db = get_db(args)
    #assert args.vocab_size == 2, 'Only support binary data'

    samples = get_batch_data(db, args, batch_size=50000)
    mean = np.mean(samples, axis=0)
    #init_dist = torch.distributions.Bernoulli(probs=torch.from_numpy(mean).to(args.device) * (1. - 2 * 1e-2) + 1e-2)
    
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    #model = EBM(net, torch.from_numpy(mean)).to(args.device)
    model = EBM(net).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    pbar = tqdm(range(1,args.num_epochs + 1)) if verbose else range(1,args.num_epochs + 1)

    start_time = time.time()
    cum_eval_time = 0
    for epoch in pbar:
        model.train()

        samples = get_batch_data(db, args)
        samples = torch.from_numpy(np.float32(samples)).to(args.device)

        if args.vocab_size == 2:
            loss = energy_discrepancy_bernoulli(model, samples) #note that this is not exactly contrastive divergence
        else:
            loss = ed_categorical(model, samples, K=args.vocab_size)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        if verbose:
            pbar.set_description(f'Epoch {epoch} Loss {loss.item()}')

        if (epoch % args.plot_every == 0) or (epoch == args.num_epochs):

            if args.vocab_size == 2:
                utils.plot_heat(model, db.f_scale, args.bm, f'{args.plot_path}/heat_{epoch}.png', args)
                utils.plot_sampler(model, f'{args.sample_path}/samples_{epoch}.png', args)

        if (epoch % args.eval_every == 0) or (epoch == args.num_epochs):
            torch.save(model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')


            if epoch < args.num_epochs:
                ais_samples = args.intermediate_ais_samples
                ais_num_steps = args.intermediate_ais_num_steps
            else: 
                ais_samples =  args.final_ais_samples
                ais_num_steps = args.final_ais_num_steps

            eval_start_time = time.time()
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['ebm_nll'], log_entry['ebm_hamming_mmd'], log_entry['bandwidth'], log_entry['ebm_euclidean_mmd'], log_entry['sigma'] = ebm_evaluation(args, db, model, batch_size=4000, ais_samples=args.final_ais_samples, ais_num_steps=args.final_ais_num_steps)
            
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)
    log_completion(args.methods, args.dataset_name, args)