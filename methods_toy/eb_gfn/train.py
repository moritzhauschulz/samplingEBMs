import numpy as np
import torch
import torch.nn as nn
import torchvision
import os, sys

import time
import random
import ipdb, pdb
from tqdm import tqdm
import argparse

from eb_gfn.model import get_GFlowNet
from eb_gfn.model  import MLPScore, EBM

from utils import utils
from utils import sampler
from utils.eval import ebm_evaluation
from utils.eval import sampler_evaluation
from utils.eval import sampler_ebm_evaluation
from utils.eval import log
from utils.utils import get_batch_data
from utils.eval import log_completion
from utils.eval import make_plots


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")


unif_dist = torch.distributions.Bernoulli(probs=0.5)

def main_loop(db, args):

    assert args.vocab_size == 2, 'GFlowNet is only specified for binary data'
    assert args.discrete_dim == 32, 'GFlowNet is only specified for 32 dimensions'

    ############# Data
    if not hasattr(args, "int_scale"):
        int_scale = db.int_scale
    else:
        int_scale = args.int_scale
    if not hasattr(args, "plot_size"):
        plot_size = db.f_scale
    else:
        db.f_scale = args.plot_size
        plot_size = args.plot_size
    # plot_size = 4.1
    
    bm = args.bm
    inv_bm = args.inv_bm

    batch_size = args.batch_size
    # multiples = {'pinwheel': 5, '2spirals': 2}                                #not sure what this is for?
    # batch_size = batch_size - batch_size % multiples.get(args.data_name, 1)   #not sure what this is for? 

    ############## EBM model
    net = MLPScore(args.discrete_dim, [256] * 3 + [1], nonlinearity='swish').to(args.device)
    energy_model = EBM(net).to(args.device)
    utils.plot_heat(energy_model, db.f_scale, bm, f'{args.plot_path}/initial_heat.png', args)
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=args.lr)

    ############## GFN
    xdim = args.discrete_dim
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, args.device)

    print("model: {:}".format(energy_model))

    itr = 0
    best_val_ll = -np.inf
    best_itr = -1
    lr = args.lr

    start_time = time.time()
    cum_eval_time = 0

    while itr < args.n_iters:
        st = time.time()

        for _ in range(args.gfn_iter_per_epoch):
            x = get_batch_data(db, args)
            x = torch.from_numpy(np.float32(x)).to(args.device)
            update_success_rate = -1.
            gfn.model.train()
            train_loss, train_logZ = gfn.train(batch_size,
                    scorer=lambda inp: -1 * energy_model(inp).detach(), silent=itr % args.print_every != 0, data=x,
                    back_ratio=args.back_ratio)

        for _ in range(args.ebm_iter_per_epoch):
            x = get_batch_data(db, args)
            x = torch.from_numpy(np.float32(x)).to(args.device)
            if args.rand_k or args.lin_k or (args.K > 0):
                if args.rand_k:
                    K = random.randrange(xdim) + 1
                elif args.lin_k:
                    K = min(xdim, int(xdim * float(itr + 1) / args.warmup_k))
                    K = max(K, 1)
                elif args.K > 0:
                    K = args.K
                else:
                    raise ValueError

                gfn.model.eval()
                x_fake, delta_logp_traj = gfn.backforth_sample(x, K)

                delta_logp_traj = delta_logp_traj.detach()
                if args.with_mh:
                    # MH step, calculate log p(x') - log p(x)
                    lp_update = (-1 * energy_model(x_fake).squeeze()) - (-1 * energy_model(x).squeeze())
                    update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
                    updates = update_dist.sample()
                    x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
                    update_success_rate = updates.mean().item()
                else:
                    update_success_rate = None

            else:
                x_fake = gfn.sample(batch_size)


            # if itr % args.ebm_every == 0:
            st = time.time() - st

            energy_model.train()
            logp_real = -1 * energy_model(x).squeeze()

            logp_fake = -1 * energy_model(x_fake).squeeze()
            obj = logp_real.mean() - logp_fake.mean()
            l2_reg = (logp_real ** 2.).mean() + (logp_fake ** 2.).mean()
            loss = -obj

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if (itr % args.print_every == 0 or itr == args.n_iters - 1) and args.ebm_iter_per_epoch > 0:
        #     print("({:5d}) | ({:.3f}s/iter) cur lr= {:.2e} |log p(real)={:.2e}, "
        #              "log p(fake)={:.2e}, diff={:.2e}, update_rate={:.1f}".format(
        #         itr, st, lr, logp_real.mean().item(), logp_fake.mean().item(), obj.item(), update_success_rate))


        if (itr + 1) % args.eval_every == 0 or (itr + 1) == args.n_iters:
            eval_start_time = time.time()
            log_entry = {'epoch':None,'timestamp':None}

            if itr + 1 < args.n_iters:
                ais_samples = args.intermediate_ais_samples
                ais_num_steps = args.intermediate_ais_num_steps
            else: 
                ais_samples =  args.final_ais_samples
                ais_num_steps = args.final_ais_num_steps


            # GFN LL
            gfn.model.eval()
            logps = []
            pbar = tqdm(range(10))
            pbar.set_description("GFN Calculating likelihood")
            for _ in pbar:
                pos_samples_bs = get_batch_data(db, args)
                pos_samples_bs = torch.from_numpy(np.float32(pos_samples_bs)).to(args.device)
                logp = gfn.cal_logp(pos_samples_bs, 20)
                logps.append(logp.reshape(-1))
                pbar.set_postfix({"logp": f"{torch.cat(logps).mean().item():.2f}"})
            gfn_test_ll = torch.cat(logps).mean()
            print(f"Test NLL ({itr}): GFN: {-gfn_test_ll.item():.3f}")


            energy_model.eval()
            gfn.model.eval()
            log_entry['ebm_nll'], log_entry['ebm_mmd'], log_entry['bandwidth'] = ebm_evaluation(args, db, energy_model, batch_size=4000, ais_samples=ais_samples, ais_num_steps=ais_num_steps) # batch_size=4000, ais_samples=1000000, ais_num_intermediate=100
            log_entry['sampler_mmd'], log_entry['bandwidth'] = sampler_evaluation(args, db, lambda x: gfn.sample(x))
            log_entry['sampler_ebm_mmd'], log_entry['bandwidth'] = sampler_ebm_evaluation(args, db, lambda x: gfn.sample(x), energy_model)
            if not update_success_rate is None:
                log_entry['MH_update_success_rate'] = update_success_rate

            torch.save(energy_model.state_dict(), f'{args.ckpt_path}ebm_model_{itr + 1}.pt')
            torch.save(gfn.model.state_dict(), f'{args.ckpt_path}gfn_model_{itr + 1}.pt')
            
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, itr + 1, timestamp)

        if (itr + 1) % args.plot_every == 0 or (itr + 1) == args.n_iters:
            eval_start_time = time.time()
            
            if args.vocab_size == 2:
                utils.plot_heat(energy_model, db.f_scale, bm, f'{args.plot_path}ebm_heat_{itr + 1}.png', args)
                utils.plot_sampler(energy_model, f'{args.sample_path}ebm_samples_{itr + 1}.png', args)
                gfn_samples = gfn.sample(2500).detach()
                gfn_samp_float = utils.bin2float(gfn_samples.data.cpu().numpy().astype(int), inv_bm, args.discrete_dim, args.int_scale)
                utils.plot_samples(gfn_samp_float, f'{args.sample_path}gfn_samples_{itr + 1}.png', im_size=4.1, im_fmt='png')

            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time

        itr += 1

    make_plots(args.log_path)
    log_completion(args.methods, args.data_name, args)
