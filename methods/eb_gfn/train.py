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

from methods.eb_gfn.model import get_GFlowNet
from methods.eb_gfn.model  import MLPScore, EBM

from utils import utils
from utils import sampler

from synthetic_utils import plot_heat, plot_samples,\
    float2bin, bin2float, get_binmap, get_true_samples, get_ebm_samples, EnergyModel
from synthetic_data import inf_train_gen, OnlineToyDataset


# sys.path.append("/home/zhangdh/EB_GFN/synthetic")
# from synthetic_utils import plot_heat, plot_samples,\
#     float2bin, bin2float, get_binmap, get_true_samples, get_ebm_samples, EnergyModel
# from synthetic_data import inf_train_gen, OnlineToyDataset

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx


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

    ############## Data
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

    batch_size = args.batch_size
    multiples = {'pinwheel': 5, '2spirals': 2}
    batch_size = batch_size - batch_size % multiples.get(args.data_name, 1)

    ############## EBM model
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    energy_model = EBM(net).to(args.device)
    utils.plot_heat(energy_model, db.f_scale, args.bm, f'{args.save_dir}/heat.pdf', args)
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=1e-4)

    ############## GFN
    xdim = args.discrete_dim
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, args.device)

    energy_model.to(args.device)
    print("model: {:}".format(energy_model))

    itr = 0
    best_val_ll = -np.inf
    best_itr = -1
    lr = args.lr
    while itr < args.n_iters:
        st = time.time()

        x = get_batch_data(db, args)
        x = torch.from_numpy(np.float32(x)).to(args.device)
        # x = get_true_samples(db, batch_size, bm, int_scale, args.discrete_dim).to(args.device)

        update_success_rate = -1.
        gfn.model.train()
        train_loss, train_logZ = gfn.train(batch_size,
                scorer=lambda inp: energy_model(inp).detach(), silent=itr % args.print_every != 0, data=x,
                back_ratio=args.back_ratio)

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
                lp_update = energy_model(x_fake).squeeze() - energy_model(x).squeeze()
                update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
                updates = update_dist.sample()
                x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
                update_success_rate = updates.mean().item()

        else:
            x_fake = gfn.sample(batch_size)


        if itr % args.ebm_every == 0:
            st = time.time() - st

            energy_model.train()
            logp_real = energy_model(x).squeeze()

            logp_fake = energy_model(x_fake).squeeze()
            obj = logp_real.mean() - logp_fake.mean()
            l2_reg = (logp_real ** 2.).mean() + (logp_fake ** 2.).mean()
            loss = -obj

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if itr % args.print_every == 0 or itr == args.n_iters - 1:
            print("({:5d}) | ({:.3f}s/iter) cur lr= {:.2e} |log p(real)={:.2e}, "
                     "log p(fake)={:.2e}, diff={:.2e}, update_rate={:.1f}".format(
                itr, st, lr, logp_real.mean().item(), logp_fake.mean().item(), obj.item(), update_success_rate))


        if (itr + 1) % args.eval_every == 0:
            if args.vocab_size == 2:
                utils.plot_heat(energy_model, db.f_scale, args.bm, f'{args.plot_path}ebm_heat_{itr}.pdf', args)
                utils.plot_sampler(energy_model, f'{args.sample_path}ebm_samples_{itr}.png', args)

            # # heat map of energy
            # plot_heat(energy_model, bm, plot_size, args.device, int_scale, arg.discrete_dim,
            #           out_file=os.path.join(args.plot_path, f'heat_{itr}.pdf'))

            # # samples of gfn
            # gfn_samples = gfn.sample(4000).detach()
            # gfn_samp_float = bin2float(gfn_samples.data.cpu().numpy().astype(int), inv_bm, int_scale, args.discrete_dim)
            # plot_samples(gfn_samp_float, os.path.join(args.sample_path, f'gfn_samples_{itr}.pdf'), lim=plot_size)

            # GFN LL
            gfn.model.eval()
            logps = []
            pbar = tqdm(range(10))
            pbar.set_description("GFN Calculating likelihood")
            for _ in pbar:
                pos_samples_bs = get_batch_data(db, args, batch_size=1000)
                pos_samples_bs = torch.from_numpy(np.float32(pos_samples_bs)).to(args.device)

                # pos_samples_bs = get_true_samples(db, 1000, bm, int_scale, args.discrete_dim).to(args.device)
                logp = gfn.cal_logp(pos_samples_bs, 20)
                logps.append(logp.reshape(-1))
                pbar.set_postfix({"logp": f"{torch.cat(logps).mean().item():.2f}"})
            gfn_test_ll = torch.cat(logps).mean()

            print(f"Test NLL ({itr}): GFN: {-gfn_test_ll.item():.3f}")


        itr += 1
        if itr > args.n_iters:
            quit(0)
