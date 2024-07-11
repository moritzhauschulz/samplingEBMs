from utils import utils
from utils.eval import exp_hamming_mmd
from utils.utils import get_batch_data
from utils.utils import setup_data
from utils.utils import get_binmap
from utils.utils import get_last_n_levels
from tqdm import tqdm
import torch
import os
import numpy as np
import pandas as pd
from utils.sampler import GibbsSampler 
import sys
from torch.distributions.categorical import Categorical

from methods.eb_gfn.model import get_GFlowNet
from methods.cd_runi_inter.model import MLPScore, EBM
from methods.cd_runi_inter.model import MLPModel as MLPFlow

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--model_types', type=str, default='ebm-data', choices=['ebm-data','ebm-dfs','ebm-gfn','dfs-data', 'gfn-data'])
    parser.add_argument('--model1', type=str, default=None)
    parser.add_argument('--model2', type=str, default=None)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--discrete_dim', type=int, default=32)
    parser.add_argument('--output_path', type=str, default='./mmd_results/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--delta_t', type=float, default='0.01')

    # for GFN
    parser.add_argument("--gfn_type", type=str, default='tblb')
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--hid_layers", "--hl", type=int, default=3)
    parser.add_argument("--leaky", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init_zero", "--iz", type=int, default=0, choices=[0, 1], )
    parser.add_argument("--gmodel", "--gm", type=str,default="mlp")
    parser.add_argument("--train_steps", "--ts", type=int, default=1)
    parser.add_argument("--l1loss", "--l1l", type=int, default=0, choices=[0, 1], help="use soft l1 loss instead of l2")
    parser.add_argument("--with_mh", "--wm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rand_k", "--rk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_k", "--lk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--warmup_k", "--wk", type=lambda x: int(float(x)), default=0, help="need to use w/ lin_k")
    parser.add_argument("--K", type=int, default=-1, help="for gfn back forth negative sample generation")
    parser.add_argument("--rand_coef", "--rc", type=float, default=0, help="for tb")
    parser.add_argument("--back_ratio", "--br", type=float, default=0.)
    parser.add_argument("--clip", type=float, default=-1., help="for gfn's linf gradient clipping")
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--glr", type=float, default=1e-3)
    parser.add_argument("--zlr", type=float, default=1)
    parser.add_argument("--momentum", "--mom", type=float, default=0.0)
    parser.add_argument("--gfn_weight_decay", "--gwd", type=float, default=0.0)
    parser.add_argument('--verbose', default=False, type=bool, help='verbose')
    args = parser.parse_args()

    return args


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

if __name__ == '__main__':


    args = get_args()
    assert args.vocab_size == 2, 'Can only handle vocab size 2 at present'

    mmd_list = []
    args.bm, args.inv_bm = get_binmap(args.discrete_dim, 'gray')
    pbar = tqdm(range(args.num_rounds))
    

    if args.model_types == 'ebm-data':

        args.data_name = args.model2

        net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
        model1 = EBM(net).to(args.device)
        try:
            model1.load_state_dict(torch.load(args.model1))
        except FileNotFoundError as e:
            print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
            sys.exit(1)
        model1.eval()
        model2, _, _ = setup_data(args)

        for i in pbar:
            gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
            x = gibbs_sampler(model1, num_rounds=args.num_rounds, num_samples=args.batch_size).to('cpu')
            y = get_batch_data(model2, args, batch_size=args.batch_size)
            y = torch.from_numpy(np.float32(y)).to('cpu')
            pbar.set_description(f'Iteration {i}')
            mmd_list.append(exp_hamming_mmd(x,y))
    elif args.model_types == 'ebm-dfs':
        setup_data(args)
        net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
        model1 = EBM(net).to(args.device)
        try:
            model1.load_state_dict(torch.load(args.model1))
        except FileNotFoundError as e:
            print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
            sys.exit(1)
        model1.eval()
        model2 = MLPFlow(args).to(args.device)
        try:
            model2.load_state_dict(torch.load(args.model2))
        except FileNotFoundError as e:
            print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
            sys.exit(1)
        model2.eval()

        for i in pbar:
            gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
            x = gibbs_sampler(model1, num_rounds=args.num_rounds, num_samples=args.batch_size).to('cpu')
            y = torch.from_numpy(gen_samples(model2, args, batch_size=args.batch_size)).to('cpu')
            pbar.set_description(f'Iteration {i}')
            mmd_list.append(exp_hamming_mmd(x,y))
    elif args.model_types == 'ebm-gfn':
        setup_data(args)
        net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
        model1 = EBM(net).to(args.device)
        try:
            model1.load_state_dict(torch.load(args.model1))
        except FileNotFoundError as e:
            print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
            sys.exit(1)
        model1.eval()
        model2 = get_GFlowNet(args.gfn_type, args.discrete_dim, args, args.device)
        try:
            model2.model.load_state_dict(torch.load(args.model2))
        except FileNotFoundError as e:
            print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
            sys.exit(1)
        for i in pbar:
            gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
            x = gibbs_sampler(model1, num_rounds=args.num_rounds, num_samples=args.batch_size).to('cpu')
            y = model2.sample(args.batch_size).detach().to('cpu')
            pbar.set_description(f'Iteration {i}')
            mmd_list.append(exp_hamming_mmd(x,y))
    elif args.model_types == 'dfs-data':

        args.data_name = args.model2

        model1 = MLPFlow(args).to(args.device)
        try:
            model1.load_state_dict(torch.load(args.model1))
        except FileNotFoundError as e:
            print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
            sys.exit(1)
        model1.eval()
        model2, _, _ = setup_data(args)

        for i in pbar:
            x = torch.from_numpy(gen_samples(model1, args, batch_size=args.batch_size)).to('cpu')
            y = get_batch_data(model2, args, batch_size=args.batch_size)
            y = torch.from_numpy(np.float32(y)).to('cpu')
            pbar.set_description(f'Iteration {i}')
            mmd_list.append(exp_hamming_mmd(x,y))
    elif args.model_types == 'gfn-data':

        args.data_name = args.model2

        model1 = get_GFlowNet(args.gfn_type, args.discrete_dim, args, args.device)
        try:
            model1.model.load_state_dict(torch.load(args.model1))
        except FileNotFoundError as e:
            print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
            sys.exit(1)
        model2, _, _ = setup_data(args)

        for i in pbar:
            x = model1.sample(args.batch_size).detach().to('cpu')
            y = get_batch_data(model2, args, batch_size=args.batch_size)
            y = torch.from_numpy(np.float32(y)).to('cpu')
            pbar.set_description(f'Iteration {i}')
            mmd_list.append(exp_hamming_mmd(x,y))

    mmd = sum(mmd_list)/args.num_rounds

    if args.vocab_size == 2:
        x_float_samples = utils.bin2float(x.detach().cpu().numpy().astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
        y_float_samples = utils.bin2float(y.detach().cpu().numpy().astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
    else:
        x_float_samples = utils.ourbase2float(x.detach().cpu().numpy().astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
        y_float_samples = utils.ourbase2float(y.detach().cpu().numpy().astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    
    args.output_path = f'{args.output_path}/{args.model_types}'
    os.makedirs(f'{args.output_path}', exist_ok=True)
    
    n, m = 1, 1
    while True:
        x_path = f'{args.output_path}/{get_last_n_levels(args.model1)}_{n}.png'
        if not os.path.exists(x_path):
            break
        n += 1
    while True:
        y_path = f'{args.output_path}/{get_last_n_levels(args.model2)}_{m}.png'
        if not os.path.exists(y_path):
            break
        m += 1
    
    utils.plot_samples(x_float_samples, x_path, im_size=4.1, im_fmt='png')
    utils.plot_samples(y_float_samples, y_path, im_size=4.1, im_fmt='png')

    log_path = f'{args.output_path}/mmd_log.csv'
    log_entry = {}
    log_entry['model_types'] = f'{args.model_types}'
    log_entry['model1'] = f'{get_last_n_levels(args.model1)}_{n}'
    log_entry['model2'] = f'{get_last_n_levels(args.model2)}_{m}'
    log_entry['mmd'] = mmd.item()
    df_log_entry = pd.DataFrame([log_entry])
    df_log_entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    print(f'mmd result logged in {log_path}')

    print(f'mmd is {mmd} between models x={get_last_n_levels(args.model1)} and y={get_last_n_levels(args.model2)}, and final samples from x and y have been saved to {args.output_path}')



