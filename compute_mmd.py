from utils import utils
from utils.eval import exp_hamming_mmd
from utils.utils import get_batch_data
from utils.utils import setup_data
from utils.utils import get_binmap
from utils.utils import get_last_n_levels
from utils.utils import remove_last_n_levels
from utils.utils import has_n_levels
from tqdm import tqdm

import time
from datetime import datetime

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

from methods.cd_runi_inter.train import gen_samples as cd_runi_inter_gen_samples
from methods.dataq_dfs.train import gen_samples as dataq_dfs_gen_samples
from methods.dataq_dfs_ebm.train import gen_samples as dataq_dfs_ebm_gen_samples
from methods.mask_dfs.train import gen_samples as mask_dfs_gen_samples
from methods.mask_dfs_2.train import gen_samples as mask_dfs_2_gen_samples
from methods.mask_dfs_3.train import gen_samples as mask_dfs_3_gen_samples
from methods.mask_dfs_4.train import gen_samples as mask_dfs_4_gen_samples
from methods.mask_dfs_5.train import gen_samples as mask_dfs_5_gen_samples
from methods.mask_dfs_ce.train import gen_samples as mask_dfs_ce_gen_samples
from methods.mask_dfs_ce_forced.train import gen_samples as mask_dfs_ce_forced_gen_samples
from methods.mask_dfs_ce_2.train import gen_samples as mask_dfs_ce_2_gen_samples

from methods.cd_runi_inter.train import make_sampler as cd_runi_inter_make_sampler
from methods.dataq_dfs.train import make_sampler as dataq_dfs_make_sampler
from methods.dataq_dfs_ebm.train import make_sampler as dataq_dfs_ebm_make_sampler
from methods.mask_dfs.train import make_sampler as mask_dfs_make_sampler
from methods.mask_dfs_2.train import make_sampler as mask_dfs_2_make_sampler
from methods.mask_dfs_3.train import make_sampler as mask_dfs_3_make_sampler
from methods.mask_dfs_4.train import make_sampler as mask_dfs_4_make_sampler
from methods.mask_dfs_5.train import make_sampler as mask_dfs_5_make_sampler
from methods.mask_dfs_ce.train import make_sampler as mask_dfs_ce_make_sampler
from methods.mask_dfs_ce_forced.train import make_sampler as mask_dfs_ce_forced_make_sampler
from methods.mask_dfs_ce_2.train import make_sampler as mask_dfs_ce_2_make_sampler
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # parser.add_argument('--methods', type=str, default='punidb', 
        # choices=[
        #     'punidb', 'runidb',
        #     'ed_ebm', 'ebm_runidb',
        #     'cd_ebm', 'cd_runi_inter',
        #     'dataq_dfs', 'dataq_dfs_ebm',
        #     'uniform_ebm', 'mask_dfs',
        #     'mask_dfs_2', 'mask_dfs_3',
        #     'mask_dfs_4', 'mask_dfs_5',
        #     'mask_dfs_ce', 'mask_dfs_ce_2',  
        #     'mask_dfs_ce_forced'
        # ],
    # )

    parser.add_argument('--model_types', type=str, default='ebm-data', 
        choices=[
            'dfs_data','dfs_ebm',
            'gfn_data','gfn_ebm', 
            'ebm_data'
        ],
    )
    parser.add_argument('--model1', type=str, default=None)
    parser.add_argument('--model2', type=str, default=None)
    #parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--discrete_dim', type=int, default=32)
    parser.add_argument('--output_path', type=str, default='./mmd_results/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--delta_t', type=float, default='0.01')
    parser.add_argument('--eta', type=float, default='0')

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

def dfs_data(args):

    #dfs
    args.methods = find_dfs_method(args.model1)
    gen_samples = eval(f'{args.methods}_gen_samples')
    make_sampler = eval(f'{args.methods}_make_sampler')
    model1 = make_sampler(args.model1, args)

    #data
    model2, _, _ = setup_data(args)

    #compute mmd
    mmd_list = []
    pbar = tqdm(range(args.num_rounds))
    for i in pbar:
        x = torch.from_numpy(gen_samples(model1, args, batch_size=args.batch_size)).to('cpu')
        y = get_batch_data(model2, args, batch_size=args.batch_size)
        y = torch.from_numpy(np.float32(y)).to('cpu')
        pbar.set_description(f'Iteration {i}')
        mmd_list.append(exp_hamming_mmd(x,y))
    mmd = sum(mmd_list)/args.num_rounds
    log_mmd(mmd, args)
    plot_samples(x, y, args)
    

def dfs_ebm(args):
    #dfs
    args.methods = find_dfs_method(args.model1)
    gen_samples = eval(f'{args.methods}_gen_samples')
    make_sampler = eval(f'{args.methods}_make_sampler')
    model1 = make_sampler(args.model1, args)

    #ebm
    model2 = make_ebm(args.model2, args)
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)

    #data
    _, _, _ = setup_data(args)

    #compute mmd
    mmd_list = []
    pbar = tqdm(range(args.num_rounds))
    for i in pbar:
        x = torch.from_numpy(gen_samples(model1, args, batch_size=args.batch_size)).to('cpu')
        y = gibbs_sampler(model2, num_rounds=args.num_rounds, num_samples=args.batch_size).to('cpu')
        pbar.set_description(f'Iteration {i}')
        mmd_list.append(exp_hamming_mmd(x,y))
    mmd = sum(mmd_list)/args.num_rounds
    log_mmd(mmd, args)
    plot_samples(x, y, args)

def gfn_data(args):
    #gfn
    model1 = get_GFlowNet(args.gfn_type, args.discrete_dim, args, args.device)

    #data
    model2, _, _ = setup_data(args)

    #compute mmd
    mmd_list = []
    pbar = tqdm(range(args.num_rounds))
    for i in pbar:
        x = model1.sample(args.batch_size).detach().to('cpu')
        y = get_batch_data(model2, args, batch_size=args.batch_size)
        y = torch.from_numpy(np.float32(y)).to('cpu')
        pbar.set_description(f'Iteration {i}')
        mmd_list.append(exp_hamming_mmd(x,y))
    mmd = sum(mmd_list)/args.num_rounds
    log_mmd(mmd, args)
    plot_samples(x, y, args)


def gfn_ebm(args):
    #gfn
    model1 = get_GFlowNet(args.gfn_type, args.discrete_dim, args, args.device)

    #ebm
    model2 = make_ebm(args.model2, args)
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)

    #data
    _, _, _ = setup_data(args)

    #compute mmd
    mmd_list = []
    pbar = tqdm(range(args.num_rounds))
    for i in pbar:
        x = model1.sample(args.batch_size).detach().to('cpu')
        y = gibbs_sampler(model2, num_rounds=args.num_rounds, num_samples=args.batch_size).to('cpu')
        pbar.set_description(f'Iteration {i}')
        mmd_list.append(exp_hamming_mmd(x,y))
    mmd = sum(mmd_list)/args.num_rounds
    log_mmd(mmd, args)
    plot_samples(x, y, args)

def ebm_data(args):
    #ebm
    model1 = make_ebm(args.model1, args)
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)


    #data
    model2, _, _ = setup_data(args)

    #compute mmd
    mmd_list = []
    pbar = tqdm(range(args.num_rounds))
    for i in pbar:
        x = gibbs_sampler(model1, num_rounds=args.num_rounds, num_samples=args.batch_size).to('cpu')
        y = get_batch_data(model2, args, batch_size=args.batch_size)
        y = torch.from_numpy(np.float32(y)).to('cpu')
        pbar.set_description(f'Iteration {i}')
        mmd_list.append(exp_hamming_mmd(x,y))
    mmd = sum(mmd_list)/args.num_rounds
    log_mmd(mmd, args)
    plot_samples(x, y, args)

def find_data_name(path_1,path_2):
    data_names = [
            '2spirals', 'checkerboard',
            'pinwheel', 'swissroll',
            'moons', '8gaussians',
            'circles'
        ]

    split_path_1 = path_1.split(os.path.sep)
    split_path_2 = path_2.split(os.path.sep)

    for data_name in data_names:
        if data_name in split_path_1:
            if data_name in split_path_2:
                return data_name
            raise NotImplementedError("Data name that appears in model 1 does not appear in model 2.")
    
    raise NotImplementedError("Method could not be inferred from pathname")

def find_dfs_method(path):
    methods = [
            'punidb', 'runidb',
            'ed_ebm', 'ebm_runidb',
            'cd_ebm', 'cd_runi_inter',
            'dataq_dfs', 'dataq_dfs_ebm',
            'uniform_ebm', 'mask_dfs',
            'mask_dfs_2', 'mask_dfs_3',
            'mask_dfs_4', 'mask_dfs_5',
            'mask_dfs_ce', 'mask_dfs_ce_2',  
            'mask_dfs_ce_forced'
        ]

    split_path = path.split(os.path.sep)

    for method in methods:
        if method in split_path:
            return method
    
    raise NotImplementedError("Method could not be inferred from pathname")


def make_ebm(model_path, args):
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    ebm = EBM(net).to(args.device)
    try:
        ebm.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
        sys.exit(1)
    return ebm

def log_mmd(mmd, args):
    log_path = f'{args.output_path_1}mmd_log.csv'
    log_entry = {}
    log_entry['model_types'] = f'{args.model_types}'
    log_entry['model1'] = args.model1_path
    log_entry['model2'] = args.model2_path
    log_entry['time'] = args.formatted_time
    log_entry['mmd'] = mmd.item()
    if args.model_types in ['dfs_ebm', 'dfs_data']:
        log_entry['dt'] = args.delta_t
        log_entry['eta'] = args.eta
    df_log_entry = pd.DataFrame([log_entry])
    df_log_entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    print(f'mmd result logged in {log_path}')
    print(f'mmd is {mmd} between models x={get_last_n_levels(args.model1)} and y={get_last_n_levels(args.model2)}, and final samples from x and y have been saved to {args.output_path}')

def plot_samples(x,y,args):
    if args.vocab_size == 2:
        x_float_samples = utils.bin2float(x.detach().cpu().numpy().astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
        y_float_samples = utils.bin2float(y.detach().cpu().numpy().astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
    else:
        x_float_samples = utils.ourbase2float(x.detach().cpu().numpy().astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
        y_float_samples = utils.ourbase2float(y.detach().cpu().numpy().astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)

    utils.plot_samples(x_float_samples, args.x_path, im_size=4.1, im_fmt='png')
    # utils.plot_samples(y_float_samples, y_path, im_size=4.1, im_fmt='png')


if __name__ == '__main__':
    args = get_args()
    assert args.vocab_size == 2, 'Can only handle vocab size 2 at present'
    args.data_name = find_data_name(args.model1, args.model2)
    args.bm, args.inv_bm = get_binmap(args.discrete_dim, 'gray')
    args.vocab_size_with_mask = args.vocab_size + 1
    #args.output_path = f'{args.output_path}/{args.model_types}'
    assert has_n_levels(args.model1, 2), 'Model 1 has fewer than 2 levels, which causes an error in saving the results. Please fix.'
    args.output_path_1 = f'{remove_last_n_levels(args.model1, 2)}/mmd_results/'
    #args.output_path_2 = f'{remove_last_n_levels(args.model2, 2)}/mmd_results/' if has_n_levels(args.model2, 2) else None
    #os.makedirs(f'{args.output_path}', exist_ok=True)
    os.makedirs(f'{args.output_path_1}', exist_ok=True)
    # n, m = 1, 1
    # while True:
    args.model1_path = f'{get_last_n_levels(args.model1)}'
    args.model2_path = f'{get_last_n_levels(args.model2)}'
    current_time = time.time()
    datetime_obj = datetime.fromtimestamp(current_time)
    args.formatted_time = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
    if args.model_types in ['dfs_ebm', 'dfs_data']:
        args.x_path = f'{args.output_path_1}{args.model1_path}_samples_dt={args.delta_t}_eta={args.eta}_{args.formatted_time}.png'
    if args.model_types in ['gfn_ebm', 'gfn_data']:
        args.x_path = f'{args.output_path_1}{args.model1_path}_samples_{args.formatted_time}.png'
    if args.model_types in ['ebm_data']:
        args.x_path = f'{args.output_path_1}{args.model1_path}_samples_{args.formatted_time}.png'
    print(f'Output path is {args.output_path_1}.')
    #     if not os.path.exists(x_path):
    #         break
    #     n += 1
    # while True:
    # args.y_path = f'{args.output_path_2}/dt={args.delta_t}_eta={args.eta}.png'
        # if not os.path.exists(y_path):
        #     break
        # m += 1
    # args.x_path = x_path
    # args.y_path = y_path

    if args.model_types == 'dfs_data':
        dfs_data(args)
    elif args.model_types == 'dfs_ebm':
        dfs_ebm(args)
    elif args.model_types == 'gfn_data':
        gfn_data(args)
    elif args.model_types == 'gfn_ebm':
        gfn_ebm(args)
    elif args.model_types == 'ebm_data':
        ebm_data(args)

    print('Done. Results were saved with model 1.')

