import os
import torch
import shutil
import argparse
import json 
import numpy as np
import datetime
import pandas as pd

from utils.eval import log_args
import utils.utils as utils

from eb_gfn.train import main_loop as eb_gfn_main_loop
from gfn.train import main_loop as gfn_main_loop
from ed_ebm.train import main_loop as ed_ebm_main_loop
from velo_dfm.train import main_loop as velo_dfm_main_loop
from velo_dfs.train import main_loop as velo_dfs_main_loop
from velo_edfm.train import main_loop as velo_edfm_main_loop
from velo_edfs.train import main_loop as velo_edfs_main_loop
from velo_ebm.train import main_loop as velo_ebm_main_loop
from velo_bootstrap_ebm.train import main_loop as velo_bootstrap_ebm_main_loop
from velo_bootstrap_v2_ebm.train import main_loop as velo_bootstrap_v2_ebm_main_loop
from velo_baf_ebm.train import main_loop as velo_baf_ebm_main_loop


# from velo_edfm_ebm_bootstrap.train import main_loop as velo_edfm_ebm_bootstrap_main_loop
# from velo_edfm_ebm_bootstrap_2.train import main_loop as velo_edfm_ebm_bootstrap_2_main_loop

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--methods', type=str, default='velo_dfm', 
        choices=[
            'eb_gfn', 'gfn', 'ed_ebm',
            'velo_dfm','velo_dfs', 'velo_edfm', 
            'velo_edfs','velo_ebm', 'velo_bootstrap_ebm', 
            'velo_bootstrap_v2_ebm',
            'velo_baf_ebm',
        ],
    )

    #data related
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--discrete_dim', type=int, default=28*28)
    parser.add_argument('--vocab_size', type=int, default=2)

    #ebm related
    parser.add_argument('--ebm_init_mean', type=int, default=0, choices=[0,1])
    parser.add_argument('--ebm_model', type=str, default='resnet-64')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_ebm', type=str, default='auto')
    parser.add_argument('--p_control', type=float, default=0.0)
    parser.add_argument('--gradnorm', "--gn", type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--store_ebm_ema', type=int, default=1, choices=[0,1])
    parser.add_argument('--warmup_iters', type=int, default=0) #10000
    parser.add_argument('--ebm_lr', type=float, default=.0001)

    # mcmc
    parser.add_argument('--save_sampling_steps', type=int, default=5000)
    parser.add_argument('--MCMC_refinement', type=int, default=0)
    parser.add_argument('--use_MCMC', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_buffer', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampler', type=str, default='dmala')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=40)
    parser.add_argument('--reinit_freq', type=float, default=0.0)
    parser.add_argument('--eval_sampling_steps', type=int, default=10000)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--buffer_init', type=str, default='mean')
    parser.add_argument('--step_size', type=float, default=0.1)

    #dfm/dfs related
    parser.add_argument('--dfs', type=int, choices=[0,1]) #only applies if the method is specified jointly for dfs/dfm (e.g. velo_ebm)
    parser.add_argument('--use_ema_dfs', type=int, default=0, choices=[0,1]) #only applies if the method is specified jointly for dfs/dfm (e.g. velo_ebm)

    parser.add_argument('--enable_backward', type=int, default=0, choices=[0,1])
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--scheduler_type', type=str, default='linear', choices=['cubic','quadratic', 'quadratic_noise', 'linear'])
    parser.add_argument('--store_dfs_ema', type=int, default=0, choices=[0,1])
    parser.add_argument('--dfm_step', type=int, default=0, choices=[0,1])
    parser.add_argument('--mixer_step', type=int, default=0, choices=[0,1])
    parser.add_argument('--mixer_type', type=str, default='uniform', choices=['uniform','data_mean'])
    parser.add_argument('--source', type=str, default='mask', choices=['mask','uniform','data','omniglot'])
    parser.add_argument('--dfs_warmup_iter', type=int, default=1000)
    parser.add_argument('--dfs_per_ebm', type=int, default=1)
    parser.add_argument('--delta_t', type=float, default=0.05)
    parser.add_argument('--impute_self_connections', type=int, default=1, choices=[0,1])
    parser.add_argument('--loss_weight', type=int, default=1, choices=[0,1])
    parser.add_argument('--norm_by_mean', type=int, default=1, choices=[0,1])
    parser.add_argument('--norm_by_max', type=int, default=0, choices=[0,1])
    parser.add_argument('--q_weight', type=float, default=0.0) #remove?
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--dfs_init_from_checkpoint', type=int, default=1, choices=[0,1])
    parser.add_argument('--init_save_every', type=int, default=500)
    parser.add_argument('--init_iter', type=int, default=1000)
    parser.add_argument('--init_sampling_steps', type=int, default=100)
    parser.add_argument('--base_dist', type=str, default='data_mean', choices=['uniform', 'data_mean', 'zero']) #note that 'uniform' is to be avoided as it just blows up the energy...
    parser.add_argument('--relu', type=int, default=0, choices=[0,1])

    #edfm/dfs
    parser.add_argument('--start_temp', type=float, default=1.0)
    parser.add_argument('--temp_decay', type=float, default=1)
    parser.add_argument('--optimal_temp', type=int, default=0, choices=[0,1])
    parser.add_argument('--optimal_temp_use_median', type=int, default=1, choices=[0,1])
    parser.add_argument('--optimal_temp_ema', type=float, default=0)
    parser.add_argument('--optimal_temp_diff', type=float, default=0.5)
    parser.add_argument('--q', type=str, default='data_mean', choices=['data_mean','random'])

    #back and forth
    parser.add_argument("--rand_t", "--rt", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_t", "--lt", type=int, default=0, choices=[0, 1])
    parser.add_argument("--warmup_baf", "--wbaf", type=lambda x: int(float(x)), default=5, help="need to use w/ lin_t")
    parser.add_argument("--t", type=float, default=-1.0, help="for gfn back forth negative sample generation")
    parser.add_argument("--ebm_with_mh", "--ewm", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dfs_with_mh", "--dwm", type=int, default=1, choices=[0, 1])



    #gfn related
    parser.add_argument("--down_sample", "--ds", default=0, type=int, choices=[0, 1])
    parser.add_argument('--ebm_every', "--ee", type=int, default=1)
    parser.add_argument('--print_every', "--pe", type=int, default=100)
    parser.add_argument('--plot_every', type=int, default=2500)
    parser.add_argument("--gfn_per_itr", type=int, default=1, help="GFN training frequency")
    parser.add_argument("--ebm_per_itr", type=int, default=1, help="EBM training frequency")

    parser.add_argument("--type", type=str, default='tblb')
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--hid_layers", "--hl", type=int, default=3)
    parser.add_argument("--leaky", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init_zero", "--iz", type=int, default=0, choices=[0, 1], )
    parser.add_argument("--gmodel", "--gm", type=str,default="mlp")
    parser.add_argument("--train_steps", "--ts", type=int, default=1)
    parser.add_argument("--l1loss", "--l1l", type=int, default=0, choices=[0, 1], help="use soft l1 loss instead of l2")

    parser.add_argument("--rand_k", "--rk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_k", "--lk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--warmup_k", "--wk", type=lambda x: int(float(x)), default=5, help="need to use w/ lin_k")
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

    #misc
    parser.add_argument('--verbose', default=False, type=bool, help='evaluate final nll and mmd')
    parser.add_argument('--experiment_name', default="", type=str, help='unique experiment name for meta data')
    parser.add_argument('--ema', type=float, default=0.999)
    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu; 0 - ?: specific gpu index')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_itr', default=200, type=int, help='num itr')
    parser.add_argument('--itr_save', default=1, type=int, help='num itrs between save')
    parser.add_argument('--eval_on', type=int, default=1, choices=[0,1])
    parser.add_argument('--eval_every', type=int, default=50)

    parser.add_argument("--final_ais_samples", type=int, default=100000)
    parser.add_argument("--intermediate_ais_samples", type=int, default=10000)
    parser.add_argument("--final_ais_num_steps", type=int, default=1000)
    parser.add_argument("--intermediate_ais_num_steps", type=int, default=100)
    parser.add_argument('--gibbs_num_rounds', type=int, default=100)

    #bootstrap ebm-dfs
    parser.add_argument('--optional_step', type=int, default=0, choices=[0,1])
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--recycle_dfs_sample', type=int, default=0, choices=[0,1])
    
    args = parser.parse_args()

    #imputed args
    args.model_has_noise = 1
    if args.methods in ['velo_dfm','velo_efm','velo_efm_ebm','velo_efm_ebm_bootstrap','velo_efm_ebm_bootstrap_2']:
        if args.scheduler_type == 'quadratic_noise':
            args.model_has_noise = 1

    gpu = args.gpu
    if torch.cuda.is_available() and gpu >= 0:
        torch.cuda.set_device(gpu)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        args.device = torch.device('cpu')
        print('use cpu')
    args.save_dir = f'./methods/{args.methods}/experiments/{args.dataset_name}' #change this
    os.makedirs(args.save_dir, exist_ok=True)
    args.data_dir = f'./methods/datasets' 
    args.completed = False
    args.vocab_size_with_mask = args.vocab_size + 1
    start_time = datetime.datetime.now()
    args.start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    args.index_path = f'{args.save_dir}/experiment_idx.json'

    toy_dataset_list = ['swissroll','circles','moons','8gaussians','pinwheel','2spirals','checkerboard']
    args.is_toy = args.dataset_name in toy_dataset_list

    #define standard pretrained ebms for toy
    toy_data_pretrained = {
        'swissroll': './methods/ed_ebm/experiments/swissroll/swissroll_0/ckpts/model_100000.pt',
        'circles': './methods/ed_ebm/experiments/circles/circles_0/ckpts/model_100000.pt',
        'moons': './methods/ed_ebm/experiments/moons/moons_0/ckpts/model_100000.pt',
        '8gaussians': './methods/ed_ebm/experiments/8gaussians/8gaussians_0/ckpts/model_100000.pt',
        'pinwheel':'./methods/ed_ebm/experiments/pinwheel/pinwheel_0/ckpts/model_100000.pt',
        '2spirals':'./methods/ed_ebm/experiments/2spirals/2spirals_0/ckpts/model_100000.pt',
        'checkerboard':'./methods/ed_ebm/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt'
    }
    real_data_pretrained = {
        'caltech': './methods/dmala_ebm/figs/ebm_caltech/best_ckpt_caltech_dmala_0.1.pt',
        'dynamic_mnist': './methods/dmala_ebm/figs/ebm_dynamic_mnist/best_ckpt_dynamic_mnist_dmala_0.1.pt',
        'omniglot': './methods/dmala_ebm/figs/ebm_omniglot/best_ckpt_omniglot_dmala_0.1.pt',
        'static_mnist': './methods/dmala_ebm/figs/ebm_stati_MNIST/best_ckpt_static_mnist_dmala_0.1.pt',
    }
    if args.is_toy and args.pretrained_ebm == 'auto':
        args.pretrained_ebm = toy_data_pretrained[args.dataset_name]
        args.input_size = [32]
        args.input_type = "binary" 
    elif args.pretrained_ebm == 'auto':
        args.input_size = [32]
        args.input_type = "binary"
        args.pretrained_ebm = real_data_pretrained[args.dataset_name]
        #load normalizing constant for evaluation of samplers:
        log_path = './methods/datasets/real_datasets_ais_stats.csv'
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            model_name = args.pretrained_ebm.split('/')[-1]
            filtered_df = df[df['model_name'].str.contains(model_name, na=False)]
            if not filtered_df.empty:
                args.logZ = filtered_df['logZ'].values[0]
                print(f"The logZ value for model '{model_name}' is: {args.logZ}")
            else:
                print(f"Caution: No logZ entries found for model '{model_name}' – any sampler evaluation will fail.")
                args.logZ = None
        else:
            print(f"Caution: No file found for logZ – any sampler evaluation will fail.")
            args.logZ = None

    return args

if __name__ == '__main__':
    args = get_args()
    log_args(args.methods, args.dataset_name, args)

    if os.path.exists(args.ckpt_path):
        print(f'removed checkpoint data that was not indexed')
        shutil.rmtree(args.ckpt_path)
    os.makedirs(args.ckpt_path, exist_ok=True)
    if os.path.exists(args.plot_path):
        print(f'removed plot data that was not indexed')
        shutil.rmtree(args.plot_path)
    os.makedirs(args.plot_path, exist_ok=True)
    if os.path.exists(args.sample_path):
        print(f'removed sample data that was not indexed')
        shutil.rmtree(args.sample_path)
    os.makedirs(args.sample_path, exist_ok=True)

    print(f'Saving experiment data to {args.save_dir}/{args.dataset_name}_{args.exp_n}')

    main_fn = eval(f'{args.methods}_main_loop')

    main_fn(args, verbose=True)

