import os
import torch
import shutil
import argparse
import json
import numpy as np
import datetime

from utils.eval import log_args
from runidb.train import main_loop as runidb_main_loop
from punidb.train import main_loop as punidb_main_loop


from runidb.train import main_loop as runidb_main_loop
from punidb.train import main_loop as punidb_main_loop
from velo_mask_dfm.train import main_loop as velo_mask_dfm_main_loop
from velo_uni_dfm.train import main_loop as velo_uni_dfm_main_loop
from velo_uni_dfs.train import main_loop as velo_uni_dfs_main_loop
from velo_mask_dfs.train import main_loop as velo_mask_dfs_main_loop
from velo_efm.train import main_loop as velo_efm_main_loop
from velo_uni_efm.train import main_loop as velo_uni_efm_main_loop
from velo_mask_edfs.train import main_loop as velo_mask_edfs_main_loop
from velo_uni_edfs.train import main_loop as velo_uni_edfs_main_loop
from velo_efm_ebm.train import main_loop as velo_efm_ebm_main_loop

# from methods.ed_ebm.train import main_loop as ed_ebm_main_loop
# from methods.ebm_runi.train import main_loop as ebm_runi_main_loop
# from methods.ebm_runidb.train import main_loop as ebm_runidb_main_loop

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--methods', type=str, default='punidb', 
        choices=[
            'punidb', 'runidb','velo_mask_dfm', 'velo_uni_dfm',
            'velo_mask_dfs', 'velo_uni_dfs', 'velo_efm', 'velo_uni_efm',
            'velo_mask_edfs', 'velo_uni_edfs', 'velo_efm_ebm'
        ],
    )

    #ebm
    parser.add_argument('--ebm_model', type=str, default='resnet-64')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_ebm', type=str, default='./methods_MNIST/dl_ebm/figs/ebm/ckpt_static_mnist_dula_0.1.pt')
    parser.add_argument('--base_dist', action='store_true')
    parser.add_argument('--p_control', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--save_sampling_steps', type=int, default=10000)

    # mcmc
    parser.add_argument('--use_dula', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_buffer', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampler', type=str, default='dula')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--reinit_freq', type=float, default=0.0)
    parser.add_argument('--eval_sampling_steps', type=int, default=10000)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--buffer_init', type=str, default='mean')
    parser.add_argument('--step_size', type=float, default=0.1)

    #training
    parser.add_argument('--optimal_temp', type=int, default=1, choices=[0,1])
    parser.add_argument('--optimal_temp_use_median', type=int, default=1, choices=[0,1])
    parser.add_argument('--optimal_temp_ema', type=float, default=0.9)
    parser.add_argument('--bernoulli_step', type=int, default=1, choices=[0,1])
    parser.add_argument('--source', type=str, default='mask', choices=['mask','uniform'])
    parser.add_argument('--dfs_warmup_iter', type=int, default=0)
    parser.add_argument('--dfs_per_ebm', type=int, default=1)
    parser.add_argument('--warmup_iters', type=int, default=1000) #10000
    parser.add_argument('--start_temp', type=float, default='1')
    parser.add_argument('--end_temp', type=float, default='1')
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--discrete_dim', type=int, default=28*28)
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--ema', type=float, default=0.999)
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--impute_self_connections', type=int, default=1, choices=[0,1])
    parser.add_argument('--loss_weight', type=int, default=1, choices=[0,1])
    parser.add_argument('--norm_by_sum', type=int, default=0, choices=[0,1])
    parser.add_argument('--norm_by_max', type=int, default=1, choices=[0,1])
    parser.add_argument('--q_weight', type=float, default=0.0)


    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--ebm_lr', type=float, default=.0001)
    parser.add_argument('--weight_decay', type=float, default=.0)

    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu; 0 - ?: specific gpu index')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=128)

    parser.add_argument('--num_epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--epoch_save', default=5, type=int, help='num epochs between save')
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--eval_on', type=int, default=1, choices=[0,1])


    args = parser.parse_args()

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

    args.save_dir = f'./methods_MNIST/{args.methods}/experiments/{args.dataset_name}'
    os.makedirs(args.save_dir, exist_ok=True)

    args.completed = False
    args.vocab_size_with_mask = args.vocab_size + 1
    start_time = datetime.datetime.now()
    args.start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    args.index_path = f'{args.save_dir}/experiment_idx.json'
    args.log_norm = None

    logger = open("{}/log.txt".format(args.save_dir), 'w')
    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')
        logger.flush()
    args.my_print = my_print

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
