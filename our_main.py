import os
import torch
import shutil
import argparse
import json
import numpy as np
import datetime

from utils import utils
from utils.eval import log_args
from utils.eval import compute_mmd_base_stats
from methods.runidb.train import main_loop as runidb_main_loop
from methods.punidb.train import main_loop as punidb_main_loop

from methods.ed_ebm.train import main_loop as ed_ebm_main_loop
from methods.ebm_runidb.train import main_loop as ebm_runidb_main_loop

from methods.cd_ebm.train import main_loop as cd_ebm_main_loop
from methods.cd_runi_inter.train import main_loop as cd_runi_inter_main_loop
from methods.dataq_dfs.train import main_loop as dataq_dfs_main_loop
from methods.dataq_dfs_ebm.train import main_loop as dataq_dfs_ebm_main_loop
from methods.uniform_ebm.train import main_loop as uniform_ebm_main_loop

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--methods', type=str, default='punidb', 
        choices=[
            'punidb', 'runidb',
            'ed_ebm', 'ebm_runidb',
            'cd_ebm', 'cd_runi_inter',
            'dataq_dfs', 'dataq_dfs_ebm',
            'uniform_ebm'
        ],
    )

    parser.add_argument('--data_name', type=str, default='moons')
    parser.add_argument('--discrete_dim', type=int, default=16)
    parser.add_argument('--gibbs_num_rounds', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=5)
    parser.add_argument('--cd_alpha', type=float, default=0.1)
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--mps', action='store_true', help='Try using apple silicon chip (if no gpu available)')
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--ebm_lr', type=float, default=1e-3)
    parser.add_argument('--dfs_lr', type=float, default=1e-3)

    parser.add_argument("--gfn_type", type=str)
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])

    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu; 0 - ?: specific gpu index')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--num_epochs', default=1000, type=int, help='num epochs')
    parser.add_argument('--surrogate_iter_per_epoch', default=1, type=int, help='surrogate sampler: num iterations per epoch')
    parser.add_argument('--ebm_iter_per_epoch', default=1, type=int, help='ebm: num iterations per epoch')
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--plot_every', type=int, default=2500)
    parser.add_argument('--experiment_name', default="", type=str, help='unique experiment name for meta data')
    parser.add_argument('--verbose', default=False, type=bool, help='evaluate final nll and mmd')

    parser.add_argument("--with_mh", "--wm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rand_k", "--rk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_k", "--lk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--warmup_k", "--wk", type=lambda x: int(float(x)), default=0, help="need to use w/ lin_k")
    parser.add_argument("--K", type=int, default=-1, help="for gfn back forth negative sample generation")
    parser.add_argument("--eta", type=int, default=1, help="eta to determine level of stochasticity in CTMC")

    parser.add_argument("--final_ais_samples", type=int, default=1000000)
    parser.add_argument("--intermediate_ais_samples", type=int, default=10000)
    parser.add_argument("--final_ais_num_steps", type=int, default=1000)
    parser.add_argument("--intermediate_ais_num_steps", type=int, default=100)
    parser.add_argument("--mixture", type=float, default=1)
    parser.add_argument('--pretrained_ebm', type=str, default='imaginary file')
    parser.add_argument('--compute_mmd_base_stats', default=False, type=bool, help='compute mmd variance and mean as a benchmark')

    args = parser.parse_args()

    gpu = args.gpu

    if args.mps:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
        else:
            args.device = torch.device("mps")
            print('using mps')
    else:
        if torch.cuda.is_available() and gpu >= 0:
            torch.cuda.set_device(gpu)
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            args.device = torch.device('cuda:' + str(gpu))
            print('use gpu indexed: %d' % gpu)
        else:
            args.device = torch.device('cpu')
            print('use cpu')

    #set paths
    args.save_dir = f'./methods/{args.methods}/experiments/{args.data_name}'
    os.makedirs(args.save_dir, exist_ok=True)

    # experiment_idx_path = f'{args.save_dir}/experiment_idx.json'
    # if os.path.exists(experiment_idx_path) and os.path.getsize(experiment_idx_path) > 0:
    #     try:
    #         with open(experiment_idx_path, 'r') as file:
    #             experiment_idx = json.load(file)
    #     except json.JSONDecodeError:
    #         print("Warning: JSON file is corrupted. Initializing a new experiment index.")
    #         experiment_idx = {}
    # else:
    #     experiment_idx = {}

    # args.idx = 0
    # while True:
    #     if str(args.idx) in experiment_idx.keys():
    #         args.idx += 1
    #     else:
    #         break

    args.completed = False
    args.mmd_mean = None
    args.mmd_var = None

    start_time = datetime.datetime.now()
    args.start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

    args.index_path = f'{args.save_dir}/experiment_idx.json'

    # Save the updated experiment index to the file
    # experiment_idx[args.idx] = convert_namespace_to_dict(args)
    # with open(experiment_idx_path, 'w') as file:
    #     json.dump(experiment_idx, file, indent=4)

    # print(f"Experiment meta data saved to {args.save_dir}experiment_idx.json")

    return args

def plot_binary_data_samples(db, args):
    data = utils.float2bin(db.gen_batch(1000), args.bm, args.discrete_dim, args.int_scale)
    float_data = utils.bin2float(data.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
    utils.plot_samples(float_data, f'{args.sample_path}/source_data_sample.png', im_size=4.1)

def plot_categorical_data_samples(db, args):
    data = utils.ourfloat2base(db.gen_batch(1000), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    float_data = utils.ourbase2float(data.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    utils.plot_samples(float_data, f'{args.sample_path}/source_data_sample.png', im_size=4.1)


if __name__ == '__main__':
    args = get_args()

    if args.vocab_size == 2:
        args.discrete_dim = 32
        db, bm, inv_bm = utils.setup_data(args)
        args.bm = bm
        args.inv_bm = inv_bm
    else:
        db = utils.our_setup_data(args)

    if args.compute_mmd_base_stats:
        N = 25
        args.mmd_mean, args.mmd_var = compute_mmd_base_stats(args, N, db)

    log_args(args.methods, args.data_name, args)

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

    print(f'Saving experiment data to {args.save_dir}/{args.data_name}_{args.exp_n}')

    if args.vocab_size == 2:
        plot_binary_data_samples(db, args)
    else:
        plot_categorical_data_samples(db, args)

    main_fn = eval(f'{args.methods}_main_loop')

    main_fn(db, args, verbose=args.verbose)
