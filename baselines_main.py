import os
import torch
import shutil
import argparse
import json
import numpy as np

from utils import utils
from methods.eb_gfn.train import main_loop as eb_gfn_main_loop

def convert_namespace_to_dict(args):
    args_dict = vars(args).copy()  # Convert Namespace to dictionary
    # Handle non-serializable objects
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)
    args_dict.pop('bm', None)
    args_dict.pop('inv_bm', None)
    return args_dict

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')
    
    parser.add_argument('--methods', type=str, default='eb_gfn', 
        choices=[
            'eb_gfn'
        ],
    )

    #gpu
    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu; 0 - ?: specific gpu index')
    parser.add_argument('--mps', action='store_true', help='Try using apple silicon chip (if no gpu available)')


    #data
    parser.add_argument('--data_name', type=str, default='moons')  # 2spirals 8gaussians pinwheel circles moons swissroll checkerboard
    parser.add_argument('--discrete_dim', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=2)

    # training
    parser.add_argument('--n_iters', "--ni", type=lambda x: int(float(x)), default=1e5)
    parser.add_argument('--batch_size', "--bs", type=int, default=128)
    parser.add_argument('--print_every', "--pe", type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument("--ebm_every", "--ee", type=int, default=1, help="EBM training frequency")

    # for GFN
    parser.add_argument("--type", type=str)
    parser.add_argument("--hid", type=int, default=512)
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
    args = parser.parse_args()

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

    device = args.device

    #set paths
    args.save_dir = f'./methods/{args.methods}/experiments/{args.data_name}/'
    os.makedirs(args.save_dir, exist_ok=True)

    experiment_idx_path = f'{args.save_dir}/experiment_idx.json'
    if os.path.exists(experiment_idx_path) and os.path.getsize(experiment_idx_path) > 0:
        try:
            with open(experiment_idx_path, 'r') as file:
                experiment_idx = json.load(file)
        except json.JSONDecodeError:
            print("Warning: JSON file is corrupted. Initializing a new experiment index.")
            experiment_idx = {}
    else:
        experiment_idx = {}
    
    args.idx = 0
    while True:
        if str(args.idx) in experiment_idx.keys():
            args.idx += 1
        else:
            break
    
    args.ckpt_path = f'{args.save_dir}/{args.data_name}_{str(args.idx)}/ckpts/'
    args.plot_path = f'{args.save_dir}/{args.data_name}_{str(args.idx)}/plots/'
    args.sample_path = f'{args.save_dir}/{args.data_name}_{str(args.idx)}/samples/'

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

    # Save the updated experiment index to the file
    experiment_idx[args.idx] = convert_namespace_to_dict(args)
    with open(experiment_idx_path, 'w') as file:
        json.dump(experiment_idx, file, indent=4)

    print(f"Experiment meta data saved to {args.save_dir}experiment_idx.json")

    print("Device:" + str(device))
    print("Args:" + str(args))

    return args

def plot_binary_data_samples(db, args):
    data = utils.float2bin(db.gen_batch(1000), args.bm, args.discrete_dim, args.int_scale)
    float_data = utils.bin2float(data.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
    utils.plot_samples(float_data, f'{args.save_dir}/data_sample.pdf', im_size=4.1, im_fmt='pdf')

def plot_categorical_data_samples(db, args):
    data = utils.ourfloat2base(db.gen_batch(1000), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    float_data = utils.ourbase2float(data.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    utils.plot_samples(float_data, f'{args.save_dir}/data_sample.pdf', im_size=4.1, im_fmt='pdf')

if __name__ == '__main__':
    args = get_args()

    if args.vocab_size == 2:
        args.discrete_dim = 32
        db, bm, inv_bm = utils.setup_data(args)
        args.bm = bm
        args.inv_bm = inv_bm
        plot_binary_data_samples(db, args)
    else:
        db = utils.our_setup_data(args)
        plot_categorical_data_samples(db, args)

    main_fn = eval(f'{args.methods}_main_loop')

    main_fn(db, args)
