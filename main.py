import os
import torch
import shutil
import argparse
import numpy as np

from utils import utils
from methods.runidb.train import main_loop as runidb_main_loop
from methods.punidb.train import main_loop as punidb_main_loop

from methods.ed_ebm.train import main_loop as ed_ebm_main_loop
from methods.ebm_runidb.train import main_loop as ebm_runidb_main_loop

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--methods', type=str, default='punidb', 
        choices=[
            'punidb', 'runidb',
            'ed_ebm', 'ebm_runidb',
        ],
    )

    parser.add_argument('--data_name', type=str, default='moons')
    parser.add_argument('--discrete_dim', type=int, default=16)
    parser.add_argument('--vocab_size', type=int, default=5)
    parser.add_argument('--cd_alpha', type=float, default=0.1)

    parser.add_argument('--mps', action='store_true', help='Try using apple silicon chip (if no gpu available)')
    parser.add_argument('--noise', type=int, default=1)
    
    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu; 0 - ?: specific gpu index')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--num_epochs', default=1000, type=int, help='num epochs')
    parser.add_argument('--iter_per_epoch', default=100, type=int, help='num iterations per epoch')
    parser.add_argument('--epoch_save', default=100, type=int, help='num epochs between save')

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

    if args.methods == 'punidb':
        noise_string = f'_noise={args.noise}'
    else:
        noise_string = ""
    args.save_dir = f'./methods/{args.methods}{noise_string}/results/voc_size={args.vocab_size}/{args.data_name}'
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

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

    main_fn(db, args, verbose=True)
