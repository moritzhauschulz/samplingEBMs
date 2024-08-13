import os
import torch
import shutil
import argparse
import json
import numpy as np
import datetime

from utils.eval import log_args
from velo_mask_dfm.train import load_and_sample as velo_mask_dfm_load_and_sample

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--methods', type=str, default='punidb', 
        choices=[
            'punidb', 'runidb','velo_mask_dfm', 'velo_uni_dfm',
            'velo_mask_dfs', 'velo_uni_dfs', 'velo_mask_efm', 'velo_uni_efm',
            'velo_mask_edfs', 'velo_uni_edfs', 'velo_mask_efm_ebm'
        ],
    )
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--impute_self_connections', type=int, default=1, choices=[0,1])
    parser.add_argument('--loss_weight', type=int, default=1, choices=[0,1])
    parser.add_argument('--discrete_dim', type=int, default=28*28)
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu; 0 - ?: specific gpu index')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')


    args = parser.parse_args()

    args.vocab_size_with_mask = args.vocab_size + 1
    args.input_size = [1, 28, 28]


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

    return args

if __name__ == '__main__':
    args = get_args()
    main_fn = eval(f'{args.methods}_load_and_sample')
    main_fn(args)
