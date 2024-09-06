import os
import torch
import shutil
import argparse
import json
import numpy as np
import datetime

from velo_dfm.train import gen_samples as dfm_gen_samples
from velo_dfs.train import gen_samples as dfs_gen_samples
from utils.utils import plot as toy_plot
from utils.model import MLPScore, EBM
from utils.toy_data_lib import get_db
from utils.sampler import GibbsSampler
from utils.utils import get_batch_data


def load_and_sample(args):

    db = get_db(args)
    plot = lambda p, x: toy_plot(p, x, args)

    eps = 1e-2
    samples = get_batch_data(db, args, batch_size=10000)
    init_mean = torch.from_numpy(np.mean(samples, axis=0) * (1. - 2 * eps) + eps).to(args.device)
    
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    model = EBM(net, init_mean).to(args.device)
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)

    plot = lambda p, x: toy_plot(p, x, args)
    try:
        model.load_state_dict(torch.load(args.model))
        print(f'successfully loaded model...')
    except FileNotFoundError as e:
        print('Model not found.')
        sys.exit(1)
    model.eval()
    model.to(args.device)

    samples = gibbs_sampler(model, num_rounds=100, num_samples=2500).to('cpu').detach()
    print('successfully generated samples')

    file_name = os.path.basename(args.model)  # file_name.txt
    model_name = os.path.splitext(file_name)[0]  #
    components = args.model.split(os.sep)
    remaining_path = os.sep.join(components[:-2])
    plot(f'{remaining_path}/samples/samples_{model_name}.png', torch.tensor(samples).float())

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('--model', type=str)
    parser.add_argument('--source', type=str, default='mask')
    parser.add_argument('--discrete_dim', type=int, default=32)
    parser.add_argument('--model_has_noise', type=int, default=1)
    parser.add_argument('--enable_backward', type=int, default=0)
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--vocab_size_with_mask', type=int, default=3)
    parser.add_argument('--relu', type=int, default=0)
    parser.add_argument('--scheduler_type', type=str, default='linear')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--delta_t', type=float, default='0.05')
    parser.add_argument('--impute_self_connections', type=int, default=1)



    args = parser.parse_args()

    toy_dataset_list = ['swissroll','circles','moons','8gaussians','pinwheel','2spirals','checkerboard']
    for name in toy_dataset_list:
        if name in args.model:
            args.dataset_name = name

    args.data_dir = f'./methods/datasets' 

    _ = get_db(args)


    if 'dfm' in args.model:
        args.dfs = 0
    else:
        args.dfs = 1

    return args

if __name__ == '__main__':
    args = get_args()
    load_and_sample(args)
