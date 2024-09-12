import os
import torch
import shutil
import argparse
from tqdm import tqdm
import json
import numpy as np
import datetime
import torchvision
import utils.mlp as mlp

from velo_dfm.train import gen_samples as dfm_gen_samples
from velo_dfs.train import gen_samples as dfs_gen_samples
from utils.utils import plot as toy_plot
from utils.model import ResNetFlow, EBM, MLPModel, MLPScore
from utils.toy_data_lib import get_db
from utils.sampler import GibbsSampler
from utils.eval import log
from utils.utils import get_sampler
import utils.vamp_utils as vamp_utils

import pandas as pd
import utils.ais as ais


def main(args):
    #make ebm
    print('starting evaluation...')
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = (init_batch.mean(0) * (1. - 2 * eps) + eps).to(args.device)

    net = mlp.mlp_ebm(np.prod(args.input_size), 256)

    ebm_model = EBM(net, init_mean).to(args.device)

    ebm_model.to(args.device)

    #save dir
    file_name = os.path.basename(args.ebm_model)  # file_name.txt
    model_name = os.path.splitext(file_name)[0]  #
    components = args.ebm_model.split(os.sep)
    save_dir = os.sep.join(components[:-2])


    #load
    try:
        ebm_model.load_state_dict(torch.load(args.ebm_model))
        print(f'successfully loaded model...')
    except FileNotFoundError as e:
        print('Model not found.')
        sys.exit(1)
    ebm_model.eval()

    # wrap model for annealing
    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(args.device))

    # get sampler
    args.sampler = 'dmala'
    sampler = get_sampler(args)

    logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ebm_model, init_dist, sampler,
                                                                train_loader, val_loader, test_loader,
                                                                preprocess, args.device,
                                                                args.eval_sampling_steps,
                                                                args.n_samples, viz_every=args.viz_every)

    log_entry = {}
    log_entry['data_name'] = args.dataset_name
    log_entry['model_name'] = args.ebm_model
    log_entry['eval_sampling_steps'] = args.eval_sampling_steps
    log_entry['n_samples'] = args.n_samples
    log_entry['logZ'] = logZ.item()
    log_entry['EMA Train log-likelihood'] = train_ll.item()
    log_entry['EMA Valid log-likelihood'] = val_ll.item()
    log_entry['EMA Test log-likelihood'] = test_ll.item()
    log_entry['eval step size'] = args.eval_step_size

    df_log_entry = pd.DataFrame([log_entry])
    log_path = f'{save_dir}/final_eval.csv'
    df_log_entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    print(f'completed and saved to {log_path}')

if __name__ == "__main__":
    print('start')
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--ckpt_path', type=str, default=None)
    # models
    parser.add_argument('--ebm_model', type=str)
    parser.add_argument('--base_dist', type=int, default=1)
    parser.add_argument('--source', type=str, default='uniform')
    parser.add_argument('--discrete_dim', type=int, default=28*28)
    parser.add_argument('--model_has_noise', type=int, default=0)
    parser.add_argument('--enable_backward', type=int, default=0)
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--vocab_size_with_mask', type=int, default=3)
    parser.add_argument('--relu', type=int, default=0)
    parser.add_argument('--scheduler_type', type=str, default='linear')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)


    # mcmc
    parser.add_argument('--sampler', type=str, default='dmala')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=40)
    parser.add_argument('--steps_per_iter', type=int, default=1)
    parser.add_argument('--eval_sampling_steps', type=int, default=30000)
    parser.add_argument('--eval_step_size', type=float, default=0.1)
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--sampler_temp', type=float, default=2.)

    # training
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--viz_every', type=int, default=1000)

    args = parser.parse_args()

    real_dataset_list = ['static_mnist','dynamic_mnist','omniglot','caltech']
    for name in real_dataset_list:
        if name in args.ebm_model:
            args.dataset_name = name

    print('sampler',args.sampler,'lr',args.eval_step_size,'data',args.dataset_name)
    main(args)
