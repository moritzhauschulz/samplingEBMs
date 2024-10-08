import argparse
import torch
import numpy as np
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import vamp_utils
import mlp
from pcd_ebm_ema import get_sampler, EBM
import ais
import pandas as pd


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    my_print("Loading data")
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    # make model
    my_print("Making Model")
    if args.model.startswith("mlp-"):
        nint = int(args.model.split('-')[1])
        net = mlp.mlp_ebm(np.prod(args.input_size), nint)
    elif args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    elif args.model.startswith("cnn-"):
        nint = int(args.model.split('-')[1])
        net = mlp.MNISTConvNet(nint)
    else:
        raise ValueError("invalid model definition")

    # get data mean and initialize buffer
    my_print("Getting init batch")
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps


    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    # d = torch.load("{}/best_ckpt_{}_{}.pt".format(args.save_dir,args.eval_sampler,args.eval_step_size))
    d = torch.load("{}/best_ckpt_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.eval_sampler,args.eval_step_size))
    # d = torch.load("{}/best_ckpt_{}_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.eval_sampler,args.eval_step_size,args.sampling_steps))

    if args.ema:
        model.load_state_dict(d['ema_model'])
    else:
        model.load_state_dict(d['model'])
    buffer = d['buffer']

    # wrap model for annealing
    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(device))

    # get sampler
    sampler = get_sampler(args)

    my_print(device)
    my_print(model)
    my_print(sampler)

    logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(model, init_dist, sampler,
                                                                train_loader, val_loader, test_loader,
                                                                preprocess, device,
                                                                args.eval_sampling_steps,
                                                                args.n_samples, viz_every=args.viz_every)

    log_path = '../datasets/real_datasets_ais_stats.csv'
    log_entry = {}
    log_entry['data_name'] = args.dataset_name
    log_entry['model_name'] = "{}/best_ckpt_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.eval_sampler,args.eval_step_size)
    log_entry['eval_sampling_steps'] = args.eval_sampling_steps
    log_entry['n_samples'] = args.n_samples
    log_entry['logZ'] = logZ.item()
    log_entry['EMA Train log-likelihood'] = train_ll.item()
    log_entry['EMA Valid log-likelihood'] = val_ll.item()
    log_entry['EMA Test log-likelihood'] = test_ll.item()

    df_log_entry = pd.DataFrame([log_entry])
    df_log_entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="./figs/ebm")
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--ckpt_path', type=str, default=None)
    # data generation
    parser.add_argument('--n_out', type=int, default=3)  # potts
    # models
    parser.add_argument('--model', type=str, default='mlp-256')
    parser.add_argument('--base_dist', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--p_control', type=float, default=0.0)
    # mcmc
    parser.add_argument('--eval_sampler', type=str, default='gibbs')
    parser.add_argument('--sampler', type=str, default='gwg',help='sampler used in AIS. Use GWG for fair comparison to previous methods')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--steps_per_iter', type=int, default=1)
    parser.add_argument('--eval_sampling_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--buffer_init', type=str, default='mean')
    parser.add_argument('--eval_step_size', type=float, default=0.08)
    parser.add_argument('--step_size', type=float, default=0.15)
    # training
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--warmup_iters', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=.0)

    args = parser.parse_args()
    args.device = device
    print('sampler',args.eval_sampler,'lr',args.eval_step_size)
    main(args)
