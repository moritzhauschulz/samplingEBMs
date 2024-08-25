import argparse
import mlp
import torch
import numpy as np
import samplers
import block_samplers
import torch.nn as nn
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import vamp_utils
import ais
import copy
import time

def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_sampler(args):
    data_dim = np.prod(args.input_size)
    if args.input_type == "binary":
        if args.sampler == "gibbs":
            sampler = samplers.PerDimGibbsSampler(data_dim, rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = samplers.PerDimGibbsSampler(data_dim, rand=True)
        elif args.sampler.startswith("bg-"):
            block_size = int(args.sampler.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(data_dim, block_size)
        elif args.sampler.startswith("hb-"):
            block_size, hamming_dist = [int(v) for v in args.sampler.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(data_dim, block_size, hamming_dist)
        elif args.sampler == "gwg":
            sampler = samplers.DiffSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif args.sampler.startswith("gwg-"):
            n_hops = int(args.sampler.split('-')[1])
            sampler = samplers.MultiDiffSampler(data_dim, 1, approx=True, temp=2., n_samples=n_hops)
        elif args.sampler == "dmala":
            sampler = samplers.LangevinSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=True)

        elif args.sampler == "dula":
            sampler = samplers.LangevinSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=False)

        
        else:
            raise ValueError("Invalid sampler...")
    else:
        if args.sampler == "gibbs":
            sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=True)
        elif args.sampler == "gwg":
            sampler = samplers.DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.)
        else:
            raise ValueError("invalid sampler")
    return sampler

class EBM(nn.Module):
    def __init__(self, net, mean=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            base_dist = torch.distributions.Bernoulli(probs=self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp + bd

def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')
        logger.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
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
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps
    if args.buffer_init == "mean":
        if args.input_type == "binary":
            init_dist = torch.distributions.Bernoulli(probs=init_mean)
            buffer = init_dist.sample((args.buffer_size,))
        else:
            buffer = None
            raise ValueError("Other types of data not yet implemented")

    elif args.buffer_init == "data":
        all_inds = list(range(init_batch.size(0)))
        init_inds = np.random.choice(all_inds, args.buffer_size)
        buffer = init_batch[init_inds]
    elif args.buffer_init == "uniform":
        buffer = (torch.ones(args.buffer_size, *init_batch.size()[1:]) * .5).bernoulli()
    else:
        raise ValueError("Invalid init")

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    ema_model = copy.deepcopy(model)


    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d['model'])
        ema_model.load_state_dict(d['ema_model'])
        #optimizer.load_state_dict(d['optimizer'])
        #buffer = d['buffer']

    # move to cuda
    model.to(device)
    ema_model.to(device)

    # get sampler
    sampler = get_sampler(args)

    samples = init_dist.sample((args.batch_size,)).to(device)
    for i in range(args.eval_sampling_steps):
        samples = sampler.step(samples.detach(), model).detach()

        if (i + 1) % args.viz_every == 0:
            plot(f'{args.save_dir}/EBM_samples_steps_{i}.png', samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="./figs/MCMC_samples_by_steps/")
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--ckpt_path', type=str, default='./figs/ebm/best_ckpt_static_mnist_dula_0.1.pt')
    # data generation
    parser.add_argument('--n_out', type=int, default=3)     # potts
    # models
    parser.add_argument('--model', type=str, default='mlp-256')
    parser.add_argument('--base_dist', action='store_true')
    parser.add_argument('--p_control', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--ema', type=float, default=0.999)
    # mcmc
    parser.add_argument('--sampler', type=str, default='dula')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--reinit_freq', type=float, default=0.0)
    parser.add_argument('--eval_sampling_steps', type=int, default=10000)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--buffer_init', type=str, default='mean')
    parser.add_argument('--step_size', type=float, default=0.08)
    # training
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--warmup_iters', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=.0)

    args = parser.parse_args()
    args.device = device
    main(args)
    
