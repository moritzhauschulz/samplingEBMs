import os
import sys
import copy
import torch
import utils.mlp as mlp
import numpy as np
import torchvision
from tqdm import tqdm
import time
from itertools import cycle
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import utils.utils as utils
from utils.utils import get_batch_data
from utils.utils import plot as toy_plot
from utils.utils import get_x0
from utils.eval import plot_weight_histogram
from utils.utils import get_optimal_temp

from utils.sampler import GibbsSampler
from utils.utils import get_sampler


import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion
from utils.eval import get_eval_timestamp
from utils.eval import exp_hamming_mmd
from utils.eval import rbf_mmd
from utils.eval import sampler_ebm_evaluation
from utils.eval import sampler_evaluation
from utils.ais import evaluate_sampler

from utils.toy_data_lib import get_db

from utils.model import ResNetFlow, EBM, MLPModel, MLPScore

from velo_dfm.train import gen_samples
from velo_dfm.train import compute_loss as compute_dfm_loss

def compute_loss(model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp=1):
    if args.optimal_temp:
        temp = get_optimal_temp(log_p_prob, log_q_prob, args, alg=1)
    else:
        temp = temp * args.temp_decay

    log_weights = log_p_prob.detach()/temp - log_q_prob.detach()
    if args.norm_by_mean and not args.norm_by_max:
        log_norm = torch.logsumexp(log_weights, dim=-1) - torch.log(torch.tensor(B)) #make this a moving average?
    elif args.norm_by_max and not args.norm_by_mean:
        log_norm = torch.max(log_weights)
    else:
        raise NotImplementedError('Must either normalize by mean or max to avoid inf.')

    weights = (log_weights - log_norm).exp()

    loss, _ = compute_dfm_loss(model, B, D, S, t, x1, x0, args, weights) 

    loss = loss.mean(dim=0)

    return loss, weights, temp


def main_loop(args, verbose=False):

    #set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.is_toy:
        main_loop_toy(args, verbose)
    else:
        main_loop_real(args, verbose)
    log_completion(args.methods, args.dataset_name, args)

def main_loop_real(args, verbose=False):

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                    p, normalize=True, nrow=int(x.size(0) ** .5))
    # load omniglot
    if args.source == 'omniglot':
        og_args = copy.deepcopy(args)
        og_args.dataset_name = 'omniglot'
        og_train_loader, og_val_loader, og_test_loader, og_args = vamp_utils.load_dataset(og_args)
        source_train_loader = copy.deepcopy(og_train_loader)
        #can use the same plot function...
    else:
        source_train_loader = copy.deepcopy(train_loader)

    def preprocess(data, args=args):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
            
    def get_independent_sample(loader, args=args):
        (x, _) = next(iter(loader))
        return preprocess(x, args)

    if args.q == 'data_mean':
        init_batch = []
        for x, _ in train_loader:
            init_batch.append(preprocess(x))
        init_batch = torch.cat(init_batch, 0)
        eps = 1e-2
        init_mean = (init_batch.mean(0) * (1. - 2 * eps) + eps).to(args.device)
        q_dist = torch.distributions.Bernoulli(probs=init_mean)
    elif args.q == 'random':
        q_dist = torch.distributions.Bernoulli(probs=0.5 * torch.ones((args.discrete_dim,)).to(args.device))

    # make dfs model
    dfs_model = ResNetFlow(64, args)
    ema_dfs_model = copy.deepcopy(dfs_model)
    optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # make ebm model
    if args.ebm_model.startswith("mlp-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.mlp_ebm(np.prod(args.input_size), nint)
    elif args.ebm_model.startswith("resnet-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    elif args.ebm_model.startswith("cnn-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.MNISTConvNet(nint)
    else:
        raise ValueError("invalid ebm_model definition")

    ebm_model = EBM(net, init_mean).to(args.device)
    try:
        d = torch.load(args.pretrained_ebm)
        ebm_model.load_state_dict(d['ema_model'])
        print(f'successfully loaded EBM...')
    except FileNotFoundError as e:
        print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
        sys.exit(1)
    ebm_model.eval()

    # move to cuda
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)
    ebm_model.to(args.device)

    #set temperature
    temp = args.start_temp

    start_time = time.time()
    cum_eval_time = 0

    itr = 1

    while itr <= args.num_itr:
        
        dfs_model.train()
        ebm_model.eval()
        pbar = tqdm(range(len(train_loader))) if verbose else range(len(train_loader))
    
        for _, (x_source, _) in zip(pbar, cycle(source_train_loader)):
            (B, D) = x_source.shape
            x1 = q_dist.sample((B,)).to(args.device).long()
            S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
            t = torch.rand((B,)).to(args.device)

            if args.source == 'data':
                x0 = preprocess(x_source).long().to(args.device)
            elif args.source == 'omniglot':
                x0 = preprocess(x_source, args=og_args).long().to(args.device)
            else:
                x0 = get_x0(B,D,S,args)
            
            log_p_prob = -ebm_model(x1.float())
            log_q_prob = q_dist.log_prob(x1.float()).sum(dim=-1).to(args.device)


            loss, weights, temp = compute_loss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if verbose:
                pbar.set_description(f'Itr {itr} Loss {loss.item()}, Temp {temp}')

            if (itr % args.itr_save == 0) or (itr == args.num_itr):
                eval_start_time = time.time()
                #save models
                torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{itr}.pt')
                torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{itr}.pt')

                #save samples
                if args.source == 'data':
                    xt = get_independent_sample(test_loader).long().to(args.device) 
                    plot(f'{args.sample_path}/source_{itr}.png', xt.float())
                elif args.source == 'omniglot':
                    xt = get_independent_sample(og_test_loader, args=og_args).long().to(args.device) 
                    plot(f'{args.sample_path}/source_{itr}.png', xt.float())
                else:
                    xt = None
                samples = gen_samples(dfs_model, args, batch_size=100, xt=xt)
                plot(f'{args.sample_path}/dfs_samples_{itr}.png', torch.tensor(samples).float())
                ema_samples = gen_samples(ema_dfs_model, args, batch_size=100, xt=xt)
                plot(f'{args.sample_path}/ema_dfs_samples_{itr}.png', torch.tensor(ema_samples).float())
                weights_dir = f'{args.plot_path}/weights_histogram_{itr}.png'
                if not os.path.exists(weights_dir):
                    plot_weight_histogram(weights, output_dir=weights_dir)
                
                #save log
                log_entry = {'itr':None,'timestamp':None}
                log_entry['loss'] = loss.item()
                log_entry['temp'] = temp
                log_entry['mean_weight'] = weights.mean().item()

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp)
            
            if (itr) % args.eval_every == 0 or (itr) == args.num_itr:
                eval_start_time = time.time()
                torch.save(dfs_model.state_dict(), f'{args.ckpt_path}dfs_model_{itr}.pt')
                
                log_entry = {'itr':None,'timestamp':None}
                log_entry['loss'] = loss.item()

                #sampler eval here 
                batches = []
                for i in range(10):
                    samples = torch.from_numpy(gen_samples(dfs_model, args, batch_size=100, xt=xt)).to(args.device).float()
                    batches.append(samples)
                sample_ll = evaluate_sampler(args, ebm_model, batches)
                log_entry['sample_ll'] = sample_ll
                
                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')
            
            itr += 1

def main_loop_toy(args, verbose=False):

    # load data
    db = get_db(args)
    plot = lambda p, x: toy_plot(p, x, args)

    if args.q == 'data_mean':
        samples = get_batch_data(db, args, batch_size=10000)
        eps = 1e-2
        init_mean = torch.from_numpy(np.mean(samples, axis=0) * (1. - 2 * eps) + eps).to(args.device)
        q_dist = torch.distributions.Bernoulli(probs=init_mean)
    elif args.q == 'random':
        q_dist = torch.distributions.Bernoulli(probs=0.5 * torch.ones((args.discrete_dim)).to(args.device))
    else:
        print(f'Type {args.q} of q distribution not supported...')
        sys.exit(0)

    # make model
    dfs_model = MLPModel(args).to(args.device)
    ema_dfs_model = copy.deepcopy(dfs_model)
    optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # make ebm model
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    ebm_model = EBM(net).to(args.device)
    try:
        ebm_model.load_state_dict(torch.load(f'./{args.pretrained_ebm}'))
        print(f'successfully loaded EBM...')
    except FileNotFoundError as e:
        print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
        sys.exit(1)
    ebm_model.eval()
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/initial_heat.png', args)

    #initial samples with DLP
    sampler = get_sampler(args) #for eval
    model = lambda x: -ebm_model(x)
    EBM_samples = torch.randint(0,2,(1000,32)).to(args.device).float().detach()
    MCMC_pbar = tqdm(range(args.MCMC_refinement))
    for d in MCMC_pbar:
        EBM_samples = sampler.step(EBM_samples.detach(), model).detach()
        MCMC_pbar.set_description('MCMC Sampling in Progress...')
        if len(sampler.a_s) > 0:
            print(f'acceptance prob: {sampler.a_s[-1]}')
    EBM_samples = EBM_samples.cpu().detach()
    plot(f'{args.sample_path}/EBM_samples_DLP.png', EBM_samples)

    # move to cuda
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)
    ebm_model.to(args.device)

    #set temperature
    temp = args.start_temp

    start_time = time.time()
    cum_eval_time = 0
    itr = 1
    
    pbar = tqdm(range(1, args.num_itr + 1)) if verbose else range(1,args.num_itr + 1)
    for itr in pbar:
        dfs_model.train()
        ebm_model.eval()

        (B, D) = args.batch_size, args.discrete_dim
        x1 = q_dist.sample((B,)).to(args.device).long()
        S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
        t = torch.rand((B,)).to(args.device)

        if args.source == 'data':
            x0 = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
        else:
            x0 = get_x0(B,D,S,args)

        log_p_prob = -ebm_model(x1.float())

        log_q_prob = q_dist.log_prob(x1.float()).sum(dim=-1).to(args.device)

        loss, weights, temp = compute_loss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update ema_model
        for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

        if verbose:
            pbar.set_description(f'Itr {itr}, Loss {loss.item()}, Temp {temp}')

        if (itr % args.itr_save == 0) or (itr == args.num_itr):
            eval_start_time = time.time()

            #save samples
            if args.source == 'data':
                xt = torch.from_numpy(get_batch_data(db, args, batch_size = 2500)).to(args.device)
                plot(f'{args.sample_path}/source_{itr}.png', xt)
            else:
                xt = None
            samples = gen_samples(dfs_model, args, batch_size = 2500, xt=xt)
            plot(f'{args.sample_path}/dfs_samples_{itr}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_dfs_model, args, batch_size = 2500, xt=xt)
            plot(f'{args.sample_path}/ema_dfs_samples_{itr}.png', torch.tensor(ema_samples).float())
            weights_dir = f'{args.plot_path}/weights_histogram_{itr}.png'
            if not os.path.exists(weights_dir):
                plot_weight_histogram(weights, output_dir=weights_dir)
            
            #save models
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{itr}.pt')
            torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{itr}.pt')

            #log
            log_entry = {'itr':None,'timestamp':None}
            log_entry['loss'] = loss.item()
            log_entry['temp'] = temp
            log_entry['mean_weight'] = weights.mean().item()

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, itr, timestamp)


        if args.eval_on:
            if (itr % args.eval_every == 0) or (itr == args.num_itr):

                eval_start_time = time.time()
                log_entry = {'itr':None,'timestamp':None}
                log_entry['loss'] = loss.item()
                log_entry['temp'] = temp
                log_entry['mean_weight'] = weights.mean().item()

                #compute mmds 1/2
                hamming_mmd, bandwidth, euclidean_mmd, sigma = sampler_evaluation(args, db, dfs_model, gen_samples)

                #log
                log_entry['sampler_hamming_mmd'], log_entry['sampler_bandwidth'] = hamming_mmd, bandwidth
                log_entry['sampler_euclidean_mmd'], log_entry['sampler_sigma'] = euclidean_mmd, sigma

                #log
                log_entry['sampler_hamming_mmd'], log_entry['sampler_bandwidth'] = hamming_mmd, bandwidth
                log_entry['sampler_euclidean_mmd'], log_entry['sampler_sigma'] = euclidean_mmd, sigma

                #compute mmds 2/2
                hamming_mmd, bandwidth, euclidean_mmd, sigma = sampler_ebm_evaluation(args, db, dfs_model, gen_samples, ebm_model)

                #log
                log_entry['sampler_ebm_hamming_mmd'], log_entry['sampler_ebm_bandwidth'] = hamming_mmd, bandwidth
                log_entry['sampler_ebm_euclidean_mmd'], log_entry['sampler_ebm_sigma'] = euclidean_mmd, sigma

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)
                log(args, log_entry, itr, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')
