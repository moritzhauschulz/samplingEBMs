import os
import sys
import copy
import torch
import utils.mlp as mlp
import numpy as np
import torchvision
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import utils.utils as utils
from utils.utils import get_batch_data
from utils.utils import plot as toy_plot
from utils.utils import get_x0
from utils.eval import plot_weight_histogram

from utils.sampler import GibbsSampler

import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion
from utils.eval import get_eval_timestamp
from utils.eval import exp_hamming_mmd
from utils.eval import rbf_mmd
from utils.toy_data_lib import get_db

from utils.model import ResNetFlow, EBM, MLPModel, MLPScore

from velo_dfm.train import gen_samples

def compute_loss(model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp=1):
    if args.source == 'mask':
        M = S - 1

    if args.scheduler_type == 'quadratic_noise':
        x_noise = torch.randint(0, S, (B, D)).to(args.device)
    else:
        x_noise = None
    
    if args.scheduler_type == 'linear':
        kappa1 = t
    elif args.scheduler_type == 'quadratic':
        kappa1 = torch.square(t)
    elif args.scheduler_type == 'quadratic_noise':
        kappa1 = torch.square(t)
        kappa2 = t - torch.square(t)

    xt = x1.clone()
    mask0 = torch.rand((B,D)).to(args.device) < (1 - kappa1[:, None])
    xt[mask0] = x0[mask0]
    if args.scheduler_type == 'quadratic_noise':
        mask_noise = torch.rand((B,D)).to(args.device) < (kappa2/(1 - kappa1))[:, None]
        mask_noise = mask_noise & mask0
        xt[mask_noise] = x_noise[mask_noise]
    
    x1_logits, noise_logits = model(xt, t)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1)
    if args.scheduler_type == 'quadratic_noise':
        noise_loss = F.cross_entropy(noise_logits.transpose(1,2), x_noise, reduction='none').sum(dim=-1)
        loss = loss + noise_loss
    

    log_weights = log_p_prob.detach() - log_q_prob.detach()

    if args.optimal_temp:
        #here, we assume q is a normalized distribution
        max_index = torch.argmax(log_weights)
        if args.optimal_temp_use_median:
            _, sorted_indices = torch.sort(log_weights)
            median_index = sorted_indices[len(log_weights) // 2]
            temp_t = 1/(torch.log(torch.tensor(args.optimal_temp_diff)) + log_q_prob[median_index] - log_q_prob[max_index])  * (log_p_prob[median_index] - log_p_prob[max_index])
        else:
            weights = log_weights.exp()
            mean_value = torch.mean(weights.float())
            diff = torch.abs(weights.float() - mean_value)
            mean_index = torch.argmin(diff) #lower complexity then median
            temp_t = 1/(torch.log(torch.tensor(args.optimal_temp_diff)) + log_q_prob[mean_index] - log_q_prob[max_index])  * (log_p_prob[mean_index] - log_p_prob[max_index])

        if temp_t < 1e-10:
            print(f'\n Reset temp_t to 1, which was at {temp_t}... \n', flush=True)
            temp_t = torch.tensor(1)

        temp = (args.optimal_temp_ema * temp + (1 - args.optimal_temp_ema) * temp_t).cpu().detach().item()
    else:
        temp = temp * args.temp_decay

    log_weights = log_p_prob.detach()/temp - log_q_prob.detach()
    if args.norm_by_sum and not args.norm_by_max:
        log_norm = torch.logsumexp(log_weights, dim=-1) #make this a moving average?
    elif args.norm_by_max and not args.norm_by_sum:
        log_norm = torch.max(log_weights)
    else:
        raise NotImplementedError('Must either normalize by sum or max to avoid inf.')

    weights = (log_weights - log_norm).exp()

    loss = weights * loss #the math is wrong?

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
        og_args.dataset_name == 'omniglot'
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
        q_dist = torch.distributions.Bernoulli(probs=0.5 * torch.tensor((args.discrete_dim)).to(args.device))

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
        net = mlp.ResNetEBM(64)
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

    for epoch in range(1, args.num_epochs + 1):
        
        dfs_model.train()
        ebm_model.eval()
        pbar = tqdm(source_train_loader) if verbose else source_train_loader
    
        for it, (x_source, _) in enumerate(pbar):
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
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}, Temp {temp}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()
            #save models
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')
            torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{epoch}.pt')

            #save samples
            if args.source == 'data':
                xt = get_independent_sample(test_loader).long().to(args.device) 
                plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
            elif args.source == 'omniglot':
                xt = get_independent_sample(og_test_loader, args=og_args).long().to(args.device) 
                plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
            else:
                xt = None
            samples = gen_samples(dfs_model, args, batch_size=100, xt=xt)
            plot(f'{args.sample_path}/dfs_samples_{epoch}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_dfs_model, args, batch_size=100, xt=xt)
            plot(f'{args.sample_path}/ema_dfs_samples_{epoch}.png', torch.tensor(ema_samples).float())
            weights_dir = f'{args.plot_path}/weights_histogram_{epoch}.png'
            if not os.path.exists(weights_dir):
                plot_weight_histogram(weights, output_dir=weights_dir)
            
            #save log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['loss'] = loss.item()
            log_entry['temp'] = temp
            log_entry['mean_weight'] = weights.mean().item()

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, epoch, timestamp)

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
        q_dist = torch.distributions.Bernoulli(probs=0.5 * torch.tensor((args.discrete_dim).to(args.device)))
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

    # move to cuda
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)
    ebm_model.to(args.device)

    #set temperature
    temp = args.start_temp

    start_time = time.time()
    cum_eval_time = 0

    pbar = tqdm(range(1, args.num_epochs + 1)) if verbose else range(1,args.num_epochs + 1)
    for epoch in pbar:
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
            pbar.set_description(f'Epoch {epoch}, Loss {loss.item()}, Temp {temp}')

        if (epoch % args.plot_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            #save samples
            if args.source == 'data':
                xt = torch.from_numpy(get_batch_data(db, args, batch_size = 2500)).to(args.device)
                plot(f'{args.sample_path}/source_{epoch}.png', xt)
            else:
                xt = None
            samples = gen_samples(dfs_model, args, batch_size = 2500, xt=xt)
            plot(f'{args.sample_path}/dfs_samples_{epoch}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_dfs_model, args, batch_size = 2500, xt=xt)
            plot(f'{args.sample_path}/ema_dfs_samples_{epoch}.png', torch.tensor(ema_samples).float())
            weights_dir = f'{args.plot_path}/weights_histogram_{epoch}.png'
            if not os.path.exists(weights_dir):
                plot_weight_histogram(weights, output_dir=weights_dir)
            
            #save models
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')
            torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{epoch}.pt')

            #log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['loss'] = loss.item()
            log_entry['temp'] = temp
            log_entry['mean_weight'] = weights.mean().item()
            
            #compute mmds 1/2
            exp_hamming_mmd_list = []
            rbf_mmd_list = []
            for _ in range(10):
                if args.source == 'data':
                    xt = torch.from_numpy(get_batch_data(db, args, batch_size = 4000)).to(args.device)
                    plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
                else:
                    xt = None
                x = torch.from_numpy(gen_samples(dfs_model, args, batch_size = 4000, xt=xt)).to('cpu')
                y = get_batch_data(db, args, batch_size=4000)
                y = torch.from_numpy(np.float32(y)).to('cpu')
                hamming_mmd, bandwidth = exp_hamming_mmd(x,y,args)
                euclidean_mmd, sigma = rbf_mmd(x,y,args)
                exp_hamming_mmd_list.append(hamming_mmd)
                rbf_mmd_list.append(euclidean_mmd)
            hamming_mmd = sum(exp_hamming_mmd_list)/10
            euclidean_mmd = sum(rbf_mmd_list)/10

            #log
            log_entry['sampler_hamming_mmd'], log_entry['sampler_bandwidth'] = hamming_mmd.item(), bandwidth.item()
            log_entry['sampler_euclidean_mmd'], log_entry['sampler_sigma'] = euclidean_mmd.item(), sigma.item()

            #compute mmds 2/2
            exp_hamming_mmd_list = []
            rbf_mmd_list = []
            gibbs_sampler = GibbsSampler(2, args.discrete_dim, args.device)
            for _ in range(10):
                if args.source == 'data':
                    xt = torch.from_numpy(get_batch_data(db, args, batch_size = 4000)).to(args.device)
                    plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
                else:
                    xt = None
                x = torch.from_numpy(gen_samples(dfs_model, args, batch_size = 4000, xt=xt)).to('cpu')
                y = gibbs_sampler(ebm_model, num_rounds=100, num_samples=4000).to('cpu')
                hamming_mmd, bandwidth = exp_hamming_mmd(x,y,args)
                euclidean_mmd, sigma = rbf_mmd(x,y,args)
                exp_hamming_mmd_list.append(hamming_mmd)
                rbf_mmd_list.append(euclidean_mmd)
            hamming_mmd = sum(exp_hamming_mmd_list)/10
            euclidean_mmd = sum(rbf_mmd_list)/10

            #log
            log_entry['sampler_ebm_hamming_mmd'], log_entry['sampler_ebm_bandwidth'] = hamming_mmd.item(), bandwidth.item()
            log_entry['sampler_ebm_euclidean_mmd'], log_entry['sampler_ebm_sigma'] = euclidean_mmd.item(), sigma.item()

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)
            log(args, log_entry, epoch, timestamp)
