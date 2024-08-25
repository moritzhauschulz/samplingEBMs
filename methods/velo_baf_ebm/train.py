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
import random

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils.eval import plot_weight_histogram
import argparse
import utils.samplers as samplers
from utils.utils import get_batch_data
from utils.utils import plot as toy_plot
import utils.block_samplers as block_samplers
from utils.toy_data_lib import get_db
import torch.nn as nn
import utils.ais as ais
import utils.utils as utils

from utils.eval import exp_hamming_mmd
from utils.eval import rbf_mmd
from utils.eval import ebm_evaluation
from utils.eval import sampler_ebm_evaluation
from utils.eval import sampler_evaluation

import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion

from utils.model import ResNetFlow, EBM, MLPModel, MLPScore
from utils.utils import get_sampler
from utils.utils import get_x0
from utils.utils import align_batchsize

from utils.sampler import GibbsSampler 

from velo_dfm.train import gen_samples
from velo_dfm.train import gen_back_samples

from velo_dfm.train import compute_loss as compute_dfm_loss
from velo_dfm.train import gen_samples as dfm_gen_samples
from velo_dfm.train import gen_back_samples as dfm_gen_back_samples
from velo_edfm.train import compute_loss as compute_edfm_loss

from velo_dfs.train import compute_loss as compute_dfs_loss
from velo_dfs.train import gen_samples as dfs_gen_samples
from velo_dfs.train import gen_back_samples as dfs_gen_back_samples
from velo_edfs.train import compute_loss as compute_edfs_loss

from utils.eval import get_eval_timestamp

def make_sampler(args):
    if args.dfs:
        gen_samples = dfs_gen_samples
    else:
        gen_samples = dfm_gen_samples
    return gen_samples

def make_back_sampler(args):
    if args.dfs:
        gen_samples = dfs_gen_back_samples
    else:
        gen_samples = dfm_gen_back_samples
    return gen_back_samples

def make_eloss(args):
    if args.dfs:
        compute_eloss = compute_edfs_loss
    else:
        compute_eloss = compute_edfm_loss
    return compute_eloss

def make_loss(args):
    if args.dfs:
        compute_loss = compute_dfs_loss
    else:
        compute_loss = compute_dfm_loss
    return compute_loss

def main_loop(args, verbose=False):

    #set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.is_toy:
        main_loop_toy(args, verbose)
    else:
        main_loop_real(args, verbose)
    log_completion(args.methods, args.dataset_name, args)

def get_x_fake(B,D,S,epoch,t_target,sampler,back_sampler, x, dfs_model, ebm_model, args, plot_fn, save_inter = False):
    x_turn, logp_to_x, logp_from_x = back_sampler(dfs_model, args, x, t_target = t_target,return_logp=True)
    x_prime, logp_to_x_prime, logp_from_x_prime = sampler(dfs_model, args, t = t_target, xt = x_turn, return_logp=True)

    logp_x_to_x_prime = logp_from_x + logp_to_x_prime
    logp_x_prime_to_x = logp_from_x_prime + logp_to_x
    logp_delta = logp_x_prime_to_x - logp_x_to_x_prime

    logp_delta = logp_delta.detach()
    if args.with_mh:
        # MH step, calculate log p(x') - log p(x)
        logp_x = -ebm_model(x.float())
        logp_x_prime = -ebm_model(x_prime.float())
        lp_update = logp_x_prime - logp_x
        update_probs = lp_update.exp().clamp(max=1)
        updates = torch.bernoulli(update_probs)
        x_fake = x_prime * updates[:, None] + x * (1. - updates[:, None])
        update_success_rate = updates.mean().item()
    else:
        x_fake = x_prime
        update_success_rate = None

    #debugging
    if save_inter:          
        plot_fn(f'{args.sample_path}/x_turn_{epoch}.png', x_turn.float())
        plot_fn(f'{args.sample_path}/x_{epoch}.png', x.float())
        plot_fn(f'{args.sample_path}/x_prime_{epoch}.png', x_prime.float())
        plot_fn(f'{args.sample_path}/x_fake_{epoch}.png', x_fake.float())

    return x_fake.long(), update_success_rate, t_target

def main_loop_real(args, verbose=False):

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))
    source_train_loader = copy.deepcopy(train_loader)
    inner_source_train_loader = copy.deepcopy(train_loader)
    inner_target_train_loader = copy.deepcopy(train_loader)

    gen_samples = make_sampler(args)
    gen_back_samples = make_back_sampler(args)
    compute_loss = make_loss(args)
    compute_eloss = make_eloss(args)

    args.itr_per_epoch = len(train_loader)

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
    
    def get_independent_sample(loader, args=args):
        (x, _) = next(iter(loader))
        return preprocess(x)

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

    init_dist = torch.distributions.Bernoulli(probs=init_mean)

    # make dfs model
    dfs_model = ResNetFlow(64, args)

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

    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr)
    ebm_optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.ebm_lr, weight_decay=args.weight_decay)

    ema_ebm_model = copy.deepcopy(ebm_model)
    ema_dfs_model = copy.deepcopy(dfs_model)

    # move to cuda
    ebm_model.to(args.device)
    ema_ebm_model.to(args.device)
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)

    #get sampler
    sampler = get_sampler(args) #for eval

    best_val_ll = -np.inf
    dfs_lr = args.lr
    ebm_lr = args.ebm_lr

    test_ll_list = []

    temp = args.start_temp
   
    start_time = time.time()
    cum_eval_time = 0

    #warmup
    for i, (inner_x_source, _) in zip(range(args.dfs_warmup_iter), cycle(inner_source_train_loader)):
        (B, D) = inner_x_source.shape
        x1 = q_dist.sample((B,)).to(args.device).long()
        S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
        t = torch.rand((B,)).to(args.device)

        if args.source in ['data','omniglot']:
            x0 = preprocess(inner_x_source).long().to(args.device)
        else:
            x0 = get_x0(B,D,S,args)
        
        log_p_prob = -ebm_model(x1.float())
        log_q_prob = q_dist.log_prob(x1.float()).sum(dim=-1).to(args.device)

        dfs_loss, weights, temp = compute_eloss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
        
        dfs_optimizer.zero_grad()
        dfs_loss.backward()
        dfs_optimizer.step()

    # update ema_model
    for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
        ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

    print(f'Warmup completed with dfs loss of {dfs_loss.item()} after {args.dfs_warmup_iter} iterations.')
    #save dfs samples
    if args.source == 'data':
        xt = get_independent_sample(test_loader).long().to(args.device)
    else:
        xt = None
    samples = gen_samples(dfs_model, args, batch_size=100, xt=xt)
    plot(f'{args.sample_path}/post_warmup_dfs_samples_.png', torch.tensor(samples).float())

    for epoch in range(1, args.num_epochs + 1):
        
        dfs_model.train()
        ebm_model.eval()
        pbar = tqdm(train_loader) if verbose else train_loader

        dfs_times = []
        ebm_times = []

        for itr, ((x, _), (x_source, _)) in enumerate(zip(pbar, cycle(source_train_loader))):
            x, x_source = align_batchsize(x, x_source)

            (B, D) = x_source.shape
            S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size

            if itr < args.warmup_iters:
                ebm_lr = args.ebm_lr * float(itr) / args.warmup_iters
                for param_group in ebm_optimizer.param_groups:
                    param_group['lr'] = ebm_lr

            ebm_start_time = time.time()
            x = preprocess(x.to(args.device).requires_grad_())

            if args.rand_t:
                t_target = random.random()
            elif args.lin_t:
                t_target = 1 - min(1, (args.itr_per_epoch * (epoch - 1) + itr + 1)/ (args.itr_per_epoch * args.warmup_baf))
            elif args.t > 0:
                t_target = args.t
            else:
                raise ValueError
            save_inter = True if itr % 100 == 0 and epoch % args.epoch_save == 0 else False
            x_fake, success_rate, t_target = get_x_fake(B,D,S,epoch,t_target,gen_samples, gen_back_samples, x.long(), dfs_model, ebm_model, args,plot, save_inter)            

            logp_real = -ebm_model(x.float()).squeeze()
            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                              create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg = 0.0

            logp_fake = -ebm_model(x_fake.float()).squeeze()

            obj = logp_real.mean() - logp_fake.mean()
            ebm_loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

            ebm_optimizer.zero_grad()
            ebm_loss.backward()
            ebm_optimizer.step()

            ebm_end_time = time.time()
            ebm_times.append(ebm_end_time - ebm_start_time)

            dfs_start_time = time.time()
            for i, (inner_x_source, _) in zip(range(args.dfs_per_ebm), cycle(inner_source_train_loader)):
                x, inner_x_source = align_batchsize(x, inner_x_source)
                (B, D) = x.shape
                S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
                t = torch.rand((B,)).to(args.device)

                x = preprocess(x.to(args.device)).long()
                
                # restore_mh = args.with_mh
                # args.with_mh = True
                if args.rand_t:
                    t_target = random.random()
                elif args.lin_t:
                    t_target = 1 - min(1, (args.itr_per_epoch * (epoch - 1) + itr + 1)/ (args.itr_per_epoch * args.warmup_baf))
                elif args.t > 0:
                    t_target = args.t
                else:
                    raise ValueError
                x1, _, _ = get_x_fake(B,D,S,epoch, t_target, gen_samples, gen_back_samples, x, dfs_model, ebm_model, args,plot)
                # args.with_mh = restore_mh

                if args.source in ['data', 'omniglot']:
                    x0 = preprocess(inner_x_source).long().to(args.device)
                else:
                    x0 = get_x0(B,D,S,args)
                
                dfs_loss, acc = compute_loss(dfs_model, B, D, S, t, x1, x0, args) 
                
                dfs_optimizer.zero_grad()
                dfs_loss.backward()
                dfs_optimizer.step()

                # update ema_model
                for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                    ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            dfs_end_time = time.time()
            dfs_times.append(dfs_end_time - dfs_start_time)

            
            # update ema_model
            for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)


            if verbose:
                pbar.set_description(f'dfs loss: {dfs_loss.item()}; ebm loss {ebm_loss.item()}; success rate {success_rate}; t: {t_target}; avg dfs step time: {sum(dfs_times)/(itr+1)}; avg ebm step time: {sum(ebm_times)/(itr+1)}; acc: {acc};')

        if verbose:
            print(f'Epoch: {epoch}\{args.num_epochs}; Final DFS Loss: {dfs_loss.item()}; EBM Loss: {ebm_loss.item()}; mean logp_real: {logp_real.mean().item()}; mean logp_fake: {logp_fake.mean().item()}; t: {t} \n')


        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            #save models
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')
            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{epoch}.pt')

            #save dfs samples
            if args.source == 'data':
                xt = get_independent_sample(test_loader).long().to(args.device) 
                plot(f'{args.sample_path}/source_{epoch}.png', xt.float())
            else:
                xt = None
            samples = gen_samples(dfs_model, args, batch_size=100, xt=xt) 
            plot(f'{args.sample_path}/dfs_samples_{epoch}.png', torch.tensor(samples).float())
            samples = gen_samples(ema_dfs_model, args, batch_size=100, xt=xt)
            plot(f'{args.sample_path}/ema_dfs_samples_{epoch}.png', torch.tensor(samples).float())

            #save ebm samples
            EBM_samples = init_dist.sample((100,))
            model = lambda x: -ebm_model(x) #because evm returns is f(x) and logp is -f(x)
            MCMC_pbar = tqdm(range(args.save_sampling_steps))
            for d in MCMC_pbar:
                EBM_samples = sampler.step(EBM_samples.detach(), model).detach()
                MCMC_pbar.set_description('MCMC Sampling in Progress...')
            EBM_samples = EBM_samples.cpu().detach()
            plot(f'{args.sample_path}/EBM_samples_{epoch}_steps_{args.save_sampling_steps}.png', EBM_samples)

            #save log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['dfs_loss'] = dfs_loss.item()
            log_entry['ebm_loss'] = ebm_loss.item()
            log_entry['success rate'] = success_rate

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, epoch, timestamp)

        if args.eval_on:
            print('Starting evaluation, this may take a while.')
            if (epoch % args.eval_every == 0) or (epoch == args.num_epochs):
                eval_start_time = time.time()

                log_entry = {'epoch':None,'timestamp':None}

                logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ema_ebm_model, init_dist, sampler,
                                                                            train_loader, val_loader, test_loader,
                                                                            preprocess, args.device,
                                                                            args.eval_sampling_steps,
                                                                            args.test_batch_size)
                log_entry['dfs_loss'] = dfs_loss.item()
                log_entry['ebm_loss'] = ebm_loss.item()
                log_entry['success rate'] = success_rate
                log_entry['EMA Train log-likelihood'] = train_ll.item()
                log_entry['EMA Train log-likelihood'] = val_ll.item()
                log_entry['EMA Test log-likelihood'] = test_ll.item()
                
                for _i, _x in enumerate(ais_samples):
                    plot(f'{args.sample_path}/EBM_sample_{args.dataset_name}_{args.sampler}_{args.step_size}_{epoch}_{_i}.png', _x)

                if val_ll.item() > 0:
                    print(f"LL was greater zero at {val_ll.item()}")
                    exit()
                if val_ll.item() > best_val_ll:
                    best_val_ll = val_ll.item()
                    torch.save(ema_ebm_model.state_dict(), f"{args.ckpt_path}/best_ema_ebm_ckpt_{args.dataset_name}_{args.sampler}_{args.step_size}_{epoch}.pt")

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, epoch, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')


def main_loop_toy(args, verbose=False):

    # load data
    db = get_db(args)
    plot = lambda p, x: toy_plot(p, x, args)

    gen_samples = make_sampler(args)
    gen_back_samples = make_back_sampler(args)
    compute_loss = make_loss(args)
    compute_eloss = make_eloss(args)

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
    
    def get_independent_sample(loader, args=args):
        (x, _) = next(iter(loader))
        return preprocess(x)

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

    init_dist = torch.distributions.Bernoulli(probs=init_mean)

    # make dfs model
    dfs_model = MLPModel(args).to(args.device)

    # make ebm model
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    ebm_model = EBM(net, init_mean).to(args.device)
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/initial_heat.png', args)


    ebm_model = EBM(net, init_mean).to(args.device)

    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr)
    ebm_optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.ebm_lr, weight_decay=args.weight_decay)

    ema_ebm_model = copy.deepcopy(ebm_model)
    ema_dfs_model = copy.deepcopy(dfs_model)

    #gibbs sampler for producing samples from EBM
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)

    # move to cuda
    ebm_model.to(args.device)
    ema_ebm_model.to(args.device)
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)

    best_val_ll = np.inf
    dfs_lr = args.lr
    ebm_lr = args.ebm_lr

    test_ll_list = []

    temp = args.start_temp
   
    start_time = time.time()
    cum_eval_time = 0

    #warmup
    for i in range(args.dfs_warmup_iter):
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

        dfs_loss, weights, temp = compute_eloss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
        
        dfs_optimizer.zero_grad()
        dfs_loss.backward()
        dfs_optimizer.step()

        # update ema_model
        for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
        
    print(f'Warmup completed with dfs loss of {dfs_loss.item()} after {args.dfs_warmup_iter} iterations.')
    #save dfs samples
    if args.source == 'data':
        xt = torch.from_numpy(get_batch_data(db, args, batch_size = 2500)).to(args.device)
    else:
        xt = None
    samples = gen_samples(dfs_model, args, batch_size = 2500, xt=xt)
    plot(f'{args.sample_path}/post_warmup_dfs_samples_.png', torch.tensor(samples).float())


    pbar = tqdm(range(1, args.num_epochs + 1)) if verbose else range(1,args.num_epochs + 1)
    for epoch in pbar:
        dfs_model.train()
        ebm_model.eval()

        dfs_times = []
        ebm_times = []

        if epoch < args.warmup_iters:
            ebm_lr = args.ebm_lr * float(epoch) / args.warmup_iters
            for param_group in ebm_optimizer.param_groups:
                param_group['lr'] = ebm_lr

        ebm_start_time = time.time()
        x = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
        (B, D) = x.shape
        S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size

        if args.rand_t:
            t_target = random.random()
        elif args.lin_t:
            t_target = 1 - min(1, epoch/args.warmup_baf)
        elif args.t > 0:
            t_target = args.t
        else:
            raise ValueError
        save_inter = True if epoch % args.epoch_save == 0 else False
        x_fake, success_rate, t_target = get_x_fake(B,D,S,epoch,t_target,gen_samples, gen_back_samples, x.long(), dfs_model, ebm_model, args, plot, save_inter)            

        logp_real = -ebm_model(x.float()).squeeze()
        if args.p_control > 0:
            grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                            create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
            grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
        else:
            grad_reg = 0.0

        logp_fake = -ebm_model(x_fake.float()).squeeze()

        obj = logp_real.mean() - logp_fake.mean()
        ebm_loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

        ebm_optimizer.zero_grad()
        ebm_loss.backward()
        ebm_optimizer.step()

        ebm_end_time = time.time()
        ebm_times.append(ebm_end_time - ebm_start_time)

        dfs_start_time = time.time()
        for i in range(args.dfs_per_ebm):
            t = torch.rand((B,)).to(args.device)

            x = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
            
            # restore_mh = args.with_mh
            # args.with_mh = True
            if args.rand_t:
                t_target = random.random()
            elif args.lin_t:
                t_target = 1 - min(1, epoch/args.warmup_baf)
            elif args.t > 0:
                t_target = args.t
            else:
                raise ValueError

            x1, _, _ = get_x_fake(B,D,S,epoch, t_target, gen_samples, gen_back_samples, x, dfs_model, ebm_model, args, plot)

            if args.source in ['data']:
                x0 = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
            else:
                x0 = get_x0(B,D,S,args)
            
            dfs_loss, acc = compute_loss(dfs_model, B, D, S, t, x1, x0, args) 
            
            dfs_optimizer.zero_grad()
            dfs_loss.backward()
            dfs_optimizer.step()

            # update ema_model
            for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

        dfs_end_time = time.time()
        dfs_times.append(dfs_end_time - dfs_start_time)

        # update ema_model
        for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

        if verbose:
            pbar.set_description(f'dfs loss: {dfs_loss.item()}; ebm loss {ebm_loss.item()}; success rate {success_rate}; t: {t_target}; avg dfs step time: {sum(dfs_times)/(epoch)}; avg ebm step time: {sum(ebm_times)/(epoch)}; acc: {acc}; \n mean logp_real: {logp_real.mean().item()}; mean logp_fake: {logp_fake.mean().item()}; t: {t} \n')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            #save models
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')
            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{epoch}.pt')

            #save dfs samples
            if args.source == 'data':
                xt = torch.from_numpy(get_batch_data(db, args, batch_size = 2500)).to(args.device)
                plot(f'{args.sample_path}/source_{epoch}.png', xt)
            else:
                xt = None
            samples = gen_samples(dfs_model, args, batch_size=100, xt=xt) 
            plot(f'{args.sample_path}/dfs_samples_{epoch}.png', torch.tensor(samples).float())
            samples = gen_samples(ema_dfs_model, args, batch_size=100, xt=xt)
            plot(f'{args.sample_path}/ema_dfs_samples_{epoch}.png', torch.tensor(samples).float())

            #save ebm samples
            EBM_samples = gibbs_sampler(ebm_model, num_rounds=100, num_samples=2500).to('cpu').detach()
            plot(f'{args.sample_path}/EBM_samples_{epoch}_steps_{args.save_sampling_steps}.png', EBM_samples)

            #save log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['dfs_loss'] = dfs_loss.item()
            log_entry['ebm_loss'] = ebm_loss.item()
            log_entry['success rate'] = success_rate

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, epoch, timestamp)

        if args.eval_on:
            print('Starting evaluation, this may take a while.')
            if (epoch % args.eval_every == 0) or (epoch == args.num_epochs):
                eval_start_time = time.time()

                #save models
                torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')
                torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{epoch}.pt')

            
                #log
                log_entry = {'epoch':None,'timestamp':None}
                log_entry['dfs_loss'] = dfs_loss.item()
                log_entry['ebm_loss'] = ebm_loss.item()
                log_entry['success_rate'] = success_rate

                #compute mmds 1/2
                hamming_mmd, bandwidth, euclidean_mmd, sigma = sampler_evaluation(args, db, dfs_model, gen_samples)

                #log
                log_entry['sampler_hamming_mmd'], log_entry['sampler_bandwidth'] = hamming_mmd, bandwidth
                log_entry['sampler_euclidean_mmd'], log_entry['sampler_sigma'] = euclidean_mmd, sigma

                #compute mmds 2/2
                hamming_mmd, bandwidth, euclidean_mmd, sigma = sampler_ebm_evaluation(args, db, dfs_model, gen_samples, ebm_model)

                #log
                log_entry['sampler_ebm_hamming_mmd'], log_entry['sampler_ebm_bandwidth'] = hamming_mmd, bandwidth
                log_entry['sampler_ebm_euclidean_mmd'], log_entry['sampler_ebm_sigma'] = euclidean_mmd, sigma

                if epoch < args.num_epochs:
                    ais_samples = args.intermediate_ais_samples
                    ais_num_steps = args.intermediate_ais_num_steps
                else: 
                    ais_samples =  args.final_ais_samples
                    ais_num_steps = args.final_ais_num_steps

                eval_start_time = time.time()

                #compute nnl, ebm for ebm and log – takes much space
                log_entry['ebm_nll'], log_entry['ebm_hamming_mmd'], log_entry['ebm_bandwidth'], log_entry['ebm_euclidean_mmd'], log_entry['ebm_sigma'] = ebm_evaluation(args, db, ebm_model, batch_size=4000, ais_samples=ais_samples, ais_num_steps=ais_num_steps)
                
                if log_entry['ebm_nll'] < 0:
                    print(f"NLL was below zero at {log_entry['ebm_nll']}")
                    exit()
                if log_entry['ebm_nll'] < best_val_ll:
                    best_val_ll = log_entry['ebm_nll']
                    torch.save(ebm_model.state_dict(), f"{args.ckpt_path}/best_ebm_ckpt_{args.dataset_name}_{epoch}.pt")

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, epoch, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')
