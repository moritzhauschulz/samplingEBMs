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

import utils.samplers as samplers
import utils.utils as utils
from utils.utils import get_batch_data
from utils.utils import plot as toy_plot
from utils.utils import get_x0
from utils.eval import plot_weight_histogram
from utils.utils import get_optimal_temp
from utils.sampler import GibbsSampler

import utils.vamp_utils as vamp_utils

from utils.eval import log
from utils.eval import log_completion
from utils.eval import get_eval_timestamp
from utils.eval import exp_hamming_mmd
from utils.eval import rbf_mmd
from utils.eval import sampler_ebm_evaluation
from utils.eval import sampler_evaluation
from utils.toy_data_lib import get_db

from utils.eval import ebm_evaluation
import utils.ais as ais

from utils.model import ResNetFlow, EBM, MLPModel, MLPScore
from utils.utils import get_sampler
from utils.utils import align_batchsize

from velo_dfs.train import gen_samples as dfs_gen_samples
from velo_edfs.train import compute_loss as compute_edfs_loss
from velo_dfs.train import compute_loss as compute_dfs_loss


from velo_dfm.train import gen_samples as dfm_gen_samples
from velo_edfm.train import compute_loss as compute_edfm_loss
from velo_dfm.train import compute_loss as compute_dfm_loss

def make_sampler(args):
    if args.dfs:
        gen_samples = dfs_gen_samples
    else:
        gen_samples = dfm_gen_samples
    return gen_samples

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


def main_loop_real(args, verbose=False):

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                    p, normalize=True, nrow=int(x.size(0) ** .5))
    inner_source_train_loader = copy.deepcopy(train_loader)
    inner_train_loader = copy.deepcopy(train_loader)

    gen_samples = make_sampler(args)
    compute_loss = make_loss(args)
    compute_eloss = make_eloss(args)

    source_train_loader1 = copy.deepcopy(train_loader)
    source_train_loader2 = copy.deepcopy(train_loader)

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

    #set temperature
    temp = args.start_temp


    start_time = time.time()
    cum_eval_time = 0
    temp = args.start_temp


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

    dfs_times = []
    dfs_optional_times = []
    ebm_times = []
    itr = 1
    while itr <= args.num_itr:
        dfs_model.train()
        ebm_model.train()
        pbar = tqdm(train_loader) if verbose else train_loader

        if itr < args.warmup_iters:
            ebm_lr = args.ebm_lr * float(itr) / args.warmup_iters
            for param_group in ebm_optimizer.param_groups:
                param_group['lr'] = ebm_lr

        for _, ((x, _), (x_source1, _), (x_source2, _)) in enumerate(zip(pbar, cycle(source_train_loader1), cycle(source_train_loader2))):

            dfs_start_time = time.time()
            x, x_source1 = align_batchsize(x, x_source1)
            _, x_source2 = align_batchsize(x, x_source2)

            x = preprocess(x.to(args.device).requires_grad_())
            if args.source == 'data':
                xt = preprocess(x_source1).long().to(args.device)
            else:
                xt = get_x0(B,D,S,args)

            if itr == 0 or not args.recycle_dfs_sample: 
                x_fake = torch.tensor(gen_samples(dfs_model, args, batch_size=x.shape[0], xt=xt)).to(args.device).detach()
            
            for i, ((inner_x, _), (inner_x_source, _)) in zip(range(args.dfs_per_ebm), zip(inner_train_loader, cycle(inner_source_train_loader))):

                inner_x, inner_x_source = align_batchsize(inner_x, inner_x_source)
                
                if i == 0:
                    x1 = x.long().to(args.device) #learn on the datapoint that the ebm will learn on next...
                    x1, _ = align_batchsize(inner_x, inner_x_source)
                else:
                    x1 = preprocess(inner_x).long().to(args.device) 

                (B, D) = x1.shape
                S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
                t = torch.rand((B,)).to(args.device)

                if args.source == 'data':
                    x0 = preprocess(inner_x_source).long().to(args.device)
                else:
                    x0 = get_x0(B,D,S,args).to(args.device)

                dfs_loss, acc = compute_loss(dfs_model,B,D,S,t,x1.long().to(args.device),x0.long().to(args.device),args)

                dfs_optimizer.zero_grad()
                dfs_loss.backward()
                dfs_optimizer.step()

            # update ema_model
            for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            #make x_dfs 
            if args.source == 'data':
                xt = preprocess(x_source2).long().to(args.device)
            else:
                xt = get_x0(B,D,S,args)
            x_dfs = torch.from_numpy(gen_samples(dfs_model, args, xt=xt)).to(args.device)
            dfs_pause_time = time.time()

            #ebm-data loss
            ebm_start_time = time.time()
            logp_real = -ebm_model(x).squeeze()
            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                                create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg_real = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg_real = 0.0
            logp_fake = -ebm_model(x_fake.float()).squeeze()
            ebm_data_loss = -(logp_real.mean() - logp_fake.mean()) + grad_reg_real + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

            #ebm-dfs loss
            logp_dfs = -ebm_model(x_dfs.float()).squeeze()
            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_dfs.sum(), x,
                                                create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg_dfs = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg_dfs = 0.0
            ebm_dfs_loss = -(logp_dfs.mean() - logp_fake.mean()) + grad_reg_dfs + args.l2 * ((logp_dfs ** 2.).mean() + (logp_fake ** 2.).mean())

            ebm_loss = args.alpha * ebm_data_loss + (1 - args.alpha) * ebm_dfs_loss

            ebm_optimizer.zero_grad()
            ebm_loss.backward()
            ebm_optimizer.step()

            for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
                ebm_end_time = time.time()

            ebm_end_time = time.time()

            #optional step towards ebm
            dfs_restart_time = time.time()
            if args.optional_step:
                logp_dfs_old = logp_dfs
                logp_dfs = -ebm_model(x_dfs.float()).squeeze()

                dfs_loss, weights, temp = compute_eloss(dfs_model, B, D, S, logp_dfs, logp_dfs_old, t, x_dfs, x0, args, temp)

                dfs_optimizer.zero_grad()
                dfs_loss.backward()
                dfs_optimizer.step()
            dfs_end_time = time.time()

            dfs_times.append(dfs_pause_time - dfs_start_time)
            dfs_optional_times.append(dfs_end_time - dfs_restart_time)
            ebm_times.append(ebm_end_time - ebm_start_time)
            

            if verbose:
                pbar.set_description(f'Itr: {itr}\{args.num_itr}; EBM loss: {ebm_loss.item()} ({ebm_data_loss.item(), ebm_dfs_loss.item()}), DFS loss: {dfs_loss.item()} avg dfs time: {sum(dfs_times)/(itr)} avg dfs optional time: {sum(dfs_optional_times)/(itr)} \n avg ebm step time: {sum(ebm_times)/(itr)}, mean logp_real was {logp_real.mean()}, mean logp_fake was {logp_fake.mean()}')
            

            if (itr % args.itr_save == 0) or (itr == args.num_itr):
                eval_start_time = time.time()

                #save models
                torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{itr}.pt')
                torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{itr}.pt')

                #save dfs samples
                if args.source == 'data':
                    xt = get_independent_sample(test_loader).long().to(args.device) 
                    plot(f'{args.sample_path}/source_{itr}.png', xt.float())
                else:
                    xt = None
                samples = gen_samples(dfs_model, args, batch_size=100, xt=xt)
                plot(f'{args.sample_path}/dfs_samples_{itr}.png', torch.tensor(samples).float())
                samples = gen_samples(ema_dfs_model, args, batch_size=100, xt=xt)
                plot(f'{args.sample_path}/ema_dfs_samples_{itr}.png', torch.tensor(samples).float())
                if args.optional_step:
                    weights_dir = f'{args.plot_path}/weights_histogram_{itr}.png'
                    if not os.path.exists(weights_dir):
                        plot_weight_histogram(weights, output_dir=weights_dir)

                #save ebm samples
                EBM_samples = init_dist.sample((100,))
                model = lambda x: -ebm_model(x) #because evm returns is f(x) and logp is -f(x)
                MCMC_pbar = tqdm(range(args.save_sampling_steps))
                for d in MCMC_pbar:
                    EBM_samples = sampler.step(EBM_samples.detach(), model).detach()
                    MCMC_pbar.set_description('MCMC Sampling in Progress...')
                EBM_samples = EBM_samples.cpu().detach()
                plot(f'{args.sample_path}/EBM_samples_{itr}_steps_{args.save_sampling_steps}.png', EBM_samples)

                #save log
                log_entry = {'itr':None,'timestamp':None}
                log_entry['dfs_loss'] = dfs_loss.item()
                log_entry['ebm_loss'] = ebm_loss.item()
                if args.optional_step:
                    log_entry['temp'] = temp
                    log_entry['mean_weight'] = weights.mean().item()

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp)

            if args.eval_on:
                print('Starting evaluation, this may take a while.')
                if (itr % args.eval_every == 0) or (itr == args.num_itr):
                    eval_start_time = time.time()

                    log_entry = {'itr':None,'timestamp':None}

                    logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ema_ebm_model, init_dist, sampler,
                                                                                train_loader, val_loader, test_loader,
                                                                                preprocess, args.device,
                                                                                args.eval_sampling_steps,
                                                                                args.test_batch_size)
                    log_entry['dfs_loss'] = dfs_loss.item()
                    log_entry['ebm_loss'] = ebm_loss.item()
                    log_entry['temp'] = temp
                    log_entry['mean_weight'] = weights.mean().item()
                    log_entry['EMA Train log-likelihood'] = train_ll.item()
                    log_entry['EMA Train log-likelihood'] = val_ll.item()
                    log_entry['EMA Test log-likelihood'] = test_ll.item()
                    
                    for _i, _x in enumerate(ais_samples):
                        plot(f'{args.sample_path}/EBM_sample_{args.dataset_name}_{args.sampler}_{args.step_size}_{itr}_{_i}.png', _x)

                    if val_ll.item() > 0:
                        print(f"LL was greater zero at {val_ll.item()}")
                        exit()
                    if val_ll.item() > best_val_ll:
                        best_val_ll = val_ll.item()
                        torch.save(ema_ebm_model.state_dict(), f"{args.ckpt_path}/best_ema_ebm_ckpt_{args.dataset_name}_{args.sampler}_{args.step_size}_{itr}.pt")

                    timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                    log(args, log_entry, itr, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')
            itr += 1

def main_loop_toy(args, verbose=False):
    assert not args.enable_backward, 'Backwards sampling not implemented for toy data.'


    gen_samples = make_sampler(args)
    compute_loss = make_loss(args)
    compute_eloss = make_eloss(args)


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

    #set temperature
    temp = args.start_temp

    best_val_ll = np.inf
    dfs_lr = args.lr
    ebm_lr = args.ebm_lr
   
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


    pbar = tqdm(range(1, args.num_itr + 1)) if verbose else range(1,args.num_itr + 1)
    for itr in pbar:
        dfs_model.train()
        ebm_model.train()

        dfs_times = []
        dfs_optional_times = []
        ebm_times = []

        if itr < args.warmup_iters:
            ebm_lr = args.ebm_lr * float(itr) / args.warmup_iters
            for param_group in ebm_optimizer.param_groups:
                param_group['lr'] = ebm_lr

        dfs_start_time = time.time()
        
        x = torch.from_numpy(get_batch_data(db, args)).to(args.device)  

        if args.source == 'data':
            xt = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
        else:
            xt = get_x0(B,D,S,args)

        # if itr == 0 or not args.recycle_dfs_sample: 
        x_fake = torch.tensor(gen_samples(dfs_model, args, xt=xt)).to(args.device).detach()
        
        for i in range(args.dfs_per_ebm):
            
            if i == 0:
                x1 = x.long() #learn on the datapoint that the ebm will learn on next...
            else:
                x1 =  torch.from_numpy(get_batch_data(db, args)).to(args.device)  

            (B, D) = x1.shape
            S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
            t = torch.rand((B,)).to(args.device)

            if args.source == 'data':
                x0 =  torch.from_numpy(get_batch_data(db, args)).to(args.device)  
            else:
                x0 = get_x0(B,D,S,args)

            dfs_loss, acc = compute_loss(dfs_model,B,D,S,t,x1,x0,args)

            dfs_optimizer.zero_grad()
            dfs_loss.backward()
            dfs_optimizer.step()

        # update ema_model
        for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
        
        if args.source == 'data':
            xt = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
        else:
            xt = get_x0(B,D,S,args)

        x_dfs = torch.from_numpy(gen_samples(dfs_model, args, xt=xt)).to(args.device)
        dfs_pause_time = time.time()

        #ebm-data loss
        ebm_start_time = time.time()
        logp_real = -ebm_model(x.float()).squeeze()
        if args.p_control > 0:
            grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                            create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
            grad_reg_real = (grad_ld ** 2. / 2.).mean() * args.p_control
        else:
            grad_reg_real = 0.0
        logp_fake = -ebm_model(x_fake.float()).squeeze()
        ebm_data_loss = -(logp_real.mean() - logp_fake.mean()) + grad_reg_real + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

        #ebm-dfs loss
        logp_dfs = -ebm_model(x_dfs.float()).squeeze()
        if args.p_control > 0:
            grad_ld = torch.autograd.grad(logp_dfs.sum(), x,
                                            create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
            grad_reg_dfs = (grad_ld ** 2. / 2.).mean() * args.p_control
        else:
            grad_reg_dfs = 0.0
        ebm_dfs_loss = -(logp_dfs.mean() - logp_fake.mean()) + grad_reg_dfs + args.l2 * ((logp_dfs ** 2.).mean() + (logp_fake ** 2.).mean())

        ebm_loss = args.alpha * ebm_data_loss + (1 - args.alpha) * ebm_dfs_loss

        ebm_optimizer.zero_grad()
        ebm_loss.backward()
        ebm_optimizer.step()

        for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
            ebm_end_time = time.time()

        ebm_end_time = time.time()

        #without the optional step for now
        dfs_restart_time = time.time()
        if args.optional_step:
            logp_dfs_old = logp_dfs
            logp_dfs = -ebm_model(x_dfs.float()).squeeze()

            dfs_loss, weights, temp = compute_eloss(dfs_model, B, D, S, logp_dfs, logp_dfs_old, t, x_dfs, x0, args, temp)

            dfs_optimizer.zero_grad()
            dfs_loss.backward()
            dfs_optimizer.step()
        dfs_end_time = time.time()

        dfs_times.append(dfs_pause_time - dfs_start_time)
        dfs_optional_times.append(dfs_end_time - dfs_restart_time)
        ebm_times.append(ebm_end_time - ebm_start_time)
        
        if verbose:
            pbar.set_description(f'EBM loss: {ebm_loss.item()} ({ebm_data_loss.item(), ebm_dfs_loss.item()}), DFS loss: {dfs_loss.item()} avg dfs time: {sum(dfs_times)/(itr)} avg dfs optional time: {sum(dfs_optional_times)/(itr)} \n avg ebm step time: {sum(ebm_times)/(itr)}, mean logp_real was {logp_real.mean()}, mean logp_fake was {logp_fake.mean()}')
            

        if (itr % args.itr_save == 0) or (itr == args.num_itr):
            eval_start_time = time.time()

            #save models
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{itr}.pt')
            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{itr}.pt')

            #save dfs samples
            if args.source == 'data':
                xt = torch.from_numpy(get_batch_data(db, args, batch_size = 2500)).to(args.device)
                plot(f'{args.sample_path}/source_{itr}.png', xt)
            else:
                xt = None
            samples = gen_samples(dfs_model, args, batch_size = 2500, xt=xt)
            plot(f'{args.sample_path}/dfs_samples_{itr}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_dfs_model, args, batch_size = 2500, xt=xt)
            plot(f'{args.sample_path}/ema_dfs_samples_{itr}.png', torch.tensor(ema_samples).float())
            if args.optional_step:
                weights_dir = f'{args.plot_path}/weights_histogram_{itr}.png'
                if not os.path.exists(weights_dir):
                    plot_weight_histogram(weights, output_dir=weights_dir)

            #save ebm samples
            EBM_samples = gibbs_sampler(ebm_model, num_rounds=100, num_samples=2500).to('cpu').detach()
            plot(f'{args.sample_path}/EBM_samples_{itr}_steps_{args.save_sampling_steps}.png', EBM_samples)
            utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/{itr}_heat.png', args)

            #save log
            log_entry = {'itr':None,'timestamp':None}
            log_entry['dfs_loss'] = dfs_loss.item()
            log_entry['ebm_loss'] = ebm_loss.item()
            if args.optional_step:
                log_entry['temp'] = temp
                log_entry['mean_weight'] = weights.mean().item()

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, itr, timestamp)

        if args.eval_on:
            print('Starting evaluation, this may take a while.')
            if (itr % args.eval_every == 0) or (itr == args.num_itr):
                eval_start_time = time.time()

                #save models
                torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{itr}.pt')
                torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{itr}.pt')

            
                #log
                log_entry = {'itr':None,'timestamp':None}
                log_entry['dfs_loss'] = dfs_loss.item()
                log_entry['ebm_loss'] = ebm_loss.item()
                if args.optional_step:
                    log_entry['temp'] = temp
                    log_entry['mean_weight'] = weights.mean().item()
                    
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

                if itr < args.num_itr:
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
                    torch.save(ebm_model.state_dict(), f"{args.ckpt_path}/best_ebm_ckpt_{args.dataset_name}_{itr}.pt")

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')
