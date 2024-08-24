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

from velo_dfm.train import gen_samples as dfm_gen_samples
from velo_edfm.train import compute_loss as compute_edfm_loss




def main_loop(args, verbose=False):

    #set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.is_toy:
        main_loop_toy(args, verbose)
    else:
        main_loop_real(args, verbose)
    log_completion(args.methods, args.dataset_name, args)

def make_sampler(args):
    if args.dfs:
        gen_samples = dfs_gen_samples
    else:
        gen_samples = dfm_gen_samples
    return gen_samples

def make_loss(args):
    if args.dfs:
        compute_loss = compute_edfs_loss
    else:
        compute_loss = compute_edfm_loss
    return compute_loss

def main_loop_real(args, verbose=False):
    assert args.source in ['mask','uniform','data'], 'Omniglot not supported in EDFM/EDFS (why?).'

    gen_samples = make_sampler(args)
    compute_loss = make_loss(args)


    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))
    source_train_loader = copy.deepcopy(train_loader)
    inner_source_train_loader = copy.deepcopy(train_loader)


    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
    
    def get_independent_sample(loader):
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

    #set temperature
    temp = args.start_temp

    best_val_ll = -np.inf
    dfs_lr = args.lr
    ebm_lr = args.ebm_lr

    temp = args.start_temp
   
    start_time = time.time()
    cum_eval_time = 0

    #warmup
    for i, (inner_x_source, _) in zip(range(args.dfs_warmup_iter), cycle(inner_source_train_loader)):
        (B, D) = inner_x_source.shape
        x1 = q_dist.sample((B,)).to(args.device).long()
        S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
        t = torch.rand((B,)).to(args.device)

        if args.source in ['data']:
            x0 = preprocess(inner_x_source).long().to(args.device)
        else:
            x0 = get_x0(B,D,S,args)
        
        log_p_prob = -ebm_model(x1.float())
        log_q_prob = q_dist.log_prob(x1.float()).sum(dim=-1).to(args.device)

        dfs_loss, weights, temp = compute_edfm_loss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
        
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
        
        pbar = tqdm(train_loader) if verbose else train_loader

        dfs_gen_times = []
        dfs_step_times = []
        ebm_times = []

        if epoch < args.warmup_iters:
            ebm_lr = args.ebm_lr * float(epoch) / args.warmup_iters
            for param_group in ebm_optimizer.param_groups:
                param_group['lr'] = ebm_lr

        for itr, ((x, _), (x_source, _)) in enumerate(zip(pbar, cycle(source_train_loader))):

            dfs_model.train()
            ebm_model.eval()

            x, x_source = align_batchsize(x, x_source)

            dfs_start_time = time.time()
            dfs_samples_list = [] #can parallelize (i.e. tensorize)?
            q_probs_list = []

            for i, (inner_x_source, _) in zip(range(args.dfs_per_ebm), cycle(inner_source_train_loader)):
                (B, D) = inner_x_source.shape
                if args.source == 'data':
                    xt = preprocess(inner_x_source).long().to(args.device)
                else:
                    xt = get_x0(B,D,S,args)
                dfs_samples = torch.tensor(gen_samples(dfs_model, args, xt=xt)).to(args.device).detach()
                for j in range(args.MCMC_refinement):
                    dfs_samples = sampler.step(dfs_samples.float(), ebm_model).detach()
                dfs_samples_list.append(dfs_samples.long())
                q_probs_list.append(-ebm_model(dfs_samples.float()).squeeze())
        
            if args.source == 'data':
                xt = preprocess(x_source).long().to(args.device)
            else:
                xt = get_x0(B,D,S,args)
            x_fake = torch.tensor(gen_samples(dfs_model, args, xt=xt)).to(args.device).detach().float()
            for j in range(args.MCMC_refinement):
                x_fake = sampler.step(x_fake, ebm_model).detach().float()
            dfs_pause_time = time.time()

            #step EBM – get E_k+1
            ebm_start_time = time.time()
            x = preprocess(x.to(args.device).requires_grad_())
            #x_fake = dfs_samples_list[-1].clone().float() #recycling the dfs samples (only last one, but could use any)
            logp_real = -ebm_model(x).squeeze()
            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                              create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg = 0.0
            logp_fake = -ebm_model(x_fake).squeeze()
            # logp_fake = q_probs_list[-1] #can't be detached!
            obj = logp_real.mean() - logp_fake.mean()
            ebm_loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean()) #note sign of objective

            ebm_optimizer.zero_grad()
            ebm_loss.backward()
            ebm_optimizer.step()


            for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
            ebm_end_time = time.time()

            dfs_restart_time = time.time()

            for (x1, log_q_prob), (inner_x_source, _) in zip(zip(reversed(dfs_samples_list), reversed(q_probs_list)), cycle(inner_source_train_loader)):
                B = x1.shape[0]
                if args.source == 'data':
                    x0 = preprocess(inner_x_source).long().to(args.device)
                else:
                    x0 = get_x0(B,D,S,args)

                t = torch.rand((B,)).to(args.device)
                xt = x1.clone()
                mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
                xt[mask] = x0[mask]

                log_p_prob = -ebm_model(x1.float()).detach()
                
                dfs_loss, weights, temp = compute_loss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
                dfs_optimizer.zero_grad()
                dfs_loss.backward()
                dfs_optimizer.step()

                # update ema_model                 
                for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                    ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            dfs_end_time = time.time()
            dfs_end_time = time.time()
            dfs_gen_times.append(dfs_pause_time - dfs_start_time)
            dfs_step_times.append(dfs_end_time - dfs_restart_time)
            ebm_times.append(ebm_end_time - ebm_start_time)

            if verbose:
                pbar.set_description(f'EBM loss: {ebm_loss}, DFS loss: {dfs_loss} avg dfs gen time: {sum(dfs_gen_times)/(itr+1)} avg dfs step time: {sum(dfs_step_times)/(itr+1)} avg ebm step time: {sum(ebm_times)/(itr+1)}, mean logp_real was {logp_real.mean()}, mean logp_fake was {logp_fake.mean()}')

        if verbose:
            print(f'Epoch: {epoch}\{args.num_epochs}; Final DFS Loss: {dfs_loss}; EBM Loss: {ebm_loss}; mean logp_real: {logp_real.mean().item()}; mean logp_fake: {logp_fake.mean().item()}; Temp: {temp} \n')


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
            weights_dir = f'{args.plot_path}/weights_histogram_{epoch}.png'
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
            plot(f'{args.sample_path}/EBM_samples_{epoch}_steps_{args.save_sampling_steps}.png', EBM_samples)

            #save log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['dfs_loss'] = dfs_loss.item()
            log_entry['ebm_loss'] = ebm_loss.item()
            log_entry['temp'] = temp
            log_entry['mean_weight'] = weights.mean().item()

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
                log_entry['temp'] = temp
                log_entry['mean_weight'] = weights.mean().item()
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
    assert args.source in ['mask','uniform','data'], 'Omniglot not supported in EDFM/EDFS (why?).'

    gen_samples = make_sampler(args)
    compute_loss = make_loss(args)

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

        dfs_loss, weights, temp = compute_loss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
        
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
        
        dfs_gen_times = []
        dfs_step_times = []
        ebm_times = []

        dfs_model.train()
        ebm_model.eval()

        if epoch < args.warmup_iters:
            ebm_lr = args.ebm_lr * float(epoch) / args.warmup_iters
            for param_group in ebm_optimizer.param_groups:
                param_group['lr'] = ebm_lr

        dfs_start_time = time.time()
        dfs_samples_list = [] #can parallelize (i.e. tensorize)?
        q_probs_list = []

        for i in range(args.dfs_per_ebm):
            (B, D) = args.batch_size, args.discrete_dim
            S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
            t = torch.rand((B,)).to(args.device)

            if args.source == 'data':
                xt = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
            else:
                xt = get_x0(B,D,S,args)

            dfs_samples = torch.tensor(gen_samples(dfs_model, args, xt=xt)).to(args.device).detach()
            for j in range(args.MCMC_refinement):
                dfs_samples = sampler.step(dfs_samples.float(), ebm_model).detach()
            dfs_samples_list.append(dfs_samples.long())
            q_probs_list.append(-ebm_model(dfs_samples.float()).squeeze())
    
        if args.source == 'data':
            xt = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
        else:
            xt = get_x0(B,D,S,args)

        x_fake = torch.tensor(gen_samples(dfs_model, args, xt=xt)).to(args.device).detach()
        for j in range(args.MCMC_refinement):
            x_fake = sampler.step(x_fake.float(), ebm_model).detach()
        dfs_pause_time = time.time()

        #step EBM – get E_k+1
        ebm_start_time = time.time()
        x = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
        
        #x_fake = dfs_samples_list[-1].clone().float() #recycling the dfs samples (only last one, but could use any)
        logp_real = -ebm_model(x.float()).squeeze()
        if args.p_control > 0:
            grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                            create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
            grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
        else:
            grad_reg = 0.0
        logp_fake = -ebm_model(x_fake.float()).squeeze()
        # logp_fake = q_probs_list[-1] #can't be detached!
        obj = logp_real.mean() - logp_fake.mean()
        ebm_loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean()) #note sign of objective

        ebm_optimizer.zero_grad()
        ebm_loss.backward()
        ebm_optimizer.step()

        for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
        ebm_end_time = time.time()

        dfs_restart_time = time.time()

        for (x1, log_q_prob) in zip(reversed(dfs_samples_list), reversed(q_probs_list)):
            B = x1.shape[0]
            if args.source == 'data':
                x0 = torch.from_numpy(get_batch_data(db, args)).to(args.device)  
            else:
                x0 = get_x0(B,D,S,args)

            t = torch.rand((B,)).to(args.device)
            xt = x1.clone()
            mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
            xt[mask] = x0[mask]

            log_p_prob = -ebm_model(x1.float()).detach()
            
            dfs_loss, weights, temp = compute_loss(dfs_model, B, D, S, log_p_prob, log_q_prob, t, x1, x0, args, temp)
            dfs_optimizer.zero_grad()
            dfs_loss.backward()
            dfs_optimizer.step()

            # update ema_model                 
            for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

        dfs_end_time = time.time()
        dfs_end_time = time.time()
        dfs_gen_times.append(dfs_pause_time - dfs_start_time)
        dfs_step_times.append(dfs_end_time - dfs_restart_time)
        ebm_times.append(ebm_end_time - ebm_start_time)

        if verbose:
            pbar.set_description(f'EBM loss: {ebm_loss}, DFS loss: {dfs_loss} avg dfs gen time: {sum(dfs_gen_times)/(epoch)} avg dfs step time: {sum(dfs_step_times)/(epoch)} avg ebm step time: {sum(ebm_times)/(epoch)}, mean logp_real was {logp_real.mean()}, mean logp_fake was {logp_fake.mean()}')

    if verbose:
        print(f'Epoch: {epoch}\{args.num_epochs}; Final DFS Loss: {dfs_loss}; EBM Loss: {ebm_loss}; mean logp_real: {logp_real.mean().item()}; mean logp_fake: {logp_fake.mean().item()}; Temp: {temp} \n')


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
        samples = gen_samples(dfs_model, args, batch_size = 2500, xt=xt)
        plot(f'{args.sample_path}/dfs_samples_{epoch}.png', torch.tensor(samples).float())
        ema_samples = gen_samples(ema_dfs_model, args, batch_size = 2500, xt=xt)
        plot(f'{args.sample_path}/ema_dfs_samples_{epoch}.png', torch.tensor(ema_samples).float())
        weights_dir = f'{args.plot_path}/weights_histogram_{epoch}.png'
        if not os.path.exists(weights_dir):
            plot_weight_histogram(weights, output_dir=weights_dir)
                    
        #save ebm samples
        EBM_samples = gibbs_sampler(ebm_model, num_rounds=100, num_samples=2500).to('cpu').detach()
        plot(f'{args.sample_path}/EBM_samples_{epoch}_steps_{args.save_sampling_steps}.png', EBM_samples)

        #log losses
        log_entry = {'epoch':None,'timestamp':None}
        log_entry['dfs_loss'] = dfs_loss.item()
        log_entry['ebm_loss'] = ebm_loss.item()
        log_entry['temp'] = temp
        log_entry['mean_weight'] = weights.mean().item()

        timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

        log(args, log_entry, epoch, timestamp)

    if args.eval_on:
        if (epoch % args.eval_every == 0) or (epoch == args.num_epochs):
            eval_start_time = time.time()

            print('Starting evaluation, this may take a while.')

            #save models
            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')
            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{epoch}.pt')

        
            #log
            log_entry = {'epoch':None,'timestamp':None}
            log_entry['dfs_loss'] = dfs_loss.item()
            log_entry['ebm_loss'] = ebm_loss.item()
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

