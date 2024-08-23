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
from utils.eval import plot_weight_histogram
import argparse
import utils.samplers as samplers
import utils.block_samplers as block_samplers
import torch.nn as nn
import utils.ais as ais

import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion

from velo_efm_ebm_bootstrap_2.model import ResNetFlow
from velo_efm_ebm_bootstrap_2.model import EBM
from velo_efm_ebm_bootstrap_2.model import Dataq



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


def gen_samples(model, args, batch_size=None, t=0.0, xt=None, print_stats=True):
    model.eval()
    S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
    D = args.discrete_dim
    # Variables, B, D for batch size and number of dimensions respectively
    B = batch_size if batch_size is not None else args.batch_size
    if not xt is None:
        B = xt.shape[0]

    if args.source == 'mask':
        M = S - 1
    # Initialize xt with the mask index value if not provided
    if xt is None:
        if args.source == 'mask':
            xt = M * torch.ones((B, D), dtype=torch.long).to(args.device)
        elif args.source == 'uniform':
            xt = torch.randint(0, S, (B, D)).to(args.device)

    t = 0.0  # Initial time

    while t < 1.0:
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 
        if args.scheduler_type == 'linear':
            a1 = 1 / (1 - t)
            b = -a1
            #adaptive dt
            if t>0:
                dt = min(args.delta_t, (1 - t))
            else:
                dt = args.delta_t
        elif args.scheduler_type == 'quadratic':
            a1 = (2 * t) / (1 - t ** 2)
            b = -a1
            #adaptive dt
            if t>0:
                dt = min(args.delta_t, (1 - t ** 2)/(2 * t))
            else:
                dt = args.delta_t
        elif args.scheduler_type == 'quadratic_noise':
            a1 = t * (2 - t)/(1 - t)
            a2 = 1 - t
            b = -1 /(1 - t)
            #adaptive dt 
            if t>0:
                dt = min(args.delta_t, 1 - t)
            else:
                dt = args.delta_t

        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            x1_logits, noise_logits = model(xt, t_)
            x1_logits = F.softmax(x1_logits, dim=-1)
            if args.scheduler_type == 'quadratic_noise':
                noise_logits = F.softmax(noise_logits, dim=-1)
                ut = a1 * x1_logits + a2 * noise_logits + b * delta_xt #a3 is zero...
            else:
                ut = a1 * x1_logits + b * delta_xt #a3 is zero...
        
        step_probs = delta_xt + (ut * dt)

        if args.impute_self_connections:
            step_probs = step_probs.clamp(max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 

        t += dt

        step_probs = step_probs.clamp(min=0) #can I avoid this?


        if t < 1.0 or not args.source == 'mask':
            xt = Categorical(step_probs).sample() #(B,D)
        else:
            if print_stats:
                if torch.any(xt == M):
                    num_masked_entries = torch.sum(xt == M).item()
                    print(f"Share of masked entries (over all entries) in the final but one tensor: {num_masked_entries/ (B * D)}")
                    print(f"Forcing mask values into range...")
                print(f'Share of samples with non-zero probability for at least one mask: {(step_probs[:,:,M].sum(dim=-1)>0.001).sum()/B}')
            step_probs[:, :, M] = 0
            step_probs_sum = step_probs.sum(dim=-1, keepdim=True)
            zero_sum_mask = step_probs_sum == 0
            if zero_sum_mask.any():
                step_probs[zero_sum_mask.expand(-1, -1, S).bool() & (torch.arange(S).to(args.device) < M).unsqueeze(0).unsqueeze(0).expand(B, D, S)] = 1/M
            # print(step_probs[zero_sum_mask.expand(-1, -1, S)])
            xt = Categorical(step_probs).sample() # (B, D)
            if torch.any(xt == M):
                num_masked_entries = torch.sum(xt == M).item()
                print(f"Forcing failed. Number of masked entries in the final tensor: {num_masked_entries}")

    return xt.detach().cpu().numpy()


def compute_bernoulli_loss(ebm_model, dfs_model, q_dist, xt, x1, t, args, x_noise=None):
    x1_logits, noise_logits = dfs_model(xt, t)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1)
    if args.scheduler_noise:
        noise_loss = F.cross_entropy(noise_logits.transpose(1,2), x_noise, reduction='none').sum(dim=-1)
        loss = loss + noise_loss

    acc = 'not computed'

    log_prob = -ebm_model(x1.float()).detach()


    log_q_density = q_dist.log_prob(x1.float()).sum(dim=-1).to(args.device).detach()
    log_weights = log_prob - log_q_density.detach()
    if args.norm_by_sum and not args.norm_by_max:
        log_norm = torch.logsumexp(log_weights, dim=-1) #make this a moving average?
    elif args.norm_by_max and not args.norm_by_sum:
        log_norm = torch.max(log_weights)
    else:
        raise NotImplementedError('Must either normalize by sum or max to avoid inf.')
    weights = (log_weights - log_norm).exp()
    loss = weights * loss #the math is wrong?
    loss = loss.mean(dim=0)

    return loss, acc, weights

def edfm_compute_loss(dfs_model, log_p_prob, log_q_prob, xt, x1, t, args, unweighted=False, temp=1, x_noise=None):
    x1_logits, noise_logits = dfs_model(xt, t)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1)
    if args.scheduler_noise:
        noise_loss = F.cross_entropy(noise_logits.transpose(1,2), x_noise, reduction='none').sum(dim=-1)
        loss = loss + noise_loss

    acc = 'not computed'
    
    if not unweighted:
        log_weights = log_p_prob - log_q_prob.detach()

        if args.optimal_temp:
            max_index = torch.argmax(log_weights)
            if args.optimal_temp_use_median:
                _, sorted_indices = torch.sort(log_weights)
                median_index = sorted_indices[len(log_weights) // 2]
                temp_t = 1/torch.log(torch.tensor(args.optimal_temp_diff)) * ((log_p_prob[median_index] - log_p_prob[max_index]) - (log_q_prob[median_index] - log_q_prob[max_index]))
            else:
                weights = log_weights.exp()
                mean_value = torch.mean(weights.float())
                diff = torch.abs(weights.float() - mean_value)
                mean_index = torch.argmin(diff) #lower complexity then median
                temp_t = 1/torch.log(torch.tensor(args.optimal_temp_diff)) * ((log_p_prob[mean_index] - log_p_prob[max_index]) - (log_q_prob[mean_index] - log_q_prob[max_index]))

            if temp_t < 1e-10:
                print(f'\n Reset temp_t to 1, which was at {temp_t}... \n', flush=True)
                temp_t = torch.tensor(1)

            temp = (args.optimal_temp_ema * temp + (1 - args.optimal_temp_ema) * temp_t).cpu().detach().item()

        log_weights = log_p_prob/temp - log_q_prob.detach()/temp
        if args.norm_by_sum and not args.norm_by_max:
            log_norm = torch.logsumexp(log_weights, dim=-1) #make this a moving average?
        elif args.norm_by_max and not args.norm_by_sum:
            log_norm = torch.max(log_weights)
        else:
            raise NotImplementedError('Must either normalize by sum or max to avoid inf.')
        # print(f'\n log norm is {log_norm} \n')

        weights = (log_weights - log_norm).exp()
        print(f'\n weights had mean {weights.mean()} with temp at {temp}', flush=True)

        loss = weights * loss #the math is wrong?
    else:
        weights = None

    loss = loss.mean(dim=0)

    return loss, acc, weights, temp

def dfm_compute_loss(model, xt, x1, t, args, x_noise=None):
    B, D = args.batch_size, args.discrete_dim
    S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
    if args.source == 'mask':
        M = S - 1

    x1_logits, noise_logits = model(xt, t)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1)
    if args.scheduler_noise:
        noise_loss = F.cross_entropy(noise_logits.transpose(1,2), x_noise, reduction='none').sum(dim=-1)
        loss = loss + noise_loss

    x_hat = torch.argmax(x1_logits, dim=-1)
    acc = (x_hat == x1).float().mean().item()

    return loss.mean(dim=0), acc


def main_loop(args, verbose=False):

    def get_independent_sample(loader):
        (x, _) = next(iter(loader))
        return preprocess(x)
    
    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    if args.source == 'data':
        train_loader_2 = copy.deepcopy(train_loader)
        if args.optional_step:
            train_loader_3 = copy.deepcopy(train_loader)
        train_loader_4 = copy.deepcopy(train_loader)
        train_loader_5 = copy.deepcopy(train_loader)

    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = (init_batch.mean(0) * (1. - 2 * eps) + eps).to(args.device)

    MCMC_init = torch.distributions.Bernoulli(probs=torch.ones((args.discrete_dim,)).to(args.device) * 0.5)


    if args.base_dist == 'uniform':
        q_init = torch.distributions.Bernoulli(probs=torch.ones((args.discrete_dim,)).to(args.device) * 0.5)
    elif args.base_dist == 'data_mean':
        q_init = torch.distributions.Bernoulli(probs=init_mean)
    elif args.base_dist == 'zero':
        q_init = torch.distributions.Bernoulli(probs=torch.ones((args.discrete_dim,)).to(args.device) * 0.5)

    if args.mixer_step:
        if args.mixer_type == 'uniform':
            q_mixer =  torch.distributions.Bernoulli(probs=torch.ones((args.discrete_dim,)).to(args.device) * 0.5)
        elif args.mixer_type == 'data_mean':
            q_mixer =  torch.distributions.Bernoulli(probs=init_mean)

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

    if args.base_dist == 'uniform':
        ebm_model = EBM(net, torch.ones((args.discrete_dim,)).to(args.device) * 0.5)
    elif args.base_dist == 'data_mean':
        ebm_model = EBM(net, init_mean)
    elif args.base_dist == 'zero':
        ebm_model = EBM(net)

    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr)
    ebm_optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.ebm_lr, weight_decay=args.weight_decay)

    ema_ebm_model = copy.deepcopy(ebm_model)
    ema_dfs_model = copy.deepcopy(dfs_model)


    # if args.checkpoint is not None:
    #     d = torch.load(args.checkpoint)
    #     model.load_state_dict(d['model'])
    #     ema_model.load_state_dict(d['ema_model'])
    #     buffer = d['buffer']

    # move to cuda
    ebm_model.to(args.device)
    ema_ebm_model.to(args.device)
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)

    #set temperature
    temp = args.start_temp
    if not args.optimal_temp:
        temp_decay = args.temp_decay
        print(f'temp decay set to {temp_decay}')

    #get sampler
    sampler = get_sampler(args) #for eval


    # my_print(args.device)
    # my_print(ebm_model)
    # my_print(dfs_model)


    best_val_ll = -np.inf
    dfs_lr = args.lr
    ebm_lr = args.ebm_lr

    test_ll_list = []

    start_time = time.time()
    cum_eval_time = 0

    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(args.device))
    reinit_dist = torch.distributions.Bernoulli(probs=torch.tensor(args.reinit_freq))

    all_inds = list(range(args.buffer_size))
    epoch = 0
    itr = 0
    if args.dfs_warmup_iter > 0:
        dfs_per_ebm = args.dfs_warmup_iter
        in_warmup = True
    else:
        dfs_per_ebm = args.dfs_per_ebm
        in_warmup = False

    #define dimensions
    B, D = args.batch_size, args.discrete_dim
    S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size

    #initialisation warmup; need p_EBM = p_EDFM
    init_iter = args.init_iter if args.source == 'uniform' else args.init_iter

    print(f'Starting initialisation. Base dist is {args.base_dist}, q_init is Bern(0.5) if constant base dist, or Bern(data_mean) if data_mean base dist...')
    
    if args.scheduler_type == 'quadratic_noise':
        has_noise = '_has_noise'
    else:
        has_noise = ''

    if not args.dfs_init_from_checkpoint:
        init_start_time = time.time()
        for i in range(1, init_iter + 1):
            ebm_model.eval()
            dfs_model.train()
            #sample x1 from q_init
            x1 = q_init.sample((B,)).to(args.device).long()
            B = x1.shape[0]

            x1_energies = ebm_model(x1.float())

            #make t, xt
            if args.source == 'mask':
                M = S - 1
                x0 = torch.ones((B,D)).to(args.device).long() * M
            elif args.source == 'uniform':
                x0 = torch.randint(0, S, (B, D)).to(args.device)
            elif args.source == 'data':
                x0 = get_independent_sample(train_loader).long().to(args.device)
            
            if args.scheduler_type == 'quadratic_noise':
                x_noise = torch.randint(0, S, (B, D)).to(args.device)

            t = torch.rand((B,)).to(args.device)

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
                loss, acc, weights = compute_bernoulli_loss(ebm_model, dfs_model, q_init, xt, x1, t, args, x_noise=x_noise)
            else:
                loss, acc, weights = compute_bernoulli_loss(ebm_model, dfs_model, q_init, xt, x1, t, args)

            dfs_optimizer.zero_grad()
            loss.backward()
            dfs_optimizer.step()

            #print weights (should be all 1 at when converged)

            if (i % args.init_save_every) == 0:
                output_dir = f'{args.plot_path}/warmup_dfs_weights_histogram_{i}.png'
                if not os.path.exists(output_dir):
                    plot_weight_histogram(weights, output_dir=output_dir)
                    print(f'Init plot weight histograms saved...')
                    dfs_model.eval()
                    if args.source == 'data':
                        xt = get_independent_sample(test_loader).long().to(args.device) 
                    else:
                        xt = None
                    dfs_samples = gen_samples(dfs_model, args, batch_size=args.batch_size, xt=xt)
                    plot(f'{args.sample_path}/warmup_dfs_samples_{i}.png', torch.tensor(dfs_samples).float())
                    EBM_samples = MCMC_init.sample((100,))
                    for d in range(args.init_sampling_steps):
                        EBM_samples = sampler.step(EBM_samples.detach(), ebm_model).detach()
                    EBM_samples = EBM_samples.float().detach()
                    plot(f'{args.sample_path}/warmup_EBM_samples_{i}_steps_{args.init_sampling_steps}.png', EBM_samples)
                print(f'Init iteration {i}, Loss: {loss.item()}')
                print(f'Energies computed for a q_init sample. Found mean {x1_energies.mean()} and variance {x1_energies.var()}')
        torch.save(dfs_model.state_dict(), f'{args.save_dir}/init_dfs_model_base_dist-{args.base_dist}_source-{args.source}_{args.init_iter}{has_noise}.pt')
        init_end_time = time.time()
        print(f'Init took {init_end_time - init_start_time} – saved initialized dfs at /init_dfs_model_base_dist-{args.base_dist}_source-{args.source}_{args.init_iter}.pt')
    else:
        params = torch.load(f'{args.save_dir}/init_dfs_model_base_dist-{args.base_dist}_source-{args.source}_{args.init_iter}{has_noise}.pt')
        dfs_model.load_state_dict(params)

    start_time = time.time()
    cum_eval_time = 0

    while epoch < args.num_epochs:
        if verbose:
            print(f'starting epoch {epoch}/{args.num_epochs}')

        pbar = tqdm(train_loader) if verbose else train_loader
        if args.source == 'data':
            iter_train_loader_2 = iter(train_loader_2 )
            if args.optional_step:
                iter_train_loader_3 = iter(train_loader_3)
            iter_train_loader_4 = iter(train_loader_4)
            iter_train_loader_5 = iter(train_loader_5)


        ebm_times = []
        dfs_times = []
        optional_step_times = []

        #torch.autograd.set_detect_anomaly(True)
            
        for k, x in enumerate(pbar):
            if itr < args.warmup_iters:
                ebm_lr = args.ebm_lr * float(itr) / args.warmup_iters
                for param_group in ebm_optimizer.param_groups:
                    param_group['lr'] = ebm_lr

            ebm_model.train()
            dfs_model.train()

            #generate samples and compute E_k
            dfs_start_time = time.time()
            #dfs_samples for EBM step
            if args.source == 'data':
                (x_source, _) = next(iter_train_loader_4)
                xt = preprocess(x_source).long().to(args.device)  
            else:
                xt = None

            #need for dfs and ebm
            x = preprocess(x[0].to(args.device).requires_grad_())


            if itr == 0 or not args.recycle_dfs_sample: 
                x_fake = torch.tensor(gen_samples(dfs_model, args, batch_size=x.shape[0], print_stats=False, xt=xt)).to(args.device).detach()
        
            x1 = x.long().detach()
            (B, D) = x1.shape

            #dfs step
            if args.source == 'mask':
                M = S - 1
                x0 = torch.ones((B,D)).to(args.device).long() * M
            elif args.source == 'uniform':
                x0 = torch.randint(0, S, (B, D)).to(args.device)
            elif args.source == 'data':
                (x_source, _) = next(iter_train_loader_2)
                x0 = preprocess(x_source).long().to(args.device)  
            if args.scheduler_type == 'quadratic_noise':
                x_noise = torch.randint(0, S, (B, D)).to(args.device)

            t = torch.rand((B,)).to(args.device)

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
                dfm_loss, acc = dfm_compute_loss(dfs_model, xt, x1, t, args, x_noise)
            else:
                dfm_loss, acc = dfm_compute_loss(dfs_model, xt, x1, t, args)

            dfs_optimizer.zero_grad()
            dfm_loss.backward()
            dfs_optimizer.step()

            dfs_end_time = time.time()

            ebm_start_time = time.time()

            #step EBM – get E_k+1
            if args.source == 'data':
                (x_source, _) = next(iter_train_loader_5)
                x0 = preprocess(x_source).long().to(args.device)  
            else:
                xt = None
            x_dfs = torch.tensor(gen_samples(dfs_model, args, batch_size=x.shape[0], print_stats=False, xt=xt)).to(args.device).detach()
            
            logp_real = -ebm_model(x).squeeze()
            logp_dfs = -ebm_model(x_dfs.float()).squeeze()
            logp_fake = -ebm_model(x_fake.float()).squeeze()

            obj_CD = logp_real.mean() - logp_fake.mean()
            obj_DFS = logp_dfs.mean() - logp_fake.mean()
            obj = args.alpha * obj_CD + (1 - args.alpha) * obj_DFS

            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                              create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg = 0.0
            # logp_fake = q_probs_list[-1] #can't be detached!
            ebm_loss = obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean()) #note sign of objective

            ebm_optimizer.zero_grad()
            ebm_loss.backward()
            ebm_optimizer.step()

            for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
            ebm_end_time = time.time()

            #without the optional step for now
            if args.optional_step:
                optional_step_start_time = time.time()
                logp_dfs_old = logp_dfs
                logp_dfs = -ebm_model(x_dfs.float()).squeeze()

                if args.source == 'mask':
                    M = S - 1
                    x0 = torch.ones((B,D)).to(args.device).long() * M
                elif args.source == 'uniform':
                    x0 = torch.randint(0, S, (B, D)).to(args.device)
                elif args.source == 'data':
                    (x_source, _) = next(iter_train_loader_3)
                    x0 = preprocess(x_source).long().to(args.device)  

                if args.scheduler_type == 'quadratic_noise':
                    x_noise = torch.randint(0, S, (B, D)).to(args.device)

                t = torch.rand((B,)).to(args.device)

                if args.scheduler_type == 'linear':
                    kappa1 = t
                elif args.scheduler_type == 'quadratic':
                    kappa1 = torch.square(t)
                elif args.scheduler_type == 'quadratic_noise':
                    kappa1 = torch.square(t)
                    kappa2 = t - torch.square(t)

                xt = x_dfs.clone()
                mask0 = torch.rand((B,D)).to(args.device) < (1 - kappa1[:, None])
                xt[mask0] = x0[mask0]
                if args.scheduler_type == 'quadratic_noise':
                    mask_noise = torch.rand((B,D)).to(args.device) < (kappa2/(1 - kappa1))[:, None]
                    mask_noise = mask_noise & mask0
                    xt[mask_noise] = x_noise[mask_noise]
                    dfs_loss, acc, weights, temp = edfm_compute_loss(dfs_model, logp_dfs, logp_dfs_old, xt, x_dfs, t, args, temp=temp, x_noise=x_noise)
                else:
                    dfs_loss, acc, weights, temp = edfm_compute_loss(dfs_model, logp_dfs, logp_dfs_old, xt, x_dfs, t, args, temp=temp)

                dfs_optimizer.zero_grad()
                dfs_loss.backward()
                dfs_optimizer.step()

                optional_step_end_time = time.time()
                optional_step_times.append(optional_step_end_time - optional_step_start_time)
            
            dfs_times.append(dfs_end_time - dfs_start_time)
            ebm_times.append(ebm_end_time - ebm_start_time)

            if args.recycle_dfs_sample:
                x_fake = x_dfs

            
            if verbose and args.optional_step:
                pbar.set_description(f'EBM loss (obj_CD, obj_DFS): {ebm_loss.item()} ({obj_CD.item()}, {obj_DFS.item()}) DFM loss: {dfm_loss.item()}, DFS loss: {dfs_loss.item()} avg dfs time: {sum(dfs_times)/(k+1)},  avg ebm step time: {sum(ebm_times)/(k+1)}, avg optional step time: {sum(optional_step_times)/(k+1)}, \n mean logp_real: {logp_real.mean()}, mean logp_fake: {logp_fake.mean()}, mean weights: {weights.mean()}, temp: {temp}')
            elif verbose:
                pbar.set_description(f'EBM loss (obj_CD, obj_DFS): {ebm_loss.item()}, ({obj_CD.item()}, {obj_DFS.item()}) DFM loss: {dfm_loss.item()}, avg dfs time: {sum(dfs_times)/(k+1)} \n avg ebm step time: {sum(ebm_times)/(k+1)}, mean logp_real: {logp_real.mean()}, mean logp_fake: {logp_fake.mean()}')


            if args.itr_save != 0 and (itr % args.itr_save == 0) and itr > 0:
                ebm_model.eval()
                dfs_model.eval()
                eval_start_time = time.time()

                if args.source == 'data':
                    xt = get_independent_sample(test_loader).long().to(args.device) 
                else:
                    xt = None
                samples = gen_samples(dfs_model, args, 100, xt=xt)
                plot(f'{args.sample_path}/dfs_samples_{epoch}_iter_{itr}.png', torch.tensor(samples).float())
                if args.store_dfs_ema:
                    ema_samples = gen_samples(ema_dfs_model, args, 100, xt=xt)
                    plot(f'{args.sample_path}/dfs_ema_samples_{epoch}_iter_{itr}.png', torch.tensor(ema_samples).float())
                if args.optional_step:
                    output_dir = f'{args.plot_path}/final_dfs_weights_histogram_{epoch}_iter_{itr}.png'
                    if not os.path.exists(output_dir):
                        plot_weight_histogram(weights, output_dir=output_dir)
                plot(f'{args.sample_path}/data_samples_{epoch}_iter_{itr}.png', x.detach().cpu())
                plot(f'{args.sample_path}/fake_samples_{epoch}_iter_{itr}.png', x_fake.float())
                EBM_samples = MCMC_init.sample((100,))
                for d in range(args.save_sampling_steps):
                    EBM_samples = sampler.step(EBM_samples.detach(), ebm_model).detach()
                EBM_samples = EBM_samples.cpu().detach()
                plot(f'{args.sample_path}/EBM_samples_{epoch}_iter_{itr}_steps_{args.save_sampling_steps}.png', EBM_samples)
                eval_end_time = time.time()
                eval_time = eval_end_time - eval_start_time
                cum_eval_time += eval_time
                timestamp = time.time() - cum_eval_time - start_time

            itr += 1

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            ebm_model.eval()
            dfs_model.eval()
            eval_start_time = time.time()

            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')

            log_entry = {'epoch':None,'timestamp':None}
            log_entry['dfm_loss'] = dfm_loss.item()
            if args.optional_step:
                log_entry['dfs_loss'] = dfs_loss.item()
                log_entry['temp'] = temp
            log_entry['ebm_loss'] = ebm_loss.item()
            if args.store_dfs_ema:
                torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{epoch}.pt')
            if args.source == 'data':
                xt = get_independent_sample(test_loader).long().to(args.device) 
            else:
                xt = None
            samples = gen_samples(dfs_model, args, 100, xt=xt)
            plot(f'{args.sample_path}/dfs_samples_{epoch}.png', torch.tensor(samples).float())
            if args.store_dfs_ema:
                ema_samples = gen_samples(ema_dfs_model, args, 100, xt=xt)
                plot(f'{args.sample_path}/dfs_ema_samples_{epoch}.png', torch.tensor(ema_samples).float())
            if args.optional_step:
                output_dir = f'{args.plot_path}/final_dfs_weights_histogram_{epoch}.png'
                if not os.path.exists(output_dir):
                    plot_weight_histogram(weights, output_dir=output_dir)
            plot(f'{args.sample_path}/data_samples_{epoch}.png', x.detach().cpu())
            plot(f'{args.sample_path}/fake_samples_{epoch}.png', x_fake.float())
            EBM_samples = MCMC_init.sample((100,))
            for d in range(args.save_sampling_steps):
                EBM_samples = sampler.step(EBM_samples.detach(), ebm_model).detach()
            EBM_samples = EBM_samples.cpu().detach()
            plot(f'{args.sample_path}/EBM_samples_{epoch}_steps_{args.save_sampling_steps}.png', EBM_samples)
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time
            
        if not args.optimal_temp:
            temp = temp * temp_decay

        if args.eval_on:
            if (epoch % args.eval_every == 0) or (epoch == args.num_epochs - 1):
                eval_start_time = time.time()

                log_entry = {'epoch':None,'timestamp':None}

                logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ema_ebm_model, init_dist, sampler,
                                                                            train_loader, val_loader, test_loader,
                                                                            preprocess, args.device,
                                                                            args.eval_sampling_steps,
                                                                            args.test_batch_size)
                log_entry['EMA Train log-likelihood'] = train_ll.item()
                log_entry['EMA Train log-likelihood'] = val_ll.item()
                log_entry['EMA Test log-likelihood'] = test_ll.item()
                
                for _i, _x in enumerate(ais_samples):
                    plot(f'{args.sample_path}/EBM_EMA_sample_{args.dataset_name}_{args.sampler}_{args.step_size}_{epoch}_{_i}.png', _x)

                ebm_model.cpu()
                d = {}
                d['ebm_model'] = ebm_model.state_dict()
                d['ema_ebm_model'] = ema_ebm_model.state_dict()
                d['ebm_optimizer'] = ebm_optimizer.state_dict()
                if val_ll.item() > 0:
                    exit()
                if val_ll.item() > best_val_ll:
                    best_val_ll = val_ll.item()
                    my_print("Best valid likelihood")
                    torch.save(d, f"{args.ckpt_path}/best_ckpt_{args.dataset_name}_{args.sampler}_{args.step_size}.pt")
                else:
                    torch.save(d, f"{args.ckpt_path}/ckpt_{args.dataset_name}_{args.sampler}_{args.step_size}.pt")

                eval_time = eval_end_time - eval_start_time
                cum_eval_time += eval_time
                timestamp = time.time() - cum_eval_time - start_time

                log(args, log_entry, epoch, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/log.csv')
        epoch += 1

    log_completion(args.methods, args.dataset_name, args)

