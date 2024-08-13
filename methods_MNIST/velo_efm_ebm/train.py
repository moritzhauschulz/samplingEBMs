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

from velo_efm_ebm.model import ResNetFlow
from velo_efm_ebm.model import EBM
from velo_efm_ebm.model import Dataq



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

    if args.source == 'mask':
        M = S - 1

    # Initialize xt with the mask index value if not provided
    if xt is None:
        if args.source == 'mask':
            xt = M * torch.ones((B, D), dtype=torch.long).to(args.device)
        else:
            xt = torch.randint(0, S, (B, D)).to(args.device)



    dt = args.delta_t  # Time step
    t = 0.0  # Initial time

    while t < 1.0:
        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            x1_logits = model(xt, t_).to(args.device)
            x1_logits = F.softmax(x1_logits, dim=-1)
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 
        ut = 1/(1-t) * (x1_logits - delta_xt)

        step_probs = delta_xt + (ut * dt)

        if args.impute_self_connections:
            step_probs = step_probs.clamp(max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 

        t += dt

        step_probs = step_probs.clamp(min=0) #can I avoid this?


        if t < 1.0 or not args.source == 'mask' :
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

def compute_bernoulli_loss(ebm_model, dfs_model, q_dist, xt, x1, t, args):
    x1_logits = dfs_model(xt, t).to(args.device)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1) #maybe need to ignore index

    x_hat = torch.argmax(x1_logits, dim=-1)
    acc = (x_hat == x1).float().mean().item()

    log_prob = -ebm_model(x1.float())


    log_q_density = q_dist.log_prob(x1.float()).sum(dim=-1).to(args.device)
    log_weights = log_prob - log_q_density
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

def compute_loss(ebm_model, temp, dfs_model, q_dist, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size_with_mask
    M = S - 1

    x1_logits = dfs_model(xt, t).to(args.device)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1) #maybe need to ignore index
    # if args.impute_self_connections:            #is this appropriate here?
    #     loss.scatter_(-1, xt[:, :, None], 0.0)
    #loss = loss.sum(dim=(1, 2))

    x_hat = torch.argmax(x1_logits, dim=-1)
    acc = (x_hat == x1).float().mean().item()

    
    log_prob = -ebm_model(x1.float())
    
    print(f'temp set to {temp}')
    log_q_density = q_dist.get_last_log_likelihood()
    log_weights = log_prob - log_q_density
    if args.norm_by_sum and not args.norm_by_max:
        log_norm = torch.logsumexp(log_weights, dim=-1) #make this a moving average?
    elif args.norm_by_max and not args.norm_by_sum:
        log_norm = torch.max(log_weights)
    else:
        raise NotImplementedError('Must either normalize by sum or max to avoid inf.')
    # print(f'\n log norm is {log_norm} \n')
    weights = (log_weights - log_norm).exp()

    if args.optimal_temp:
        max_index = torch.argmax(weights)
        if args.optimal_temp_use_median:
            _, sorted_indices = torch.sort(weights)
            median_index = sorted_indices[len(weights) // 2]
            # print(f'diff in log q is {log_q_density[median_index] - log_q_density[max_index]}')
            log_base = torch.log(torch.tensor(0.5)).to(args.device) + log_q_density[median_index] - log_q_density[max_index] #higher complexity then mean
            # print(f'log base is {log_base.item()}')
            # print(f'mean log prob is {log_prob[median_index]}')
            # print(f'max log prob is {log_prob[max_index]}')
            temp_t = 1/log_base.to(args.device) * (log_prob[median_index] - log_prob[max_index])
        else:
            mean_value = torch.mean(weights.float())
            # print(f'mean weight is {mean_value.item()}')
            diff = torch.abs(weights.float() - mean_value)
            mean_index = torch.argmin(diff) #lower complexity then median
            # print(f'diff in log q is {log_q_density[mean_index] - log_q_density[max_index]}')
            log_base = torch.log(torch.tensor(0.5)).to(args.device) + log_q_density[mean_index] - log_q_density[max_index]  #higher complexity then mean
            # print(f'log base is {log_base.item()}')
            # print(f'mean log prob is {log_prob[mean_index]}')
            # print(f'max log prob is {log_prob[max_index]}')
            temp_t = 1/log_base * (log_prob[mean_index] - log_prob[max_index])

        if temp_t < 1e-10:
            temp_t = torch.tensor(1)
            print(f'\n Reset temp_t to 1, which was at {temp_t}... \n')

        temp = (args.optimal_temp_ema * temp + (1 - args.optimal_temp_ema) * temp_t).cpu().detach().item()
        print(f'temp is {temp}')

    log_weights = log_prob/temp - log_q_density

    if args.norm_by_sum and not args.norm_by_max:
        log_norm = torch.logsumexp(log_weights, dim=-1) #make this a moving average?
    elif args.norm_by_max and not args.norm_by_sum:
        log_norm = torch.max(log_weights)
    else:
        raise NotImplementedError('Must either normalize by sum or max to avoid inf.')
    # print(f'\n log norm is {log_norm} \n')
    weights = (log_weights - log_norm).exp()

    loss = weights * loss #the math is wrong?
    loss = loss.mean(dim=0)

    return loss, acc, weights, temp


def main_loop(args, verbose=False):

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

     # make dfs model
    dfs_model = ResNetFlow(64, args)
    
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = (init_batch.mean(0) * (1. - 2 * eps) + eps).to(args.device)

    if args.use_dula and args.use_buffer:
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
        
        buffer = buffer.to('cpu')


    q_dist = Dataq(args, bernoulli_mean=init_mean)
    bernoulli_dist = torch.distributions.Bernoulli(probs=init_mean)

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

    if args.base_dist:
        ebm_model = EBM(net, init_mean)
    else:
        ebm_model = EBM(net)

    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr)
    ebm_optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.ebm_lr, weight_decay=args.weight_decay)

    ema_ebm_model = copy.deepcopy(ebm_model)
    ema_dfs_model = copy.deepcopy(dfs_model)


    if args.checkpoint is not None:
        d = torch.load(args.checkpoint)
        model.load_state_dict(d['model'])
        ema_model.load_state_dict(d['ema_model'])
        buffer = d['buffer']

    # move to cuda
    ebm_model.to(args.device)
    ema_ebm_model.to(args.device)
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)

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

    temp = args.start_temp
    if not args.optimal_temp:
        temp_decay = (args.end_temp/args.start_temp) ** (1/args.num_epochs)
        print(f'temp decay set to {temp_decay}')
    start_time = time.time()
    cum_eval_time = 0
    while epoch < args.num_epochs:
        if verbose:
            print(f'starting epoch {epoch}/{args.num_epochs}')


        pbar = tqdm(train_loader) if verbose else train_loader
        dfs_times = []
        ebm_times = []
        for k, x in enumerate(pbar):
            if itr < args.warmup_iters:
                ebm_lr = args.ebm_lr * float(itr) / args.warmup_iters
                for param_group in ebm_optimizer.param_groups:
                    param_group['lr'] = ebm_lr

            ebm_model.eval()
            dfs_model.train()

            dfs_start_time = time.time()
            if not args.use_dula:
                for i in range(dfs_per_ebm):
                    B, D = args.batch_size, args.discrete_dim
                    S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size

                    if args.bernoulli_step:
                        x1 = bernoulli_dist.sample((args.batch_size,)).to(args.device).long()
                        if args.source == 'mask':
                            M = S - 1
                            x0 = torch.ones((B,D)).to(args.device).long() * M
                        else:
                            x0 = torch.randint(0, S, (B, D)).to(args.device)

                        t = torch.rand((B,)).to(args.device)
                        xt = x1.clone()
                        mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
                        xt[mask] = x0[mask]

                        dfs_loss, acc, weights = compute_bernoulli_loss(ebm_model, dfs_model, bernoulli_dist, xt, x1, t, args)
                        dfs_optimizer.zero_grad()
                        dfs_loss.backward()
                        dfs_optimizer.step()

                    dfs_samples = torch.tensor(gen_samples(dfs_model, args, print_stats=False)).to(args.device)
                    x1 = q_dist.sample(dfs_samples).to(args.device).long()
                    
                    if args.source == 'mask':
                        M = S - 1
                        x0 = torch.ones((B,D)).to(args.device).long() * M
                    else:
                        x0 = torch.randint(0, S, (B, D)).to(args.device)

                    t = torch.rand((B,)).to(args.device)
                    xt = x1.clone()
                    mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
                    xt[mask] = x0[mask]

                    dfs_loss, acc, weights, temp = compute_loss(ebm_model, temp, dfs_model, q_dist, xt, x1, t, args)
                    dfs_optimizer.zero_grad()
                    dfs_loss.backward()
                    dfs_optimizer.step()

                    # update ema_model
                    for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                        ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

                dfs_end_time = time.time()
                dfs_times.append(dfs_end_time - dfs_start_time)
            
                dfs_per_ebm = args.dfs_per_ebm
                in_warmup = False

            ebm_model.train()
            dfs_model.eval()

            ebm_start_time = time.time()
            x = preprocess(x[0].to(args.device).requires_grad_())
            if not args.use_dula:
                x_fake = dfs_samples.float() #recycling the dfs samples
            elif args.use_buffer:
                ebm_model.eval()
                # choose random inds from buffer
                buffer_inds = sorted(np.random.choice(all_inds, args.batch_size, replace=False))
                x_buffer = buffer[buffer_inds].to(args.device)
                reinit = reinit_dist.sample((args.batch_size,)).to(args.device)
                x_reinit = init_dist.sample((args.batch_size,)).to(args.device)
                x_fake = x_reinit * reinit[:, None] + x_buffer * (1. - reinit[:, None])

                st = time.time()
                for k in range(args.sampling_steps):
                    x_fake_new = sampler.step(x_fake.detach(), ebm_model).detach()
                    x_fake = x_fake_new
                st = time.time() - st

                # update buffer
                buffer[buffer_inds] = x_fake.detach().cpu()

                ebm_model.train()
            else:
                ebm_model.eval()
                EBM_samples = init_dist.sample((args.batch_size,))
                for d in range(args.sampling_steps):
                    EBM_samples = sampler.step(EBM_samples.detach(), ebm_model).detach()
                x_fake = EBM_samples.float().detach()
                ebm_model.train()

                
            logp_real = ebm_model(x).squeeze()
            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                              create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg = 0.0

            logp_fake = ebm_model(x_fake).squeeze()

            obj = logp_real.mean() - logp_fake.mean()
            ebm_loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

            ebm_optimizer.zero_grad()
            ebm_loss.backward()
            ebm_optimizer.step()
            

            # update ema_model
            for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            ebm_end_time = time.time()
            ebm_times.append(ebm_end_time - ebm_start_time)

            # if verbose:
            #     pbar.set_description(f'avg dfs step time: {sum(dfs_times)/(k+1)}; avg ebm step time: {sum(ebm_times)/(k+1)}; temp: {temp}')
            itr += 1
        
        if not args.optimal_temp:
            temp = temp * temp_decay

        if verbose and not args.use_dula:
            print(f'Epoch: {epoch}\{args.num_epochs} EBM Loss: {ebm_loss} Final DFS Loss: {dfs_loss} logp_real: {logp_real} logp_fake: {logp_fake} Temp: {temp} \n avg dfs step time: {sum(dfs_times)/(k+1)} avg ebm step time: {sum(ebm_times)/(k+1)}')
        else:
            print(f'Epoch: {epoch}\{args.num_epochs} EBM Loss: {ebm_loss} logp_real: {logp_real} logp_fake: {logp_fake}')


        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            eval_start_time = time.time()

            torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')

            log_entry = {'epoch':None,'timestamp':None}
            
            if not args.use_dula:
                log_entry['dfs_loss'] = dfs_loss.item()
                log_entry['temp'] = temp
            # log_entry['dfs_acc'] = acc
            log_entry['ebm_loss'] = ebm_loss.item()
            torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{epoch}.pt')

            if not args.use_dula:
                plot(f'{args.sample_path}/q_samples_{epoch}_first_ten_types_{q_dist.get_last_is_empirical()[0:10].cpu().numpy()}.png', torch.tensor(x1).float())
                samples = gen_samples(dfs_model, args, 100)
                plot(f'{args.sample_path}/dfs_samples_{epoch}.png', torch.tensor(samples).float())
                ema_samples = gen_samples(ema_dfs_model, args, 100)
                plot(f'{args.sample_path}/dfs_ema_samples_{epoch}.png', torch.tensor(ema_samples).float())
                output_dir = f'{args.plot_path}/dfs_weights_histogram_{epoch}.png'
                if not os.path.exists(output_dir):
                    print(f'weights shape: {weights.shape}')
                    plot_weight_histogram(weights, output_dir=output_dir)
            plot(f'{args.sample_path}/data_samples_{epoch}.png', x.detach().cpu())
            plot(f'{args.sample_path}/fake_samples_{epoch}.png', x_fake)
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            EBM_samples = init_dist.sample((100,))
            model = lambda x: ebm_model(x)
            for d in range(args.save_sampling_steps):
                EBM_samples = sampler.step(EBM_samples.detach(), model).detach()
            EBM_samples = EBM_samples.cpu().detach()
            plot(f'{args.sample_path}/EBM_samples_{epoch}_steps_{args.save_sampling_steps}.png', EBM_samples)

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
                # d['buffer'] = buffer
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

