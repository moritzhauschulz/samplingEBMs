import numpy as np
import torch
import torch.nn as nn
import torchvision
import os, sys
import utils.mlp as mlp

import time
import random
import ipdb, pdb
from tqdm import tqdm
import argparse

from eb_gfn.model import get_GFlowNet
from utils.model  import MLPScore, EBM

import utils.vamp_utils as vamp_utils


from utils import utils
from utils import sampler
from utils.utils import get_sampler
from utils.utils import plot as toy_plot
from utils.eval import ebm_evaluation
from utils.eval import sampler_evaluation
from utils.eval import sampler_ebm_evaluation
from utils.eval import log
from utils.utils import get_batch_data
from utils.eval import log_completion
from utils.eval import get_eval_timestamp
from utils.eval import make_plots
import utils.ais as ais

from utils.toy_data_lib import get_db

def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

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

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    if args.down_sample:
        assert args.ebm_model.startswith("mlp-")

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
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = (init_batch.mean(0) * (1. - 2 * eps) + eps).to(args.device)

    init_dist = torch.distributions.Bernoulli(probs=init_mean)

    if args.base_dist:
        ebm_model = EBM(net, init_mean)
    else:
        ebm_model = EBM(net)

    #get sampler
    sampler = get_sampler(args) #for eval


    optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.ebm_lr)

    xdim = np.prod(args.input_size)
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, args.device)
    ebm_model.to(args.device)

    start_time = time.time()
    cum_eval_time = 0

    best_val_ll = -np.inf

    itr = 1
    while itr <= args.num_itr:
        for _, x in enumerate(train_loader):
            st = time.time()
            x = preprocess(x[0].to(args.device))  #  -> (bs, 784)
            B = x.shape[0]

            if args.gradnorm > 0:
                x.requires_grad_()
            gfn.model.train()

            update_success_rate = -1.
            assert "tb" in args.type
            gfn_loss_start = time.time()
            train_loss, train_logZ = gfn.train(B, scorer=lambda inp: -ebm_model(inp).detach(),
                   silent=itr % args.print_every != 0, data=x, back_ratio=args.back_ratio)
            gfn_loss_end = time.time()
            if itr % args.print_every == 0:
                print(f'gfn train took {gfn_loss_end-gfn_loss_start}')



            if args.rand_k or args.lin_k or (args.K > 0):
                if args.rand_k:
                    K = random.randrange(xdim) + 1
                elif args.lin_k:
                    K = min(xdim, int((xdim *  itr)/ args.warmup_baf))
                    K = max(K, 1)
                elif args.K > 0:
                    K = args.K
                else:
                    raise ValueError

                gfn.model.eval()
                x_fake, delta_logp_traj = gfn.backforth_sample(x, K)

                delta_logp_traj = delta_logp_traj.detach()
                if args.with_mh:
                    # MH step, calculate log p(x') - log p(x)
                    lp_update = -ebm_model(x_fake).squeeze() - (-ebm_model(x).squeeze())
                    update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
                    updates = update_dist.sample()
                    x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
                    update_success_rate = updates.mean().item()
                else:
                    update_success_rate = None
            else:
                x_fake = gfn.sample(args.batch_size)
                update_success_rate = None
                K = None
            

            if itr % args.ebm_every == 0:
                st = time.time() - st

                ebm_model.train()
                logp_real = -ebm_model(x).squeeze()
                if args.gradnorm > 0:
                    grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                      create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                    grad_reg = (grad_ld ** 2. / 2.).mean()
                else:
                    grad_reg = torch.tensor(0.).to(args.device)

                logp_fake = -ebm_model(x_fake).squeeze()
                obj = logp_real.mean() - logp_fake.mean()
                l2_reg = (logp_real ** 2.).mean() + (logp_fake ** 2.).mean()
                loss = -obj + grad_reg * args.gradnorm + args.l2 * l2_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if itr % args.print_every == 0:
                print(f'Itr: {itr}, K: {K}, gfn_loss: { train_loss.item()}, ebm_loss: {loss.item()}, success_rate: {update_success_rate}')


            if (itr % args.itr_save == 0) or (itr == args.num_itr):
                eval_start_time = time.time()

                #save models
                torch.save(gfn.model.state_dict(), f'{args.ckpt_path}/gfn_model_{itr}.pt')
                torch.save(ebm_model.state_dict(), f'{args.ckpt_path}/ebm_model_{itr}.pt')

                #save gfn samples
                gfn_samples = gfn.sample(100).detach()
                gfn_samp_float = gfn_samples.data.cpu().numpy().astype(int)
                plot(f'{args.sample_path}/gfn_samples_{itr}.png', torch.tensor(gfn_samp_float).float())

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
                log_entry['gfn_loss'] = train_loss.item()
                log_entry['ebm_loss'] = loss.item()

                #save success rate
                if not update_success_rate is None:
                    log_entry['MH_update_success_rate'] = update_success_rate


                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp)
            
            if (itr % args.eval_every == 0) or (itr == args.num_itr):
                print('Starting evaluation, this may take a while...')
                eval_start_time = time.time()

                log_entry = {'itr':None,'timestamp':None}

                logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ebm_model, init_dist, sampler,
                                                                            train_loader, val_loader, test_loader,
                                                                            preprocess, args.device,
                                                                            args.eval_sampling_steps,
                                                                            args.test_batch_size)
                log_entry['gfn_loss'] = train_loss.item()
                log_entry['ebm_loss'] = loss.item()
                log_entry['Train log-likelihood'] = train_ll.item()
                log_entry['Train log-likelihood'] = val_ll.item()
                log_entry['Test log-likelihood'] = test_ll.item()
                
                for _i, _x in enumerate(ais_samples):
                    plot(f'{args.sample_path}/EBM_sample_{args.dataset_name}_{args.sampler}_{args.step_size}_{itr}_{_i}.png', _x)

                if val_ll.item() > 0:
                    print(f"LL was greater zero at {val_ll.item()}")
                    exit()
                if val_ll.item() > best_val_ll:
                    best_val_ll = val_ll.item()
                    torch.save(ebm_model.state_dict(), f"{args.ckpt_path}/best_ebm_ckpt_{args.dataset_name}_{args.sampler}_{args.step_size}_{itr}.pt")

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')
            
            itr += 1


def main_loop_toy(args, verbose):
    #load data
    db = get_db(args)
    plot = lambda p, x: toy_plot(p, x, args)

    assert args.vocab_size == 2, 'GFlowNet is only specified for binary data'
    assert args.discrete_dim == 32, 'GFlowNet is only specified for 32 dimensions'

    ############# Data
    if not hasattr(args, "int_scale"):
        int_scale = db.int_scale
    else:
        int_scale = args.int_scale
    if not hasattr(args, "plot_size"):
        plot_size = db.f_scale
    else:
        db.f_scale = args.plot_size
        plot_size = args.plot_size
    # plot_size = 4.1
    
    bm = args.bm
    inv_bm = args.inv_bm

    batch_size = args.batch_size
    # multiples = {'pinwheel': 5, '2spirals': 2}                                #not sure what this is for?
    # batch_size = batch_size - batch_size % multiples.get(args.data_name, 1)   #not sure what this is for? 

    ############## EBM model
    samples = get_batch_data(db, args, batch_size=10000)
    eps = 1e-2
    init_mean = torch.from_numpy(np.mean(samples, axis=0) * (1. - 2 * eps) + eps).to(args.device)

    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    ebm_model = EBM(net, init_mean).to(args.device)
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/initial_heat.png', args)
    optimizer = torch.optim.Adam(ebm_model.parameters(), lr=args.lr)

    ############## GFN
    xdim = args.discrete_dim
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, args.device)

    print("model: {:}".format(ebm_model))

    itr = 0
    best_val_ll = np.inf
    best_itr = -1

    start_time = time.time()
    cum_eval_time = 0

    for itr in range(1, args.num_itr + 1):
        st = time.time()

        for _ in range(args.gfn_iter_per_itr):
            x = get_batch_data(db, args)
            x = torch.from_numpy(np.float32(x)).to(args.device)
            update_success_rate = -1.
            gfn.model.train()
            train_loss, train_logZ = gfn.train(batch_size,
                    scorer=lambda inp: -1 * ebm_model(inp).detach(), silent=(itr - 1) % args.print_every != 0, data=x,
                    back_ratio=args.back_ratio)

        for _ in range(args.ebm_iter_per_itr):
            x = get_batch_data(db, args)
            x = torch.from_numpy(np.float32(x)).to(args.device)
            if args.rand_k or args.lin_k or (args.K > 0):
                if args.rand_k:
                    K = random.randrange(xdim) + 1
                elif args.lin_k:
                    K = min(xdim, int(xdim * float(itr) / args.warmup_baf))
                    K = max(K, 1)
                elif args.K > 0:
                    K = args.K
                else:
                    raise ValueError

                gfn.model.eval()
                x_fake, delta_logp_traj = gfn.backforth_sample(x, K)

                delta_logp_traj = delta_logp_traj.detach()
                if args.with_mh:
                    # MH step, calculate log p(x') - log p(x)
                    lp_update = (-1 * ebm_model(x_fake).squeeze()) - (-1 * ebm_model(x).squeeze())
                    update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
                    updates = update_dist.sample()
                    x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
                    update_success_rate = updates.mean().item()
                else:
                    update_success_rate = None

            else:
                x_fake = gfn.sample(batch_size)
                update_success_rate = None
                K = None


            # if itr % args.ebm_every == 0:
            st = time.time() - st

            ebm_model.train()
            logp_real = -1 * ebm_model(x).squeeze()

            logp_fake = -1 * ebm_model(x_fake).squeeze()
            obj = logp_real.mean() - logp_fake.mean()
            l2_reg = (logp_real ** 2.).mean() + (logp_fake ** 2.).mean()
            loss = -obj

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if itr % args.print_every == 0:
            print(f'itr: {itr}, K: {K}, gfn_loss: { train_loss.item()}, ebm_loss: {loss.item()}, success_rate: {update_success_rate}')


        if (itr) % args.eval_every == 0 or (itr) == args.num_itr:
            print('Starting evaluation, this may take a while...')
            eval_start_time = time.time()
                        #log losses
            log_entry = {'itr':None,'timestamp':None}
            log_entry['gfn_loss'] = train_loss.item()
            log_entry['ebm_loss'] = loss.item()

            if itr < args.num_itr:
                ais_samples = args.intermediate_ais_samples
                ais_num_steps = args.intermediate_ais_num_steps
            else: 
                ais_samples =  args.final_ais_samples
                ais_num_steps = args.final_ais_num_steps

            ebm_model.eval()
            gfn.model.eval()

            gen_samples = lambda model, args, batch_size, xt: model.sample(batch_size).cpu().numpy()

            #compute mmds 1/2
            hamming_mmd, bandwidth, euclidean_mmd, sigma = sampler_evaluation(args, db, gfn, gen_samples)

            #log
            log_entry['sampler_hamming_mmd'], log_entry['sampler_bandwidth'] = hamming_mmd, bandwidth
            log_entry['sampler_euclidean_mmd'], log_entry['sampler_sigma'] = euclidean_mmd, sigma

            #compute mmds 2/2
            hamming_mmd, bandwidth, euclidean_mmd, sigma = sampler_ebm_evaluation(args, db, gfn, gen_samples, ebm_model)

            #log
            log_entry['sampler_ebm_hamming_mmd'], log_entry['sampler_ebm_bandwidth'] = hamming_mmd, bandwidth
            log_entry['sampler_ebm_euclidean_mmd'], log_entry['sampler_ebm_sigma'] = euclidean_mmd, sigma

            torch.save(gfn.model.state_dict(), f'{args.ckpt_path}gfn_model_{itr}.pt')
            torch.save(ebm_model.state_dict(), f'{args.ckpt_path}ebm_model_{itr}.pt')

            
            #save success rate
            if not update_success_rate is None:
                log_entry['MH_update_success_rate'] = update_success_rate

            #compute nll, ebm for ebm and log – takes much space
            log_entry['ebm_nll'], log_entry['ebm_hamming_mmd'], log_entry['ebm_bandwidth'], log_entry['ebm_euclidean_mmd'], log_entry['ebm_sigma'] = ebm_evaluation(args, db, ebm_model, batch_size=4000, ais_samples=ais_samples, ais_num_steps=ais_num_steps)
            
            if log_entry['ebm_nll'] < 0:
                print(f"NLL was below zero at {log_entry['ebm_nll']}")
                exit()
            if log_entry['ebm_nll'] < best_val_ll:
                best_val_ll = log_entry['ebm_nll']
                torch.save(ebm_model.state_dict(), f"{args.ckpt_path}/best_ebm_ckpt_{args.dataset_name}_{itr}.pt")

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, itr, timestamp, log_path=f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/eval_log.csv')

        if (itr) % args.itr_save == 0 or (itr) == args.itr_save:
            eval_start_time = time.time()
            
            if args.vocab_size == 2:
                utils.plot_heat(ebm_model, db.f_scale, bm, f'{args.plot_path}/ebm_heat_{itr}.png', args)
                utils.plot_sampler(ebm_model, f'{args.sample_path}/ebm_samples_{itr}.png', args)
                gfn_samples = gfn.sample(2500).detach()
                gfn_samp_float = gfn_samples.data.cpu().numpy().astype(int)
                plot(f'{args.sample_path}/gfn_samples_{itr}.png', torch.tensor(gfn_samp_float).float())
            
            #log losses
            log_entry = {'itr':None,'timestamp':None}
            log_entry['gfn_loss'] = train_loss.item()
            log_entry['ebm_loss'] = loss.item()

            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, itr, timestamp)

