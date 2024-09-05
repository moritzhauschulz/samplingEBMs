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
from utils.utils import plot as toy_plot
from utils.utils import get_sampler
from utils.eval import ebm_evaluation
from utils.eval import sampler_evaluation
from utils.eval import sampler_ebm_evaluation
from utils.eval import log
from utils.utils import get_batch_data
from utils.eval import log_completion
from utils.eval import get_eval_timestamp
from utils.eval import make_plots
from utils.toy_data_lib import get_db
import utils.ais as ais

from utils.ais import evaluate_sampler
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

def main_loop_real(args, verbose):

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

    ebm_model = EBM(net, init_mean).to(args.device)
    try:
        d = torch.load(args.pretrained_ebm)
        ebm_model.load_state_dict(d['ema_model'])
        print(f'successfully loaded EBM...')
    except FileNotFoundError as e:
        print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
        sys.exit(1)
    ebm_model.eval()

    #get sampler
    sampler = get_sampler(args) #for eval

    #make gfn model
    xdim = np.prod(args.input_size)
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, args.device)

    cum_eval_time = 0

    batch_size = args.batch_size

    start_time = time.time()
    cum_eval_time = 0

    itr = 1
    while itr <= args.num_itr:
        for _, x in enumerate(train_loader):

            x = preprocess(x[0].to(args.device))  #  -> (bs, 784)
            gfn.model.train()
            B = x.shape[0]
            update_success_rate = -1.
            assert "tb" in args.type
            gfn_loss_start = time.time()
            train_loss, train_logZ = gfn.train(B, scorer=lambda inp: -ebm_model(inp).detach(),
                   silent=itr % args.print_every != 0, data=x, back_ratio=args.back_ratio)
            gfn_loss_end = time.time()

            if itr % args.print_every == 0:
                print(f'Itr: {itr},  gfn_loss: { train_loss.item()}, gfn_time: {gfn_loss_end-gfn_loss_start}')

            if (itr) % args.eval_every == 0 or (itr) == args.num_itr:
                eval_start_time = time.time()
                torch.save(gfn.model.state_dict(), f'{args.ckpt_path}gfn_model_{itr}.pt')
                
                log_entry = {'itr':None,'timestamp':None}
                log_entry['gfn_loss'] = train_loss.item()

                #sampler eval here 
                batches = []
                for i in range(10):
                    gfn_samples = gfn.sample(args.test_batch_size).detach()
                    batches.append(gfn_samples)
                sample_ll = evaluate_sampler(args, ebm_model, batches)
                log_entry['sample_ll'] = sample_ll
                
                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp)

            if (itr) % args.itr_save == 0 or (itr) == args.itr_save:
                eval_start_time = time.time()

                #save model
                torch.save(gfn.model.state_dict(), f'{args.ckpt_path}/gfn_model_{itr}.pt')

                #save gfn samples
                gfn_samples = gfn.sample(100).detach()
                gfn_samp_float = gfn_samples.data.cpu().numpy().astype(int)
                plot(f'{args.sample_path}/gfn_samples_{itr}.png', torch.tensor(gfn_samp_float).float())

                #save log
                log_entry = {'itr':None,'timestamp':None}
                log_entry['gfn_loss'] = train_loss.item()

                timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

                log(args, log_entry, itr, timestamp)

            itr += 1
            

def main_loop_toy(args, verbose):

    #load data
    db = get_db(args)
    plot = lambda p, x: toy_plot(p, x, args)

    assert args.vocab_size == 2, 'GFlowNet is only specified for binary data'
    assert args.discrete_dim == 32, 'GFlowNet is only specified for 32 dimensions'

    ############# Data
    # if not hasattr(args, "int_scale"):
    #     int_scale = db.int_scale
    # else:
    #     int_scale = args.int_scale
    # if not hasattr(args, "plot_size"):
    #     plot_size = db.f_scale
    # else:
    #     db.f_scale = args.plot_size
    #     plot_size = args.plot_size
    # # plot_size = 4.1
    
    bm = args.bm
    inv_bm = args.inv_bm

    batch_size = args.batch_size
    # multiples = {'pinwheel': 5, '2spirals': 2}                                #not sure what this is for?
    # batch_size = batch_size - batch_size % multiples.get(args.data_name, 1)   #not sure what this is for? 

    # make ebm model
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    ebm_model = EBM(net).to(args.device)
    try:
        ebm_model.load_state_dict(torch.load(f'./{args.pretrained_ebm}'))
        print(f'successfully loaded EBM... {args.pretrained_ebm}')
    except FileNotFoundError as e:
        print('Training on EBM requires a trained EBM model. Specify a model checkpoint to load as --pretrained_ebm and try again.')
        sys.exit(1)
    ebm_model.eval()
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.plot_path}/initial_heat.png', args)

    ############## GFN
    xdim = args.discrete_dim
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, args.device)

    print("model: {:}".format(ebm_model))

    itr = 0

    start_time = time.time()
    cum_eval_time = 0

    pbar = tqdm(range(1, args.num_itr + 1)) if verbose else range(1,args.num_itr + 1)
    for itr in pbar:

        x = get_batch_data(db, args)
        x = torch.from_numpy(np.float32(x)).to(args.device)
        gfn.model.train()
        train_loss, train_logZ = gfn.train(batch_size,
                scorer=lambda inp: -1 * ebm_model(inp).detach(), silent=(itr - 1)% args.print_every != 0, data=x,
                back_ratio=args.back_ratio)

        if (itr) % args.eval_every == 0 or (itr) == args.num_itr:
            eval_start_time = time.time()
            log_entry = {'itr':None,'timestamp':None}
            log_entry['gfn_loss'] = train_loss.item()

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
            
            timestamp, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)

            log(args, log_entry, itr, timestamp)



        if (itr) % args.itr_save == 0 or (itr) == args.itr_save:
            eval_start_time = time.time()
            
            if args.vocab_size == 2:
                gfn_samples = gfn.sample(2500).detach()
                gfn_samp_float = gfn_samples.data.cpu().numpy().astype(int)
                plot(f'{args.sample_path}/gfn_samples_{itr}.png', torch.tensor(gfn_samp_float).float())
            
            #log losses
            log_entry = {'itr':None,'timestamp':None}
            log_entry['gfn_loss'] = train_loss.item()

            _, cum_eval_time = get_eval_timestamp(eval_start_time, cum_eval_time, start_time)
