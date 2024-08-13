import os
import copy
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import utils.vamp_utils as vamp_utils
from utils.eval import log
from utils.eval import log_completion

from velo_uni_dfm.model import ResNetFlow

def gen_samples(model, args, batch_size=None, t=0.0, xt=None):
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    # Variables, B, D for batch size and number of dimensions respectively
    B = batch_size if batch_size is not None else args.batch_size

    # Initialize xt with the mask index value if not provided
    if xt is None:
        xt = torch.randint(0, S, (B, D)).to(args.device)
    
    # forced_unmask = torch.zeros((B,)).to(args.device)

    dt = args.delta_t  # Time step
    t = 0.0  # Initial time

    while t < 1.0:
        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            x1_logits = F.softmax(model(xt, t_), dim=-1)
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 
        ut = 1/(1-t) * (x1_logits - delta_xt)

        step_probs = delta_xt + (ut * dt)

        if args.impute_self_connections:
            step_probs = step_probs.clamp(max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)) 

        t += dt

        xt = Categorical(step_probs).sample() #(B,D)

    return xt.detach().cpu().numpy() #, forced_unmask


def compute_loss(model, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size

    x1_logits = model(xt, t)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='mean')

    x_hat = torch.argmax(x1_logits, dim=-1)
    acc = (x_hat == x1).float().mean().item()

    return loss, acc

def main_loop(args, verbose=False):
    my_print = args.my_print

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
    model = ResNetFlow(64, args)
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # move to cuda
    model.to(args.device)
    ema_model.to(args.device)

    start_time = time.time()
    cum_eval_time = 0

    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(train_loader) if verbose else train_loader

        for it, (x, _) in enumerate(pbar):
            
            
            
            x1 = preprocess(x).long().to(args.device)
            
            (B, D), S = x1.size(), args.vocab_size
            x0 = torch.randint(0, S, (B, D)).to(args.device)

            t = torch.rand((B,)).to(args.device)
            xt = x1.clone()
            mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
            xt[mask] = x0[mask]

            loss, acc = compute_loss(model, xt, x1, t, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}, Acc {acc}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            eval_start_time = time.time()

            torch.save(model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')

            log_entry = {'epoch':None,'timestamp':None}
            
            log_entry['loss'] = loss.item()
            log_entry['acc'] = acc
            torch.save(ema_model.state_dict(), f'{args.ckpt_path}/ema_model_{epoch}.pt')

            samples = gen_samples(model, args)
            plot(f'{args.sample_path}/samples_{epoch}.png', torch.tensor(samples).float())
            ema_samples = gen_samples(ema_model, args)
            plot(f'{args.sample_path}/ema_samples_{epoch}.png', torch.tensor(ema_samples).float())
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            cum_eval_time += eval_time
            timestamp = time.time() - cum_eval_time - start_time

            log(args, log_entry, epoch, timestamp)
    log_completion(args.methods, args.dataset_name, args)

