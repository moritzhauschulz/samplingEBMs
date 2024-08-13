import os
import copy
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.eval import log
from utils.eval import log_completion

import utils.vamp_utils as vamp_utils
from punidb.model import ResNetFlow

def gen_samples(model, args, batch_size=None, t=0.0, xt=None):
    """
    sampling using R^* + R_DB
    """
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    t = 0.0
    dt = args.delta_t
    #num_samples = 100
    B = batch_size if batch_size is not None else args.batch_size
    xt = torch.randint(0, S, (B, D)).to(args.device)

    while t < 1.0:
        # t_ = t * torch.ones((B,)).to(args.device)
        # with torch.no_grad():
        #     logits = model(xt, t_)
        # x1_probs = F.softmax(logits, dim=-1) # (B, D, S)
        # x1_probs_at_xt = torch.gather(x1_probs, -1, xt[:, :, None]) # (B, D, 1)

        # # Don't add noise on the final step
        # if t + dt < 1.0:
        #     N = 0 #args.eta #commented out to ensure zero eta
        # else:
        #     N = 0

        # # Calculate the off-diagonal step probabilities
        # step_probs = (
        #     dt * ((1 + N + N * (S - 1) * t ) / (1-t)) * x1_probs + 
        #     dt * N * x1_probs_at_xt
        # ).clamp(max=1.0) # (B, D, S)

        # # Calculate the on-diagnoal step probabilities
        # # 1) Zero out the diagonal entries
        # step_probs.scatter_(-1, xt[:, :, None], 0.0)
        # # 2) Calculate the diagonal entries such that the probability row sums to 1
        # step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)) 

        # xt = Categorical(step_probs).sample() # (B, D)

        # t += dt
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

    return xt.detach().cpu().numpy()

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

    cum_eval_time = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(train_loader) if verbose else train_loader

        for it, (x, _) in enumerate(pbar):
            x1 = preprocess(x).long()
            
            (B, D), S = x1.size(), args.vocab_size
            t = torch.rand((B,))
            xt = x1.clone()
            uniform_noise = torch.randint(0, S, (B, D))
            corrupt_mask = torch.rand((B, D)) < (1 - t[:, None])
            xt[corrupt_mask] = uniform_noise[corrupt_mask]

            x1 = x1.to(args.device)
            xt = xt.to(args.device)
            t = t.to(args.device)

            logits = model(xt, t)
            loss = F.cross_entropy(logits.transpose(1,2), x1, reduction='mean', ignore_index=-1)

            x_hat = torch.argmax(logits, dim=-1)
            acc = (x_hat == x1).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()} Acc {acc}')

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
