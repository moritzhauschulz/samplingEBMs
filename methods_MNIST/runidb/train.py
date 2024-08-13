import os
import copy
import torch
import numpy as np
import torchvision
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import utils.vamp_utils as vamp_utils
from runidb.model import ResNetFlow

def gen_samples(model, args):
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    t = 0.0
    dt = 0.001
    num_samples = 100
    xt = torch.randint(0, S, (num_samples, D)).to(args.device)

    while t < 1.0:
        t_ = t * torch.ones((num_samples,)).to(args.device)
        with torch.no_grad():
            step_probs = model(xt, t_) * dt

        step_probs = step_probs.clamp(max=1.0)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)) 

        xt = Categorical(step_probs).sample() # (B, D)

        t += dt

    return xt.detach().cpu().numpy()

def compute_loss(model, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size

    R_star = torch.zeros((B, D, S)).to(args.device)
    R_star = R_star.scatter_(-1, x1[:, :, None], 1.0) / (1 - t[:, None, None])
    R_star[xt == x1] = 0.0
    
    eta = args.eta
    R_DB_1 = torch.zeros((B, D, S)).to(args.device)
    R_DB_1[xt == x1] = 1 * eta
    R_DB_2 = torch.zeros((B, D, S)).to(args.device)
    R_DB_2 = R_DB_2.scatter_(-1, x1[:, :, None], 1.0) * eta * ((S*t + 1 - t) / (1-t))[:, None, None]
    R_DB = R_DB_1 + R_DB_2

    R_true = (R_star + R_DB) * (1 - t[:, None, None])
    R_est = model(xt, t) * (1 - t[:, None, None])
    loss = (R_est - R_true).square()
    loss.scatter_(-1, xt[:, :, None], 0.0)
    loss = loss.sum(dim=(1,2)).mean(dim=0)
    
    return loss

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

            loss = compute_loss(model, xt, x1, t, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')
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