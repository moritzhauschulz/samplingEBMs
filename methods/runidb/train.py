import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils import utils
from methods.runidb.model import MLPModel

def gen_samples(model, args):
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    t = 0.0
    dt = args.delta_t
    num_samples = 1000
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

def get_batch_data(db, args):
    bx = db.gen_batch(args.batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def compute_loss_casted(model, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size

    R_star = torch.zeros((B, D, S)).to(args.device)
    R_star = R_star.scatter_(-1, x1[:, :, None], 1.0) / (1 - t[:, None, None])
    R_star[xt == x1] = 0.0
    R_star = R_star.clamp(max=1.0)
    R_star.scatter_(-1, xt[:, :, None], 0.0)
    R_star.scatter_(-1, xt[:, :, None], (0.0 - R_star.sum(dim=-1, keepdim=True))) 
    
    eta = 1.0
    R_DB_1 = torch.zeros((B, D, S)).to(args.device)
    R_DB_1[xt == x1] = 1 * eta
    R_DB_2 = torch.zeros((B, D, S)).to(args.device)
    R_DB_2 = R_DB_2.scatter_(-1, x1[:, :, None], 1.0) * eta * ((S*t + 1 - t) / (1-t))[:, None, None]
    R_DB = R_DB_1 + R_DB_2
    R_DB = R_DB.clamp(max=1.0)
    R_DB.scatter_(-1, xt[:, :, None], 0.0)
    R_DB.scatter_(-1, xt[:, :, None], (0.0 - R_DB.sum(dim=-1, keepdim=True)))

    R_true = R_star + R_DB
    R_true = R_true.clamp(max=1.0)
    R_true.scatter_(-1, xt[:, :, None], 0.0)
    R_true.scatter_(-1, xt[:, :, None], (0.0 - R_true.sum(dim=-1, keepdim=True)))
    
    R_true = R_true.detach()
    R_est = model(xt, t)
    loss = (R_est - R_true).square().sum(dim=(1,2)).mean(dim=0)
    
    return loss

def compute_loss(model, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size

    R_star = torch.zeros((B, D, S)).to(args.device)
    R_star = R_star.scatter_(-1, x1[:, :, None], 1.0) / (1 - t[:, None, None])
    R_star[xt == x1] = 0.0
    
    #see F2.1. in appendix of Discrete Flow Models
    eta = 1.0
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

def main_loop(db, args, verbose=False):
    """
    model Rate matrix
    """
    model = MLPModel(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)

        for it in pbar:
            x1 = get_batch_data(db, args)
            x1 = torch.from_numpy(x1)

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

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            torch.save(model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')

            samples = gen_samples(model, args)
            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{args.sample_path}/sample_{epoch}.png', im_size=4.1, im_fmt='png')
