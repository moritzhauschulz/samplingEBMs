import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils import utils
from methods.punidb.model import MLPModel

def gen_samples(model, args):
    """
    sampling using R^* + R_DB
    """
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    t = 0.0
    dt = args.delta_t
    num_samples = 1000
    noise = args.noise
    xt = torch.randint(0, S, (num_samples, D)).to(args.device)

    while t < 1.0:
        t_ = t * torch.ones((num_samples,)).to(args.device)
        with torch.no_grad():
            logits = model(xt, t_)
        x1_probs = F.softmax(logits, dim=-1) # (B, D, S)
        x1_probs_at_xt = torch.gather(x1_probs, -1, xt[:, :, None]) # (B, D, 1)

        # Don't add noise on the final step
        if t + dt < 1.0:
            N = noise
        else:
            N = 0

        # Calculate the off-diagonal step probabilities
        step_probs = (
            dt * ((1 + N + N * (S - 1) * t ) / (1-t)) * x1_probs + 
            dt * N * x1_probs_at_xt
        ).clamp(max=1.0) # (B, D, S)

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

def main_loop(db, args, verbose=False):
    """
    model p(x_1 | x_t)
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

            logits = model(xt, t)
            loss = F.cross_entropy(logits.transpose(1,2), x1, reduction='mean', ignore_index=-1)

            x_hat = torch.argmax(logits, dim=-1)
            acc = (x_hat == x1).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()} Acc {acc}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
        # if True:
            ckpt_path = f'{args.save_dir}/ckpts'
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

            sample_path = f'{args.save_dir}/samples'
            os.makedirs(sample_path, exist_ok=True)
            samples = gen_samples(model, args)
            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{sample_path}/sample_{epoch}.png', im_size=4.1, im_fmt='png')
