import os
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions.categorical import Categorical

from utils import utils
from methods.ed_ebm.model import MLPScore, EBM
from methods.ebm_runidb.model import MLPModel as MLPFlow


def gen_samples(model, args):
    model.eval()
    S, D = args.vocab_size, args.discrete_dim

    t = 0.0
    dt = 0.01
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
        # print(step_probs)

        xt = Categorical(step_probs).sample() # (B, D)

        t += dt

    return xt.detach().cpu().numpy()

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def compute_loss(ebm_model, flow_model, q_dist, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size

    R_star = torch.zeros((B, D, S)).to(args.device)
    R_star = R_star.scatter_(-1, x1[:, :, None], 1.0) / (1 - t[:, None, None])
    R_star[xt == x1] = 0.0

    eta = 1.0
    R_DB_1 = torch.zeros((B, D, S)).to(args.device)
    R_DB_1[xt == x1] = 1 * eta
    R_DB_2 = torch.zeros((B, D, S)).to(args.device)
    R_DB_2 = R_DB_2.scatter_(-1, x1[:, :, None], 1.0) * eta * ((S*t + 1 - t) / (1-t))[:, None, None]
    R_DB = R_DB_1 + R_DB_2

    R_true = (R_star + R_DB) * (1 - t[:, None, None])
    R_est = flow_model(xt, t) * (1 - t[:, None, None])
    loss = (R_est - R_true).square()
    loss.scatter_(-1, xt[:, :, None], 0.0)
    loss = loss.sum(dim=(1,2))

    energy = torch.exp(-ebm_model(x1.float()))
    q_density = q_dist.log_prob(x1.float()).sum(dim=-1).exp()
    loss = (energy / q_density * loss).mean(dim=0)
    
    return loss

def main_loop(db, args, verbose=False):
    assert args.vocab_size == 2, 'Only support binary data'

    samples = get_batch_data(db, args, batch_size=50000)
    mean = np.mean(samples, axis=0)
    q_dist = torch.distributions.Bernoulli(probs=torch.from_numpy(mean).to(args.device) * (1. - 2 * 1e-2) + 1e-2)

    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    ebm_model = EBM(net, torch.from_numpy(mean)).to(args.device)
    ebm_model.load_state_dict(torch.load(f'methods/ed_ebm/ckpts/{args.data_name}/model_999.pt', map_location=args.device))
    ebm_model.eval()
    utils.plot_heat(ebm_model, db.f_scale, args.bm, f'{args.save_dir}/heat.pdf', args)

    flow_model = MLPFlow(args).to(args.device)
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        flow_model.train()
        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)
        
        for it in pbar:
            # x1_q = q_dist.sample((args.batch_size,)).long()
            # x1_p = torch.from_numpy(get_batch_data(db, args)).to(args.device)
            # mask = (torch.rand((args.batch_size,)).to(args.device) < 0.5).int().unsqueeze(1)
            # x1 = mask * x1_q + (1 - mask) * x1_p

            x1 = q_dist.sample((args.batch_size,)).long() #remember that there is no data available under the assumptions

            (B, D), S = x1.size(), args.vocab_size
            t = torch.rand((B,)).to(args.device)
            xt = x1.clone()
            uniform_noise = torch.randint(0, S, (B, D)).to(args.device)
            corrupt_mask = torch.rand((B, D)).to(args.device) < (1 - t[:, None])
            xt[corrupt_mask] = uniform_noise[corrupt_mask]
            
            loss = compute_loss(ebm_model, flow_model, q_dist, xt, x1, t, args) #basically fit flow_model to ebm model
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
        # if True:
            ckpt_path = f'{args.save_dir}/ckpts'
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(flow_model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

            sample_path = f'{args.save_dir}/samples'
            os.makedirs(sample_path, exist_ok=True)
            samples = gen_samples(flow_model, args)
            if args.vocab_size == 2:
                float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
            else:
                float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
            utils.plot_samples(float_samples, f'{sample_path}/sample_{epoch}.png', im_size=4.1, im_fmt='png')


