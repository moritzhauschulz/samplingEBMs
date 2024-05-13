import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

from utils import utils
from methods.ed_ebm.model import MLPScore, EBM
from methods.ed_ebm.ed_categorical import ed_categorical

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def energy_discrepancy_bernoulli(energy_net, samples, epsilon=0.1, m_particles=32, w_stable=1.):
    device = samples.device
    bs, dim = samples.shape

    noise_dist = dists.Bernoulli(probs=epsilon * torch.ones((dim,)).to(device))
    beri = (noise_dist.sample((bs,)) + samples) % 2.    # [bs, dim]
    pert_data = (noise_dist.sample((bs * m_particles,)).view(bs, m_particles, dim) + beri.unsqueeze(1)) % 2.    # [bs, m_particles, dim]
    # print((samples.unsqueeze(1) - pert_data).abs().mean())

    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(pert_data.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss

def main_loop(db, args, verbose=False):

    #assert args.vocab_size == 2, 'Only support binary data'

    ckpt_path = f'{os.path.abspath(os.path.join(os.path.dirname(__file__)))}/ckpts/{args.data_name}/'
    plot_path = f'{os.path.abspath(os.path.join(os.path.dirname(__file__)))}/plots/{args.data_name}/'
    if os.path.exists(ckpt_path):
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.makedirs(plot_path, exist_ok=True)

    samples = get_batch_data(db, args, batch_size=50000)
    mean = np.mean(samples, axis=0)
    #init_dist = torch.distributions.Bernoulli(probs=torch.from_numpy(mean).to(args.device) * (1. - 2 * 1e-2) + 1e-2)
    
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    #model = EBM(net, torch.from_numpy(mean)).to(args.device)
    model = EBM(net).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)

        for it in pbar:
            samples = get_batch_data(db, args)
            samples = torch.from_numpy(np.float32(samples)).to(args.device)

            if args.vocab_size == 2:
                loss = energy_discrepancy_bernoulli(model, samples) #note that this is not exactly contrastive divergence
            else:
                loss = ed_categorical(model, samples, K=args.vocab_size)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
        # if True:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

            if args.vocab_size == 2:
                utils.plot_heat(model, db.f_scale, args.bm, f'{plot_path}/heat_{epoch}.pdf', args)
                utils.plot_sampler(model, f'{plot_path}/samples_{epoch}.png', args)

