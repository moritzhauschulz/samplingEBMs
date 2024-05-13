import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

from utils import utils
from methods.cd_ebm.model import MLPScore, EBM

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

class Sampler:

    def __init__(self, model, db, args, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            args - Arguments passed when calling the function
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.db = db
        self.discrete_dim = args.discrete_dim
        self.sample_size = sample_size
        self.max_len = max_len

    #@staticmethod
    def generate_samples(self, model, steps=60):
        """
        Function for performing Gibbs sampling on a given model. 
        Inputs:
            model - Neural network to use for modeling E_theta
            steps - Number of iterations in the MCMC algorithm.
        """
        assert self.args.vocab_size == 2, 'gibbs sampling only implemented for binary data'
        samples = get_batch_data(self.db, self.args, batch_size=500)
        B, D = samples.size()

        model.eval()
        
        # Perform Gibbs sampling
        for _ in range(steps):
            
            for dim in range(D):
                dim_1, dim_0 = samples.clone(), samples.clone()
                dim_1[:,dim] = 1
                dim_0[:,dim] = 0
                cond_prob_1 = torch.exp(-model(dim_1)) / (torch.exp(-model(dim_1)) + torch.exp(-model(dim_0)))
                dim_x = torch.bernoulli(cond_prob_1)
                samples[:,dim] = dim_x

        return samples

def main_loop(db, args, verbose=False):

    ckpt_path = f'{os.path.abspath(os.path.join(os.path.dirname(__file__)))}/ckpts/{args.data_name}/'
    plot_path = f'{os.path.abspath(os.path.join(os.path.dirname(__file__)))}/plots/{args.data_name}/'
    if os.path.exists(ckpt_path):
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.makedirs(plot_path, exist_ok=True)

    samples = get_batch_data(db, args, batch_size=50000)
    mean = np.mean(samples, axs=0)
    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    model = EBM(net).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)

        for it in pbar:
            data_samples = get_batch_data(db, args)
            data_samples = torch.from_numpy(np.float32(samples)).to(args.device)


            
