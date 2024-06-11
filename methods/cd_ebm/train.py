import sys
import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.distributions as dists

from utils import utils
from methods.cd_ebm.model import MLPScore, EBM
from utils import sampler



def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

class CustomGibbsSampler:

    def __init__(self, db, args, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            args - Arguments passed when calling the function
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.args = args
        self.db = db
        self.discrete_dim = args.discrete_dim
        self.sample_size = sample_size
        self.max_len = max_len

    def generate_samples(self, model, chains=32, thinning=4, batch_size=128, burn_in=64):
        """
        Function for performing Gibbs sampling on a given model. 
        Inputs:
            model - Neural network to use for modeling E_theta
            steps - Number of iterations in the MCMC algorithm.
        """
        assert self.args.vocab_size == 2, 'gibbs sampling only implemented for binary data'
        assert (batch_size * thinning) % chains  == 0, 'chains must devide batch_size * thinning'

        samples = torch.from_numpy(get_batch_data(self.db, self.args, batch_size=chains))
        B, D = samples.shape

        model.eval()
        
        iterations = burn_in + ((batch_size - 1) * thinning) // chains 
        # Perform Gibbs sampling

        output = None
        for i in range(iterations):

            
            for dim in range(D):
                dim_1, dim_0 = samples.clone(), samples.clone()
                dim_1[:,dim] = 1
                dim_0[:,dim] = 0
                cond_prob_1 = torch.exp(-model(dim_1)) / (torch.exp(-model(dim_1)) + torch.exp(-model(dim_0)))
                dim_x = torch.bernoulli(cond_prob_1)
                samples[:,dim] = dim_x

            if burn_in != 0:
                burn_in -= 1
            elif output is None:
                output = samples.clone() 
            elif (i + 1) % thinning == 0:
                output = torch.cat((output, samples), dim = 0)
        
        return output

def main_loop(db, args, verbose=False):

    #uncomment to use custom gibbs sampler
    #sampler = Sampler(db, args, sample_size=1000)
    
    gibbs_sampler = sampler.GibbsSampler(args.vocab_size, args.discrete_dim, args.device)

    net = MLPScore(args.discrete_dim, [256] * 3 + [1]).to(args.device)
    model = EBM(net).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):

        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)

        for it in pbar:
            data_samples = get_batch_data(db, args)
            data_samples = torch.from_numpy(np.float32(data_samples)).to(args.device)
            #uncomment to use custom gibbs sampler
            #model_samples = sampler.generate_samples(model, batch_size=128).to(args.device)
            model.eval() #double check
            model_samples = gibbs_sampler.forward(model, 50, num_samples=args.batch_size)
            model.train()

            data_nrg = model(data_samples)
            model_nrg = model(model_samples)

            reg_loss = args.cd_alpha * (data_nrg ** 2 + model_nrg ** 2)
            cd_loss = data_nrg - model_nrg
            loss = (reg_loss + cd_loss).logsumexp(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()     

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):

            torch.save(model.state_dict(), f'{args.ckpt_path}/model_{epoch}.pt')

            if args.vocab_size == 2:
                utils.plot_heat(model, db.f_scale, args.bm, f'{args.plot_path}/heat_{epoch}.png', args)
                utils.plot_sampler(model, f'{args.sample_path}/samples_{epoch}.png', args)

 
        




            

