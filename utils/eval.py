import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy.combinatorics.graycode import GrayCode
import json
import gc

from utils.utils import get_batch_data
from utils.utils import print_cuda_memory_stats

from utils.sampler import GibbsSampler 

def energy_source(x):
    return -torch.sum(torch.log((1 - 0.5) ** x * 0.5 ** (1-x)), dim=-1)

def ais_mcmc_step(args, x, energy_fn, step_size=0.1):
    gc.collect()
    torch.cuda.empty_cache()
    x_new = torch.bernoulli(torch.full_like(x, 0.5))
    gc.collect()
    torch.cuda.empty_cache()
    # print_cuda_memory_stats()
    with torch.no_grad():
      energy_old = energy_fn(x).detach().cpu()
    gc.collect()
    torch.cuda.empty_cache()
    # print_cuda_memory_stats()
    with torch.no_grad():
      energy_new =  energy_fn(x_new).detach().cpu()
    gc.collect()
    torch.cuda.empty_cache()
    # print_cuda_memory_stats()
    accept_prob = torch.exp(energy_old - energy_new)
    accept = torch.rand(x.shape[0]) < accept_prob
    x[accept] = x_new[accept]
    return x

def annealed_importance_sampling(args, score_fn, num_samples, num_intermediate, num_mcmc_steps, latent_dim):
    gc.collect()
    torch.cuda.empty_cache()
    x = torch.bernoulli(torch.full((num_samples, latent_dim), 0.5)).to(args.device)  # Initial samples from the source distribution
    betas = np.linspace(0, 1, num_intermediate)
    
    log_weights = torch.zeros(num_samples).cpu()

    pbar = tqdm(range(num_intermediate - 1))
    
    for i in pbar:
      beta0, beta1 = betas[i], betas[i + 1]
      
      def energy_fn(x):
          gc.collect()
          torch.cuda.empty_cache()
          with torch.no_grad():
            energy = (1 - beta0) * energy_source(x) + beta0 * score_fn(x)
          return energy
      
      for _ in range(num_mcmc_steps):
          gc.collect()
          torch.cuda.empty_cache()
          x = ais_mcmc_step(args, x, energy_fn)
      
      with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        log_weights += (beta1 - beta0) * (score_fn(x).detach().cpu() - energy_source(x).detach().cpu())

      pbar.set_description(f'AIS iteration {i}')
    
    max_log_weight = torch.max(log_weights)
    weights = torch.exp(log_weights - max_log_weight)
    normalized_weights = weights / torch.sum(weights)
    
    log_partition_ratio = max_log_weight + torch.log(torch.sum(weights)) - torch.log(torch.tensor(num_samples, dtype=torch.float32))
    
    return log_partition_ratio

def exp_hamming_sim(x, y, bd):
  torch.cuda.empty_cache()
  x = x.unsqueeze(1)
  y = y.unsqueeze(0)
  d = torch.sum(torch.abs(x - y), dim=-1)
  return torch.exp(-bd * d)

def exp_hamming_mmd(x, y, bandwidth=0.1):
  torch.cuda.empty_cache()
  x = x.float()
  y = y.float()

  with torch.no_grad():
      kxx = exp_hamming_sim(x, x, bd=bandwidth)
      idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
      kxx[idx, idx] = 0.0
      kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

      kyy = exp_hamming_sim(y, y, bd=bandwidth)
      idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
      kyy[idx, idx] = 0.0
      kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)

      kxy = torch.sum(exp_hamming_sim(x, y, bd=bandwidth)) / x.shape[0] / y.shape[0]

      mmd = kxx + kyy - 2 * kxy
  return mmd


def ebm_evaluation(args, db, ebm, write_to_index=True, batch_size=4000, ais_samples=1000000, num_ais_mcmc_steps=25, ais_num_intermediate=1000):
  #NLL
  Z = torch.exp(annealed_importance_sampling(args, ebm, ais_samples, ais_num_intermediate, num_ais_mcmc_steps, args.discrete_dim))
  nll_samples = get_batch_data(db, args, batch_size=batch_size)
  nll_samples = torch.from_numpy(np.float32(nll_samples)).to(args.device)
  nll = ebm(nll_samples)
  nll = torch.sum(nll / Z)

  #MDD
  mmd_list = []
  for _ in range(10):
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
    x = gibbs_sampler(ebm, num_rounds=100, num_samples=batch_size).to('cpu')
    y = get_batch_data(db, args, batch_size=batch_size)
    y = torch.from_numpy(np.float32(y)).to('cpu')
    mmd_list.append(exp_hamming_mmd(x,y))
  mmd = sum(mmd_list)/10

  print(f'Final NLL on {batch_size} samples with AIS on {ais_samples} samlpes: {nll}; Final exponential Hamming MMD on 10x{batch_size} samples: {mmd}')
  
  if write_to_index:
    experiment_idx_path = f'{args.save_dir}/experiment_idx.json'
    if os.path.exists(experiment_idx_path) and os.path.getsize(experiment_idx_path) > 0:
      try:
          with open(experiment_idx_path, 'r') as file:
              experiment_idx = json.load(file)
      except json.JSONDecodeError:
          print("Warning: JSON file is corrupted. Cannot store evaluation results.")
      experiment_idx[str(args.idx)]['EBM_nll'] = nll.item()
      experiment_idx[str(args.idx)]['EBM_mmd'] = mmd.item()
      with open(experiment_idx_path, 'w') as file:
        json.dump(experiment_idx, file, indent=4)
      print(f'Evaluation results written to index file: {experiment_idx_path}')
    else:
      print('Could not write evaluation to index because index file was not found.')
  
  return nll, mmd


def sampler_evaluation(args, db, sampler_function, write_to_index=True, batch_size=4000):
  #note: there is no immedaite way to obtain NLL for DFS – this would require sampling a backward trajectory...

  #MDD
  mmd_list = []
  for _ in range(10):
    x = sampler_function(batch_size).to('cpu')
    y = get_batch_data(db, args, batch_size=batch_size)
    y = torch.from_numpy(np.float32(y)).to('cpu')
    mmd_list.append(exp_hamming_mmd(x,y))
  mmd = sum(mmd_list)/10

  print(f'Final exponential Hamming MMD of SAMPLER against data on 10x{batch_size} samples: {mmd}')
  
  if write_to_index:
    experiment_idx_path = f'{args.save_dir}/experiment_idx.json'
    if os.path.exists(experiment_idx_path) and os.path.getsize(experiment_idx_path) > 0:
      try:
          with open(experiment_idx_path, 'r') as file:
              experiment_idx = json.load(file)
      except json.JSONDecodeError:
          print("Warning: JSON file is corrupted. Cannot store evaluation results.")
      experiment_idx[str(args.idx)]['sampler_mmd'] = mmd.item()
      with open(experiment_idx_path, 'w') as file:
        json.dump(experiment_idx, file, indent=4)
      print(f'Evaluation results written to index file: {experiment_idx_path}')
    else:
      print('Could not write evaluation to index because index file was not found.')
  
  return mmd

def sampler_ebm_evaluation(args, db, sampler_function, ebm, write_to_index=True, batch_size=4000):
  #note: there is no immedaite way to obtain NLL for DFS – this would require sampling a backward trajectory...

  #MDD
  mmd_list = []
  for _ in range(10):
    x = sampler_function(batch_size).to('cpu')
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
    y = gibbs_sampler(ebm, num_rounds=100, num_samples=batch_size).to('cpu')
    mmd_list.append(exp_hamming_mmd(x,y))
  mmd = sum(mmd_list)/10

  print(f'Final exponential Hamming MMD of SAMPLER against EBM on 10x{batch_size} samples: {mmd}')
  
  if write_to_index:
    experiment_idx_path = f'{args.save_dir}/experiment_idx.json'
    if os.path.exists(experiment_idx_path) and os.path.getsize(experiment_idx_path) > 0:
      try:
          with open(experiment_idx_path, 'r') as file:
              experiment_idx = json.load(file)
      except json.JSONDecodeError:
          print("Warning: JSON file is corrupted. Cannot store evaluation results.")
      experiment_idx[str(args.idx)]['sampler_EBM_mmd'] = mmd.item()
      with open(experiment_idx_path, 'w') as file:
        json.dump(experiment_idx, file, indent=4)
      print(f'Evaluation results written to index file: {experiment_idx_path}')
    else:
      print('Could not write evaluation to index because index file was not found.')
  
  return mmd


