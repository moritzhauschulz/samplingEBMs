import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy.combinatorics.graycode import GrayCode
import json
import gc
import pandas as pd
import matplotlib.pyplot as plt

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

def make_plots(log_path):

  # Read the CSV file
  df = pd.read_csv(log_path)

  # Extract metrics dynamically from the first line (excluding 'epoch' and 'timestamp')
  metrics = df.columns[2:]  # Exclude 'epoch' and 'timestamp'

  # Define the main metric for the second y-axis
  main_metric = 'ebm_nll'

  # Plot metrics over epochs and save as PNG
  fig, ax1 = plt.subplots(figsize=(12, 6))

  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Value')
  ax1.set_title('Metrics over Epochs')

  # Plot other metrics on the left y-axis
  for metric in metrics:
      if metric != main_metric:
          ax1.plot(df['epoch'], df[metric], label=metric)

  ax1.grid(True)
  ax1.legend(loc='upper left')

  # Create a second y-axis for the main metric
  ax2 = ax1.twinx()
  ax2.set_ylabel(main_metric)
  ax2.plot(df['epoch'], df[main_metric], color='tab:red', label=main_metric)
  ax2.legend(loc='upper right')

  # Save the plot as a PNG file
  plt.savefig('metrics_over_epochs.png')
  plt.show()

  # Plot metrics over time and save as PNG
  fig, ax1 = plt.subplots(figsize=(12, 6))

  ax1.set_xlabel('Timestamp')
  ax1.set_ylabel('Value')
  ax1.set_title('Metrics over Time')

  # Plot other metrics on the left y-axis
  for metric in metrics:
      if metric != main_metric:
          ax1.plot(df['timestamp'], df[metric], label=metric)

  ax1.grid(True)
  ax1.legend(loc='upper left')

  # Create a second y-axis for the main metric
  ax2 = ax1.twinx()
  ax2.set_ylabel(main_metric)
  ax2.plot(df['timestamp'], df[main_metric], color='tab:red', label=main_metric)
  ax2.legend(loc='upper right')

  # Save the plot as a PNG file
  plt.savefig('metrics_over_time.png')
  plt.show()

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


def ebm_evaluation(args, db, ebm, write_to_index=True, batch_size=4000, ais_samples=1000000, num_ais_mcmc_steps=25, ais_num_steps=1000):
  print_cuda_memory_stats()
  #NLL
  log_Z = annealed_importance_sampling(args, ebm, ais_samples, ais_num_steps, num_ais_mcmc_steps, args.discrete_dim)
  nll_samples = get_batch_data(db, args, batch_size=batch_size)
  nll_samples = torch.from_numpy(np.float32(nll_samples)).to(args.device)
  nll = ebm(nll_samples) - log_Z
  nll = torch.sum(nll) / batch_size

  #MDD
  mmd_list = []
  for _ in range(10):
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
    x = gibbs_sampler(ebm, num_rounds=100, num_samples=batch_size).to('cpu')
    y = get_batch_data(db, args, batch_size=batch_size)
    y = torch.from_numpy(np.float32(y)).to('cpu')
    mmd_list.append(exp_hamming_mmd(x,y))
  mmd = sum(mmd_list)/10

  print(f'NLL on {batch_size} samples with AIS on {ais_samples} samlpes: {nll}; Final exponential Hamming MMD on 10x{batch_size} samples: {mmd}')
  
  return nll.item(), mmd.item()


def sampler_evaluation(args, db, sampler_function, write_to_index=True, batch_size=4000):
  #note: there is no immedaite way to obtain NLL for DFS – this would require sampling a backward trajectory...
  print_cuda_memory_stats()
  #MDD
  mmd_list = []
  for _ in range(10):
    x = sampler_function(batch_size).to('cpu')
    y = get_batch_data(db, args, batch_size=batch_size)
    y = torch.from_numpy(np.float32(y)).to('cpu')
    mmd_list.append(exp_hamming_mmd(x,y))
  mmd = sum(mmd_list)/10

  print(f'Exponential Hamming MMD of SAMPLER against data on 10x{batch_size} samples: {mmd}')
  
  return mmd.item()

def sampler_ebm_evaluation(args, db, sampler_function, ebm, write_to_index=True, batch_size=4000):
  #note: there is no immedaite way to obtain NLL for DFS – this would require sampling a backward trajectory...
  print_cuda_memory_stats()
  #MDD
  mmd_list = []
  for _ in range(10):
    x = sampler_function(batch_size).to('cpu')
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
    y = gibbs_sampler(ebm, num_rounds=100, num_samples=batch_size).to('cpu')
    mmd_list.append(exp_hamming_mmd(x,y))
  mmd = sum(mmd_list)/10

  print(f'Exponential Hamming MMD of SAMPLER against EBM on 10x{batch_size} samples: {mmd}')
  
  return mmd.item()


def log(args, log_entry, epoch, timestamp):
  log_entry['epoch'] = epoch
  log_entry['timestamp'] = timestamp
  df_log_entry = pd.DataFrame([log_entry])
  df_log_entry.to_csv(args.log_path, mode='a', header=not os.path.exists(args.log_path), index=False)
  print(f'logged epoch {epoch} to log file')

def log_completion(method, data, args):
  if os.path.exists(args.index_path) and os.path.getsize(args.index_path) > 0:
    try:
        with open(args.index_path, 'r') as file:
            experiment_idx = json.load(file)
    except json.JSONDecodeError:
        print("Warning: JSON file is corrupted. Cannot log completion.")
    experiment_idx[args.exp_n]['completed'] = True
    with open(args.index_path, 'w') as file:
      json.dump(experiment_idx, file, indent=4)
    print(f'Completion logged.')


def convert_namespace_to_dict(args):
    args_dict = vars(args).copy()  # Convert Namespace to dictionary
    # Handle non-serializable objects
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)
    args_dict.pop('bm', None)
    args_dict.pop('inv_bm', None)
    return args_dict

def log_args(method, data, args):
  if os.path.exists(args.index_path) and os.path.getsize(args.index_path) > 0:
      try:
          with open(args.index_path, 'r') as file:
              experiment_idx = json.load(file)
      except json.JSONDecodeError:
          print("Warning: JSON file is corrupted. Initializing a new experiment index.")
          experiment_idx = {}
  else:
      experiment_idx = {}


  experiment_number = 0
  # if not method in experiment_idx.keys():
  #   experiment_idx[method] = {}
  # if not data in experiment_idx[method].keys():
  #   experiment_idx[method][data] = {}
  while True:
    if str(experiment_number) in experiment_idx.keys():
        experiment_number += 1
    else:
        break
  args.exp_n = experiment_number
  args.exp_id = f'{method}_{data}_{experiment_number}'
  experiment_idx[args.exp_n] = convert_namespace_to_dict(args)

  with open(args.index_path, 'w') as file:
      json.dump(experiment_idx, file, indent=4)

  print(f"Experiment meta data saved to {args.index_path}")

