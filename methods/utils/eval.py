import torch
import os
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import gc
import pandas as pd
import matplotlib.pyplot as plt
import sys
from filelock import FileLock, Timeout
import re
import time

from utils.utils import get_batch_data
from utils.utils import print_cuda_memory_stats
from utils.utils import bin2float
from utils.utils import ourbase2float

from utils.sampler import GibbsSampler 

def plot_weight_histogram(weights_tensor, output_dir=None, bins=100, title='Histogram of Weights'):
    """
    Plots a histogram of the weights in the given tensor.

    Args:
        weights_tensor (torch.Tensor): A tensor containing weights between 0 and 1.
        bins (int): The number of bins to use for the histogram.
        title (str): The title of the histogram plot.
    """
    if output_dir is None:
      output_dir = './example_weight_histogram.png'

    # Ensure the tensor is on the CPU and convert it to a NumPy array
    weights_np = weights_tensor.cpu().detach().numpy()

    # Plot the histogram
    plt.hist(weights_np, bins=bins, range=None, alpha=0.75, color='blue', edgecolor='black')

    # Set the title and labels
    plt.title(title)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

    # save the plot
    plt.savefig(output_dir)
    plt.clf()

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

def make_plots(log_path, output_dir='', hamming_mmd_reference_value=None, euclidean_mmd_reference_value=None, nll_reference_value=None, last_n=5, mmd_lower_y_lim=None, mmd_upper_y_lim=None, nll_lower_y_lim=None, nlll_upper_y_lim=None):

    folder_path = os.path.dirname(log_path)

    if output_dir == '':
        epochs_output = folder_path + '/metrics_over_epochs.png'
        time_output = folder_path + '/metrics_over_time.png'
        loss_output = folder_path + '/loss_over_epochs.png'
    else: 
        epochs_output = output_dir + '/metrics_over_epochs.png'
        time_output = output_dir + '/metrics_over_time.png'
        loss_output = output_dir + '/loss_over_epochs.png'

    # Read the CSV file
    df = pd.read_csv(log_path)
    df_noloss = df.drop(['bandwidth', 'sigma','loss'], axis=1)

    # Extract metrics dynamically from the first line (excluding 'epoch' and 'timestamp')
    metrics = df_noloss.columns[2:]  # Exclude 'epoch' and 'timestamp'

    def extract_numeric_value(tensor_str):
      # Use regular expression to find the numeric value
      match = re.search(r'tensor\(([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*),', tensor_str)
      if match:
          return float(match.group(1))
      return None

    for metric in metrics:
      df_noloss[metric] = df_noloss[metric].apply(lambda x: extract_numeric_value(x) if isinstance(x, str) else x)

    def add_averages_text(ax, df_noloss, metrics):
        avg_text = f"Average of last {last_n} datapoints\n"
        for metric in metrics:
            avg_value = df_noloss[metric].tail(last_n).mean()
            avg_text += f"{metric}: {avg_value:.5f}\n"
        
        # Add text box to the plot
        ax.text(0.95, 0.95, avg_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5))

    # Plot metrics over epochs and save as PNG
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Value')
    ax1.set_title('Metrics over Epochs')
    if mmd_upper_y_lim is not None and mmd_lower_y_lim is not None:
      ax1.set_ylim(mmd_lower_y_lim, mmd_upper_y_lim)

    # Plot other metrics on the left y-axis
    for metric in metrics:
        if not metric in ['ebm_nll', 'loss']:
            ax1.plot(df_noloss['epoch'], df_noloss[metric], label=metric)

    # Add a dotted horizontal line for the reference value
    if hamming_mmd_reference_value is not None:
        ax1.axhline(y=hamming_mmd_reference_value, color='gray', linestyle='--', label=f'MMD reference value at {hamming_mmd_reference_value}')

    if euclidean_mmd_reference_value is not None:
        ax1.axhline(y=euclidean_mmd_reference_value, color='black', linestyle='--', label=f'MMD reference value at {euclidean_mmd_reference_value}')


    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Create a second y-axis for the main metric
    if 'ebm_nll' in metrics:
        ax2 = ax1.twinx()
        ax2.set_ylabel('ebm_nll')
        if nll_lower_y_lim is not None and nlll_upper_y_lim is not None:
          ax2.set_ylim(nll_lower_y_lim,nlll_upper_y_lim)
        ax2.plot(df_noloss['epoch'], df_noloss['ebm_nll'], color='tab:red', label='ebm_nll')
        if nll_reference_value is not None:
          ax2.axhline(y=nll_reference_value, color='red', linestyle='--', label=f'NLL reference value at {nll_reference_value}')
        ax2.legend(loc='lower right')

    # Add averages text
    add_averages_text(ax1, df_noloss, metrics)

    # Save the plot as a PNG file
    plt.savefig(epochs_output)
    plt.show()

    # Plot metrics over time and save as PNG
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Value')
    ax1.set_title('Metrics over Time')
    if mmd_upper_y_lim is not None and mmd_lower_y_lim is not None:
      ax1.set_ylim(mmd_lower_y_lim, mmd_upper_y_lim)


    # Plot other metrics on the left y-axis
    for metric in metrics:
        if not metric in ['ebm_nll', 'loss']:
            ax1.plot(df_noloss['timestamp'], df_noloss[metric], label=metric)

    # Add a dotted horizontal line for the reference value
    if hamming_mmd_reference_value is not None:
        ax1.axhline(y=hamming_mmd_reference_value, color='gray', linestyle='--', label=f'Reference value at {hamming_mmd_reference_value}')

    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Create a second y-axis for the main metric
    if 'ebm_nll' in metrics:
        ax2 = ax1.twinx()
        ax2.set_ylabel('ebm_nll')
        if nll_lower_y_lim is not None and nlll_upper_y_lim is not None:
          ax2.set_ylim(nll_lower_y_lim,nlll_upper_y_lim)
        ax2.plot(df_noloss['timestamp'], df_noloss['ebm_nll'], color='tab:red', label='ebm_nll')
        ax2.legend(loc='lower right')

    metrics = df.columns[2:]
    if 'loss' in metrics:

      loss_plot_metrics = ['loss']

      # Plot metrics over time and save as PNG
      fig, ax1 = plt.subplots(figsize=(12, 6))

      ax1.set_xlabel('Timestamp')
      ax1.set_ylabel('Value')
      ax1.set_title('Metrics over Time')

      # Plot other metrics on the left y-axis
      ax1.plot(df['epoch'], df['loss'], label='loss')

      ax1.grid(True)
      ax1.legend(loc='upper left')

      # Create a second y-axis for the main metric
      if 'ebm_nll' in metrics:
          ax2 = ax1.twinx()
          ax2.set_ylabel('ebm_nll')
          if nll_lower_y_lim is not None and nlll_upper_y_lim is not None:
            ax2.set_ylim(nll_lower_y_lim,nlll_upper_y_lim)
          ax2.plot(df['epoch'], df['ebm_nll'], color='tab:red', label='ebm_nll')
          ax2.legend(loc='lower right')
          loss_plot_metrics.append('ebm_nll')
          if nll_reference_value is not None:
            ax2.axhline(y=nll_reference_value, color='red', linestyle='--', label=f'NLL reference value at {nll_reference_value}')


      # Add averages text
      add_averages_text(ax1, df, loss_plot_metrics)

      # Save the plot as a PNG file
      plt.savefig(loss_output)
      plt.show()

    print('Saved plots...')


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

def median_hamming_dist(x, y):
  torch.cuda.empty_cache()
  x = x.unsqueeze(1)
  y = y.unsqueeze(0)
  d = torch.sum(torch.abs(x - y), dim=-1)
  return torch.median(d).item()

def mean_hamming_dist(x, y):
  torch.cuda.empty_cache()
  x = x.unsqueeze(1)
  y = y.unsqueeze(0)
  d = torch.sum(torch.abs(x - y), dim=-1)
  return torch.mean(d).item()

def compute_dataset_stats(vocab_size = 2, discrete_dim = 32, batch_size=4000):
  class toy_args:
    def __init__(self, batch_size, bm, inv_bm, discrete_dim, int_scale, f_scale, vocab_size):
      self.batch_size=batch_size
      self.bm = bm
      self.inv_bm = inv_bm
      self.discrete_dim = discrete_dim
      self.int_scale = int_scale
      self.f_scale = f_scale
      self.vocab_size = vocab_size

  log_path = 'toy_data_stats.csv'
  assert not os.path.exists(log_path), f'Statistics have been computed before; see {log_path}'
  for data_name in {'swissroll', 'circles', 'moons', '8gaussians', 'pinwheel', '2spirals', 'checkerboard'}:
    db = toy_data_lib.OnlineToyDataset(data_name)
    bm, inv_bm = get_binmap(discrete_dim, 'gray')
    int_scale = float(db.int_scale)
    f_scale = float(db.f_scale)
    bx = db.gen_batch(batch_size)
    my_args = toy_args(batch_size, bm, inv_bm, discrete_dim, int_scale, f_scale, vocab_size)
    if vocab_size == 2:
        bx = float2bin(bx, bm, discrete_dim, int_scale)
    else:
        bx = ourfloat2base(bx, discrete_dim, f_scale, int_scale, vocab_size)
    bx = torch.from_numpy(np.float32(bx)).to('cpu')
    log_entry = {}
    log_entry['data_name'] = data_name
    log_entry['vocab_size'] = vocab_size
    log_entry['discrete_dim'] = discrete_dim
    log_entry['batch_size'] = batch_size
    log_entry['hamming_mean'] = mean_hamming_dist(bx, bx)
    log_entry['hamming_median'] = median_hamming_dist(bx, bx)
    log_entry['euclidean_mean'] = mean_euclidean_dist(bx, bx)
    log_entry['euclidean_median'] = median_euclidean_dist(bx, bx)
    log_entry['hamming_mean_mmd'], log_entry['hamming_var_mmd'], log_entry['bandwidth'], log_entry['euclidean_mean_mmd'], log_entry['euclidean_var_mmd'], log_entry['sigma'] = compute_mmd_base_stats(my_args, 20, db)

    df_log_entry = pd.DataFrame([log_entry])
    df_log_entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
  print(f'Mean and median hamming distances for all toy datasets have been saved to {log_path}')

def median_euclidean_dist(x, y):
  torch.cuda.empty_cache()
  x = x.unsqueeze(1)
  y = y.unsqueeze(0)
  d = torch.sqrt(torch.sum(torch.square(x - y), dim=-1))
  return torch.median(d).item()

def mean_euclidean_dist(x, y):
  torch.cuda.empty_cache()
  x = x.unsqueeze(1)
  y = y.unsqueeze(0)
  d = torch.sqrt(torch.sum(torch.square(x - y), dim=-1))
  return torch.mean(d).item()

def compute_euclidean_dist_stats(batch_size=4000):
  log_path = 'toy_data_euclidean_dist_stats.csv'
  assert not os.path.exists(log_path), f'Euclidean distance statistics have been computed before; all mmd computations should use the same sigma...; use {log_path}'
  for data_name in {'swissroll', 'circles', 'moons', '8gaussians', 'pinwheel', '2spirals', 'checkerboard'}:
    db = toy_data_lib.OnlineToyDataset(data_name)
    bx = db.gen_batch(batch_size)
    bx = torch.from_numpy(np.float32(bx)).to('cpu')
    log_entry = {}
    log_entry['data_name'] = data_name
    log_entry['batch_size'] = batch_size
    log_entry['mean'] = mean_euclidean_dist(bx, bx)
    log_entry['median'] = median_euclidean_dist(bx, bx)
    df_log_entry = pd.DataFrame([log_entry])
    df_log_entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
  print(f'Mean and median euclidean distances for all toy datasets have been saved to {log_path}')
  
def exp_hamming_sim(x, y, bd):
  torch.cuda.empty_cache()
  x = x.unsqueeze(1)
  y = y.unsqueeze(0)
  d = torch.sum(torch.abs(x - y), dim=-1)
  return torch.exp(-bd * d)

def rbf_kernel(x, y, sigma):
  torch.cuda.empty_cache()
  x = x.unsqueeze(1)
  y = y.unsqueeze(0)
  d = torch.sum((x - y) ** 2, dim=-1)
  return torch.exp(-d / (2 * sigma ** 2))

def exp_hamming_mmd(x, y, args, bandwidth=None, log_path='toy_data_hamming_dist_stats.csv'):
  if bandwidth == None:
  #   df = pd.read_csv(log_path)

  #   # Filter the DataFrame based on the criteria
  #   filtered_df = df[(df['data_name'] == args.data_name) & 
  #                    (df['vocab_size'] == args.vocab_size) & 
  #                    (df['discrete_dim'] == args.discrete_dim)]
    
  #   assert not filtered_df.shape[0] > 1, f"Warning: There are {filtered_df.shape[0]} duplicates for data_name={data_name}, vocab_size={vocab_size}, discrete_dim={discrete_dim}; resolve this and try again..."
    
  #   assert filtered_df.shape[0] > 0, f"Warning: There are no median hamming distances available for data_name={data_name}, vocab_size={vocab_size}, discrete_dim={discrete_dim}; resolve this and try again..."

  #   # Get the median value
  #   median = filtered_df['median'].values[0]
  #   bandwidth = 1 / median
  #   print(f'Imputed bandwidth {bandwidth} as reciprocal of the median {median} of the given dataset')
    bandwidth = torch.tensor(0.1)
  else:
    bandwidth = torch.tensor(bandwidth)


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
  return mmd, bandwidth

def rbf_mmd(x, y, args, sigma=None, log_path='toy_data_euclidean_dist_stats.csv'):
  if sigma == None:
    # df = pd.read_csv(log_path)

    # # Filter the DataFrame based on the criteria
    # filtered_df = df[(df['data_name'] == args.data_name)]    
    # assert not filtered_df.shape[0] > 1, f"Warning: There are {filtered_df.shape[0]} duplicates for data_name={data_name}; resolve this and try again..."
    
    # assert filtered_df.shape[0] > 0, f"Warning: There are no median hamming distances available for data_name={data_name}; resolve this and try again..."

    # # Get the median value
    # median = filtered_df['median'].values[0]
    # sigma = median
    # print(f'Imputed sigma {sigma} as median of the given dataset')
    sigma= torch.tensor(0.1)
  else:
    sigma = torch.tensor(sigma)

  if args.vocab_size == 2:
    x = bin2float(x.detach().cpu().numpy().astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
    y = bin2float(y.detach().cpu().numpy().astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
  else:
    x = ourbase2float(x.detach().cpu().numpy().astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    y = ourbase2float(y.detach().cpu().numpy().astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)

  x = torch.from_numpy(x)
  y = torch.from_numpy(y)

  with torch.no_grad():
      kxx = rbf_kernel(x, x, sigma=sigma)
      idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
      kxx[idx, idx] = 0.0
      kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

      kyy = rbf_kernel(y, y, sigma=sigma)
      idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
      kyy[idx, idx] = 0.0
      kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)

      kxy = torch.sum(rbf_kernel(x, y, sigma=sigma)) / x.shape[0] / y.shape[0]

      mmd = kxx + kyy - 2 * kxy
  return mmd, sigma


def ebm_evaluation(args, db, ebm, write_to_index=True, batch_size=4000, ais_samples=1000000, num_ais_mcmc_steps=25, ais_num_steps=1000, eval_nll = None):
  #MDD
  exp_hamming_mmd_list = []
  rbf_mmd_list = []
  for k in range(10):
    print('starting MMD iteration {k}')
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
    x = gibbs_sampler(ebm, num_rounds=args.gibbs_num_rounds, num_samples=batch_size).to('cpu')
    y = get_batch_data(db, args, batch_size=batch_size)
    y = torch.from_numpy(np.float32(y)).to('cpu')
    hamming_mmd, bandwidth = exp_hamming_mmd(x,y,args)
    euclidean_mmd, sigma = rbf_mmd(x,y,args)
    exp_hamming_mmd_list.append(hamming_mmd)
    rbf_mmd_list.append(euclidean_mmd)
  hamming_mmd = sum(exp_hamming_mmd_list)/10
  euclidean_mmd = sum(rbf_mmd_list)/10

  #NLL
  if eval_nll is None:
    eval_nll = args.eval_nll
  if eval_nll:
    print(f'starting ais as eval_nll was {eval_nll}')
    log_Z = annealed_importance_sampling(args, ebm, ais_samples, ais_num_steps, num_ais_mcmc_steps, args.discrete_dim)
    nll_samples = get_batch_data(db, args, batch_size=batch_size)
    nll_samples = torch.from_numpy(np.float32(nll_samples)).to(args.device)
    with torch.no_grad():
      nll = ebm(nll_samples) + log_Z
    nll = torch.sum(nll) / batch_size
  else:
    nll = torch.tensor(0)


  # print(f'NLL on {batch_size} samples with AIS on {ais_samples} samlpes: {nll}; Final exponential Hamming MMD on 10x{batch_size} samples: {mmd}')
  
  return nll.item(), hamming_mmd.item(), bandwidth.item(), euclidean_mmd.item(), sigma.item()


def sampler_evaluation(args, db, model, gen_samples, batch_size=4000):
  #note: there is no immedaite way to obtain NLL for DFS – this would require sampling a backward trajectory...
  exp_hamming_mmd_list = []
  rbf_mmd_list = []
  for _ in range(10):
    if args.source == 'data':
      xt = torch.from_numpy(get_batch_data(db, args, batch_size = 4000)).to(args.device)
    else:
      xt = None
    x = torch.from_numpy(gen_samples(model, args, batch_size = 4000, xt=xt)).to('cpu')
    y = get_batch_data(db, args, batch_size=4000)
    y = torch.from_numpy(np.float32(y)).to('cpu')
    hamming_mmd, bandwidth = exp_hamming_mmd(x,y,args)
    euclidean_mmd, sigma = rbf_mmd(x,y,args)
    exp_hamming_mmd_list.append(hamming_mmd)
    rbf_mmd_list.append(euclidean_mmd)
  hamming_mmd = sum(exp_hamming_mmd_list)/10
  euclidean_mmd = sum(rbf_mmd_list)/10
  
  return hamming_mmd.item(), bandwidth.item(), euclidean_mmd.item(), sigma.item()

def compute_mmd_base_stats(args, N, db, write_to_index=True, batch_size=4000):

  #computes variance and mean in mmd over N computations – should be used to judge the size of mmd in experiments
  hamming_mmd_outer_list = []
  euclidean_mmd_outer_list = []
  pbar = tqdm(range(N))
  for i in pbar:
    hamming_mmd_inner_list = []
    euclidean_mmd_inner_list = []
    for _ in range(10):
      y_1 = get_batch_data(db, args, batch_size=batch_size)
      y_1 = torch.from_numpy(np.float32(y_1)).to('cpu')
      y_2 = get_batch_data(db, args, batch_size=batch_size)
      y_2 = torch.from_numpy(np.float32(y_2)).to('cpu')
      hamming_mmd, bandwidth = exp_hamming_mmd(y_1,y_2,args)
      hamming_mmd_inner_list.append(hamming_mmd)
      euclidean_mmd, sigma = rbf_mmd(y_1,y_2,args)
      euclidean_mmd_inner_list.append(euclidean_mmd)
    hamming_mmd = sum(hamming_mmd_inner_list)/10
    euclidean_mmd = sum(euclidean_mmd_inner_list)/10
    hamming_mmd_outer_list.append(hamming_mmd.item())
    euclidean_mmd_outer_list.append(euclidean_mmd.item())
    pbar.set_description(f'Computed {i}/{N} MMDs')
  hamming_var_mmd = statistics.variance(hamming_mmd_outer_list)
  hamming_mean_mmd = statistics.mean(hamming_mmd_outer_list)
  euclidean_var_mmd = statistics.variance(euclidean_mmd_outer_list)
  euclidean_mean_mmd = statistics.mean(euclidean_mmd_outer_list)


  # print(f'MMD variance and mean over {N} computations were {var_mmd} and {mean_mmd}')
  
  return hamming_mean_mmd, hamming_var_mmd, bandwidth.item(), euclidean_mean_mmd, euclidean_var_mmd, sigma.item(),

def sampler_ebm_evaluation(args, db, model, gen_samples, ebm_model, batch_size=4000):
  #note: there is no immedaite way to obtain NLL for DFS – this would require sampling a backward trajectory...
  exp_hamming_mmd_list = []
  rbf_mmd_list = []
  gibbs_sampler = GibbsSampler(2, args.discrete_dim, args.device)
  for _ in range(10):
    if args.source == 'data':
      xt = torch.from_numpy(get_batch_data(db, args, batch_size = 4000)).to(args.device)
    else:
      xt = None
    x = torch.from_numpy(gen_samples(model, args, batch_size = 4000, xt=xt)).to('cpu')
    y = gibbs_sampler(ebm_model, num_rounds=100, num_samples=4000).to('cpu')
    hamming_mmd, bandwidth = exp_hamming_mmd(x,y,args)
    euclidean_mmd, sigma = rbf_mmd(x,y,args)
    exp_hamming_mmd_list.append(hamming_mmd)
    rbf_mmd_list.append(euclidean_mmd)
  hamming_mmd = sum(exp_hamming_mmd_list)/10
  euclidean_mmd = sum(rbf_mmd_list)/10

  return hamming_mmd.item(), bandwidth.item(), euclidean_mmd.item(), sigma.item()


def log(args, log_entry, itr, timestamp, log_path=None):
  if log_path is None:
    log_path = args.log_path
  log_entry['itr'] = itr
  log_entry['timestamp'] = timestamp
  df_log_entry = pd.DataFrame([log_entry])
  df_log_entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
  print(f'logged itr {itr} to log file')

def log_completion(method, data, args):
  lock_file_path = f"{args.index_path}.lock"
  lock = FileLock(lock_file_path, timeout=10)

  try:
    with lock:
      if os.path.exists(args.index_path) and os.path.getsize(args.index_path) > 0:
        try:
            with open(args.index_path, 'r') as file:
                experiment_idx = json.load(file)
        except json.JSONDecodeError:
            print("Warning: JSON file is corrupted. Cannot log completion.")
        experiment_idx[str(args.exp_n)]['completed'] = True
        with open(args.index_path, 'w') as file:
          json.dump(experiment_idx, file, indent=4)
        print(f'Completion logged.')
  except Timeout:
    print(f"Could not acquire the lock on {args.index_path} after 10 seconds.")

  except Exception as e:
    print(f"An error occurred: {e}")  


def convert_namespace_to_dict(args):
    args_dict = vars(args).copy()  # Convert Namespace to dictionary
    # Handle non-serializable objects
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)
    args_dict.pop('my_print', None)
    args_dict.pop('bm', None)
    args_dict.pop('inv_bm', None)
    return args_dict

def log_args(method, data, args):
  lock_file_path = f"{args.index_path}.lock"
  lock = FileLock(lock_file_path, timeout=10)

  try:
    with lock:
      if os.path.exists(args.index_path) and os.path.getsize(args.index_path) > 0:
          try:
              with open(args.index_path, 'r') as file:
                  experiment_idx = json.load(file)
          except json.JSONDecodeError:
              print("Warning: JSON file is corrupted. Aborting.")
              sys.exit(1)
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

      args.ckpt_path = f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/ckpts/'
      args.plot_path = f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/plots/'
      args.sample_path = f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/samples/'
      args.log_path = f'{args.save_dir}/{args.dataset_name}_{args.exp_n}/log.csv'

      experiment_idx[args.exp_n] = convert_namespace_to_dict(args)

      with open(args.index_path, 'w') as file:
          json.dump(experiment_idx, file, indent=4)
  except Timeout:
    print(f"Could not acquire the lock on {args.index_path} after 10 seconds.")

  except Exception as e:
    print(f"An error occurred: {e}")

  print(f"Experiment meta data saved to {args.index_path}")
  return args

def get_eval_timestamp(eval_start_time, cum_eval_time, start_time):
  eval_end_time = time.time()
  eval_time = eval_end_time - eval_start_time
  cum_eval_time += eval_time
  timestamp = time.time() - cum_eval_time - start_time
  return timestamp, cum_eval_time