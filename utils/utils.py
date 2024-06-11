import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy.combinatorics.graycode import GrayCode
import json

from utils import toy_data_lib
from utils.sampler import GibbsSampler


def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def recover(bx):
  x = int(bx[1:], 2)
  return x if bx[0] == '0' else -x

def our_recover(bx, vocab_size):
  x = int(bx, vocab_size)
  return x

def compress(x, discrete_dim):
  bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
  bx = '0' + bx if x >= 0 else '1' + bx
  return bx

def our_compress(x, discrete_dim, vocab_size):
  bx = np.base_repr(int(abs(x)), base=vocab_size).zfill(discrete_dim // 2)
  return bx

def ourbase2float(samples, discrete_dim, f_scale, int_scale, vocab_size):
  """Convert binary to float numpy."""
  floats = []
  for i in range(samples.shape[0]):
    s = ''
    for j in range(samples.shape[1]):
      s += str(samples[i, j])
    x, y = s[:discrete_dim//2], s[discrete_dim//2:]
    x, y = our_recover(x, vocab_size), our_recover(y, vocab_size)
    x = x / int_scale * 2. - f_scale
    y = y / int_scale * 2. - f_scale
    floats.append((x, y))
  return np.array(floats)

def ourfloat2base(samples, discrete_dim, f_scale, int_scale, vocab_size):
  base_list = []
  for i in range(samples.shape[0]):
    x, y = (samples[i] + f_scale) / 2 * int_scale
    bx, by = our_compress(x, discrete_dim, vocab_size), our_compress(y, discrete_dim, vocab_size)
    base_list.append(np.array(list(bx + by), dtype=int))
  return np.array(base_list)

def bin2float(samples, inv_bm, discrete_dim, int_scale):
  """Convert binary to float numpy."""
  floats = []
  for i in range(samples.shape[0]):
    s = ''
    for j in range(samples.shape[1]):
      s += str(samples[i, j])
    x, y = s[:discrete_dim//2], s[discrete_dim//2:]
    x, y = inv_bm[x], inv_bm[y]
    x, y = recover(x), recover(y)
    x /= int_scale
    y /= int_scale
    floats.append((x, y))
  return np.array(floats)

def get_binmap(discrete_dim, binmode):
  """Get binary mapping."""
  b = discrete_dim // 2 - 1
  all_bins = []
  for i in range(1 << b):
    bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
    all_bins.append('0' + bx)
    all_bins.append('1' + bx)
  vals = all_bins[:]
  if binmode == 'gray':
    print('remapping binary repr with gray code')
    a = GrayCode(b)
    vals = []
    for x in a.generate_gray():
      vals.append('0' + x)
      vals.append('1' + x)
  else:
    assert binmode == 'normal'
  bm = {}
  inv_bm = {}
  for i, key in enumerate(all_bins):
    bm[key] = vals[i]
    inv_bm[vals[i]] = key
  return bm, inv_bm

def float2bin(samples, bm, discrete_dim, int_scale):
  bin_list = []
  for i in range(samples.shape[0]):
    x, y = samples[i] * int_scale
    bx, by = compress(x, discrete_dim), compress(y, discrete_dim)
    bx, by = bm[bx], bm[by]
    bin_list.append(np.array(list(bx + by), dtype=int))
  return np.array(bin_list)

def setup_data(args):
  bm, inv_bm = get_binmap(args.discrete_dim, 'gray') 
  db = toy_data_lib.OnlineToyDataset(args.data_name)
  args.int_scale = float(db.int_scale)
  args.plot_size = float(db.f_scale)
  return db, bm, inv_bm

def our_setup_data(args):
  db = toy_data_lib.OurPosiOnlineToyDataset(args.data_name, args.vocab_size, args.discrete_dim)
  args.int_scale = float(db.int_scale)
  args.f_scale = float(db.f_scale)
  args.plot_size = float(db.f_scale)
  return db

def plot_samples(samples, out_name, im_size=0, axis=False, im_fmt='png'):
  """Plot samples."""
  plt.scatter(samples[:, 0], samples[:, 1], marker='.')
  plt.axis('image')
  if im_size > 0:
    plt.xlim(-im_size, im_size)
    plt.ylim(-im_size, im_size)
  if not axis:
    plt.axis('off')
  if isinstance(out_name, str):
    im_fmt = None
  plt.savefig(out_name, bbox_inches='tight', format=im_fmt)
  plt.close()

def plot_heat(score_func, size, bm, out_file, args, im_fmt='png'):
    score_func.eval()
    w = 100
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    heat_samples = float2bin(np.concatenate((xx, yy), axis=-1), bm, args.discrete_dim, args.int_scale)
    heat_samples = torch.from_numpy(np.float32(heat_samples)).to(args.device)
    heat_score = F.softmax(-1 * score_func(heat_samples).view(1, -1), dim=-1)
    a = heat_score.view(w, w).data.cpu().numpy()
    a = np.flip(a, axis=0)
    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, format=im_fmt)
    plt.close()

def plot_sampler(score_func, out_file, args):
    gibbs_sampler = GibbsSampler(2, args.discrete_dim, args.device)
    samples = []
    for _ in tqdm(range(10)):
        with torch.no_grad():
            init_samples = gibbs_sampler(score_func, num_rounds=50, num_samples=128) ### *(-1) ???
        samples.append(bin2float(init_samples.data.cpu().numpy(), args.inv_bm, args.discrete_dim, args.int_scale))
    samples = np.concatenate(samples, axis=0)
    plot_samples(samples, out_file, im_size=4.1)

def energy_source(x):
    return -torch.sum(torch.log((1 - 0.5) ** x * 0.5 ** (1-x)), dim=-1)

def ais_mcmc_step(args, x, energy_fn, step_size=0.1):
    x_new = torch.bernoulli(torch.full_like(x, 0.5))
    accept_prob = torch.exp(energy_fn(x) - energy_fn(x_new))
    accept = torch.rand(x.shape[0]).to(args.device) < accept_prob
    x[accept] = x_new[accept]
    return x

def annealed_importance_sampling(args, score_fn, num_samples, num_intermediate, num_mcmc_steps, latent_dim):
    x = torch.bernoulli(torch.full((num_samples, latent_dim), 0.5)).to(args.device)  # Initial samples from the source distribution
    betas = np.linspace(0, 1, num_intermediate)
    
    log_weights = torch.zeros(num_samples).to(args.device)
    
    for i in range(num_intermediate - 1):
        beta0, beta1 = betas[i], betas[i + 1]
        
        def energy_fn(x):
            energy = (1 - beta0) * energy_source(x) + beta0 * score_fn(x)
            return energy
        
        for _ in range(num_mcmc_steps):
            x = ais_mcmc_step(args, x, energy_fn)
        
        log_weights += (beta1 - beta0) * (score_fn(x).to(args.device) - energy_source(x).to(args.device))
    
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

def ebm_evaluation(args, db, ebm, write_to_index=True, batch_size=4000, ais_samples=1000000, num_ais_mcmc_steps=25, ais_num_intermediate=1000,discrete_dim=32):
  #NLL
  Z = torch.exp(annealed_importance_sampling(args, ebm, ais_samples, ais_num_intermediate, num_ais_mcmc_steps, discrete_dim))
  nll_samples = get_batch_data(db, args, batch_size=batch_size)
  nll_samples = torch.from_numpy(np.float32(nll_samples)).to(args.device)
  nll = ebm(nll_samples)
  nll = torch.sum(nll / Z)

  #MDD
  mmd_list = []
  for _ in range(10):
    gibbs_sampler = GibbsSampler(n_choices = args.vocab_size, discrete_dim=args.discrete_dim, device=args.device)
    x = gibbs_sampler(ebm, num_rounds=100, num_samples=batch_size).to('cpu')
    y = get_batch_data(db, args, batch_size=4000)
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
      experiment_idx[str(args.idx)]['nll'] = nll.item()
      experiment_idx[str(args.idx)]['mmd'] = mmd.item()
      with open(experiment_idx_path, 'w') as file:
        json.dump(experiment_idx, file, indent=4)
      print(f'Evaluation results written to index file: {experiment_idx_path}')
    else:
      print('Could not write evaluation to index because index file was not found.')
  
  return nll, mmd



  
  