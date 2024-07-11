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

def get_last_n_levels(path, n=6):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Check if the path has at least six levels
    n = min(len(parts),n)
    
    # Get the last six levels and join them back into a string
    last_n_levels = parts[-n:]
    result = '_'.join(last_n_levels)
    
    return result


def print_cuda_memory_stats():
    if torch.cuda.is_available():
        # Get the total and available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = reserved_memory - allocated_memory

        print(f"Total memory: {total_memory / 1024**2:.2f} MB")
        print(f"Reserved memory: {reserved_memory / 1024**2:.2f} MB")
        print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
        print(f"Free memory (within reserved): {free_memory / 1024**2:.2f} MB")
        print(f"Available memory: {(total_memory - reserved_memory + free_memory) / 1024**2:.2f} MB")
    else:
        print("CUDA is not available on this system. Cannot print CUDA stats.")


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
