"""Toy data generators."""

import numpy as np
import pandas as pd
import sklearn
import os
import sklearn.datasets
import utils.utils as utils


def get_db(args):
  #toy data related
  if args.vocab_size == 2:
    args.discrete_dim = 32
    db, bm, inv_bm = utils.setup_data(args)
    args.bm = bm
    args.inv_bm = inv_bm
  else:
    db = utils.our_setup_data(args)
  if os.path.exists(f'{args.data_dir}/toy_data_stats.csv'):
    toy_data_stats = pd.read_csv(f'{args.data_dir}/toy_data_stats.csv')
  else:
    print('First run "compute_toy_data_stats.py", then try again.')
    sys.exit(0)
  args.hamming_mmd_mean = toy_data_stats.loc[toy_data_stats['dataset_name'] == args.dataset_name, 'hamming_mean_mmd'].values[0]
  args.hamming_mmd_var =  toy_data_stats.loc[toy_data_stats['dataset_name'] == args.dataset_name, 'hamming_var_mmd'].values[0]
  args.hamming_bandwidth_base_stats =  toy_data_stats.loc[toy_data_stats['dataset_name'] == args.dataset_name, 'bandwidth'].values[0]
  args.euclidean_mmd_mean = toy_data_stats.loc[toy_data_stats['dataset_name'] == args.dataset_name, 'euclidean_mean_mmd'].values[0]
  args.euclidean_mmd_var = toy_data_stats.loc[toy_data_stats['dataset_name'] == args.dataset_name, 'euclidean_var_mmd'].values[0]
  args.euclidean_sigma_base_stats = toy_data_stats.loc[toy_data_stats['dataset_name'] == args.dataset_name, 'sigma'].values[0]

  return db


def inf_train_gen(data, rng=None, batch_size=200):
  """Sample batch of synthetic data."""
  if rng is None:
    rng = np.random.RandomState()

  if data == "swissroll":
    data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
    data = data.astype("float32")[:, [0, 2]]
    data /= 5

    if data.shape[0] != batch_size:
        # If fewer samples are generated, resample the difference
        difference = batch_size - data.shape[0]
        additional_data = sklearn.datasets.make_swiss_roll(n_samples=difference, noise=1.0)[0]
        additional_data = additional_data.astype("float32")[:, [0, 2]]
        additional_data /= 5
        # Concatenate the original and additional data
        data = np.vstack((data, additional_data))

    return data

  elif data == "circles":
    data = sklearn.datasets.make_circles(
        n_samples=batch_size, factor=.5, noise=0.08)[0]
    data = data.astype("float32")
    data *= 3
    return data

  elif data == "moons":
    data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])
    return data

  elif data == "8gaussians":
    scale = 4.
    centers = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for _ in range(batch_size):
      point = rng.randn(2) * 0.5
      idx = rng.randint(8)
      center = centers[idx]
      point[0] += center[0]
      point[1] += center[1]
      dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    return dataset

  elif data == "pinwheel":
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = batch_size // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rng.randn(
        num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles),
                          np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

  # Check if more samples are needed to reach the batch size
    if data.shape[0] < batch_size:
        additional_samples_needed = batch_size - data.shape[0]
        
        additional_features = rng.randn(additional_samples_needed, 2) * np.array([radial_std, tangential_std])
        additional_features[:, 0] += 1.
        additional_labels = rng.choice(np.arange(num_classes), additional_samples_needed, replace=True)
        
        additional_angles = rads[additional_labels] + rate * np.exp(additional_features[:, 0])
        additional_rotations = np.stack([np.cos(additional_angles), -np.sin(additional_angles), np.sin(additional_angles), np.cos(additional_angles)])
        additional_rotations = np.reshape(additional_rotations.T, (-1, 2, 2))
        
        additional_data = 2 * np.einsum("ti,tij->tj", additional_features, additional_rotations)
        
        data = np.concatenate([data, additional_data], axis=0)
    
    return data

  elif data == "2spirals":
    n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return x

  elif data == "checkerboard":
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

  elif data == "line":
    x = rng.rand(batch_size) * 5 - 2.5
    y = x
    return np.stack((x, y), 1)
  elif data == "cos":
    x = rng.rand(batch_size) * 5 - 2.5
    y = np.sin(x) * 2.5
    return np.stack((x, y), 1)
  else:
    raise NotImplementedError


class OnlineToyDataset(object):
  """Wrapper of inf_datagen."""

  def __init__(self, data_name):
    self.dim = 2
    self.data_name = data_name
    self.rng = np.random.RandomState()

    rng = np.random.RandomState(1)
    samples = inf_train_gen(self.data_name, rng, 5000)
    self.f_scale = np.max(np.abs(samples)) + 1
    self.int_scale = 2 ** 15 / (self.f_scale + 1)
    print("f_scale,", self.f_scale, "int_scale,", self.int_scale)

  def gen_batch(self, batch_size):
    return inf_train_gen(self.data_name, self.rng, batch_size)

  def data_gen(self, batch_size):
    while True:
      yield self.gen_batch(batch_size)


class OurOnlineToyDataset(object):
  """Wrapper of inf_datagen."""

  def __init__(self, data_name, vocab_size, discrete_dim):
    self.dim = vocab_size
    self.data_name = data_name
    self.rng = np.random.RandomState()

    rng = np.random.RandomState(1)
    samples = inf_train_gen(self.data_name, rng, 5000)
    self.f_scale = np.max(np.abs(samples)) + 1
    self.int_scale = self.dim ** (discrete_dim // 2 - 1) / (self.f_scale + 1)
    print("f_scale,", self.f_scale, "int_scale,", self.int_scale)

  def gen_batch(self, batch_size):
    return inf_train_gen(self.data_name, self.rng, batch_size)

  def data_gen(self, batch_size):
    while True:
      yield self.gen_batch(batch_size)


class OurPosiOnlineToyDataset(object):
  """Wrapper of inf_datagen."""

  def __init__(self, data_name, vocab_size, discrete_dim):
    self.dim = vocab_size
    self.data_name = data_name
    self.rng = np.random.RandomState()

    rng = np.random.RandomState(1)
    samples = inf_train_gen(self.data_name, rng, 5000)
    self.f_scale = np.max(np.abs(samples)) + 1
    self.int_scale = self.dim ** (discrete_dim // 2) / (self.f_scale + 1)
    print("f_scale,", self.f_scale, "int_scale,", self.int_scale)

  def gen_batch(self, batch_size):
    return inf_train_gen(self.data_name, self.rng, batch_size)

  def data_gen(self, batch_size):
    while True:
      yield self.gen_batch(batch_size)