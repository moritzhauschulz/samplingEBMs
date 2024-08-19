import torch
import os

import torch.nn as nn
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import utils.rbm as rbm
import utils.samplers as samplers
from tqdm import tqdm
from sympy.combinatorics.graycode import GrayCode
import json
from utils import toy_data_lib
from utils.sampler import GibbsSampler
import utils.utils as utils

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()


def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d


def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d

def langevin_approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    return gx.detach()

def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


def short_run_mcmc(logp_net, x_init, k, sigma, step_size=None):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    if step_size is None:
        step_size = (sigma ** 2.) / 2.
    for i in range(k):
        f_prime = torch.autograd.grad(logp_net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + sigma * torch.randn_like(x_k)

    return x_k


def get_data(args):
    if args.data == "mnist":
        transform = tr.Compose([tr.Resize(args.img_size), tr.ToTensor(), lambda x: (x > .5).float().view(-1)])
        train_data = torchvision.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
        test_data = torchvision.datasets.MNIST(root="../data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=True, drop_last=True)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.img_size, args.img_size),
                                                         p, normalize=True, nrow=sqrt(x.size(0)))
        encoder = None
        viz = None

    elif args.data_file is not None:
        with open(args.data_file, 'rb') as f:
            x = pickle.load(f)
        x = torch.tensor(x).float()
        train_data = TensorDataset(x)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = train_loader
        viz = None
        if args.model == "lattice_ising" or args.model == "lattice_ising_2d":
            plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                             p, normalize=False, nrow=int(x.size(0) ** .5))
        elif args.model == "lattice_potts":
            plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.dim, args.dim, 3).transpose(3, 1),
                                                             p, normalize=False, nrow=int(x.size(0) ** .5))
        else:
            plot = lambda p, x: None
    else:
        raise ValueError

    return train_loader, test_loader, plot, viz


def generate_data(args):
    if args.data_model == "lattice_potts":
        model = rbm.LatticePottsModel(args.dim, args.n_state, args.sigma)
        sampler = samplers.PerDimMetropolisSampler(model.data_dim, args.n_out, rand=False)
    elif args.data_model == "lattice_ising":
        model = rbm.LatticeIsingModel(args.dim, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
    elif args.data_model == "lattice_ising_3d":
        model = rbm.LatticeIsingModel(args.dim, args.sigma, lattice_dim=3)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.sigma)
        print(model.G)
        print(model.J)
    elif args.data_model == "er_ising":
        model = rbm.ERIsingModel(args.dim, args.degree, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.G)
        print(model.J)
    else:
        raise ValueError

    model = model.to(args.device)
    samples = model.init_sample(args.n_samples).to(args.device)
    print("Generating {} samples from:".format(args.n_samples))
    print(model)
    for _ in tqdm(range(args.gt_steps)):
        samples = sampler.step(samples, model).detach()

    return samples.detach().cpu(), model


def load_synthetic(mat_file, batch_size):
    import scipy.io
    mat = scipy.io.loadmat(mat_file)
    ground_truth_J = mat['eij']
    ground_truth_h = mat['hi']
    ground_truth_C = mat['C']
    q = mat['q']
    n_out = q[0, 0]

    x_int = mat['sample']
    n_samples, dim = x_int.shape

    x_int = torch.tensor(x_int).long() - 1
    x_oh = torch.nn.functional.one_hot(x_int, n_out)
    assert x_oh.size() == (n_samples, dim, n_out)

    x = torch.tensor(x_oh).float()
    train_data = TensorDataset(x)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    J = torch.tensor(ground_truth_J)
    j = J
    jt = j.transpose(0, 1).transpose(2, 3)
    ground_truth_J = (j + jt) / 2
    return train_loader, test_loader, x, \
           torch.tensor(ground_truth_J), torch.tensor(ground_truth_h), torch.tensor(ground_truth_C)



def load_real_protein(args):
    from data_utils import Alignment, load_distmap, MyDCA, map_matrix
    a2m = {
        "BPT1_BOVIN": f"{args.data_root}/BPT1_BOVIN/BPT1_BOVIN_full_b03.a2m",
        "CADH1_HUMAN": f"{args.data_root}/CADH1_HUMAN/CADH1_HUMAN_full_b02.a2m",
        "CHEY_ECOLI": f"{args.data_root}/CHEY_ECOLI/CHEY_ECOLI_full_b09.a2m",
        "ELAV4_HUMAN": f"{args.data_root}/ELAV4_HUMAN/ELAV4_HUMAN_full_b02.a2m",
        "O45418_CAEEL": f"{args.data_root}/O45418_CAEEL/O45418_CAEEL_full_b02.a2m",
        "OMPR_ECOLI": f"{args.data_root}/OMPR_ECOLI/OMPR_ECOLI_full_b08.a2m",
        "OPSD_BOVIN": f"{args.data_root}/OPSD_BOVIN/OPSD_BOVIN_full_b03.a2m",
        "PCBP1_HUMAN": f"{args.data_root}/PCBP1_HUMAN/PCBP1_HUMAN_full_b01.a2m",
        "RNH_ECOLI": f"{args.data_root}/RNH_ECOLI/RNH_ECOLI_full_b04.a2m",
        "THIO_ALIAC": f"{args.data_root}/THIO_ALIAC/THIO_ALIAC_full_b09.a2m",
        "TRY2_RAT": f"{args.data_root}/TRY2_RAT/TRY2_RAT_full_b02.a2m",
    }[args.data]
    print("Loading alignment...")
    with open(a2m, "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")
    print("Done")
    print("Loading distmap(s)")
    intra_file = {
        "BPT1_BOVIN": f"{args.data_root}/BPT1_BOVIN/BPT1_BOVIN_full_b03_distance_map_monomer",
        "CADH1_HUMAN": f"{args.data_root}/CADH1_HUMAN/CADH1_HUMAN_full_b02_distance_map_monomer",
        "CHEY_ECOLI": f"{args.data_root}/CHEY_ECOLI/CHEY_ECOLI_full_b09_distance_map_monomer",
        "ELAV4_HUMAN": f"{args.data_root}/ELAV4_HUMAN/ELAV4_HUMAN_full_b02_distance_map_monomer",
        "O45418_CAEEL": f"{args.data_root}/O45418_CAEEL/O45418_CAEEL_full_b02_distance_map_monomer",
        "OMPR_ECOLI": f"{args.data_root}/OMPR_ECOLI/OMPR_ECOLI_full_b08_distance_map_monomer",
        "OPSD_BOVIN": f"{args.data_root}/OPSD_BOVIN/OPSD_BOVIN_full_b03_distance_map_monomer",
        "PCBP1_HUMAN": f"{args.data_root}/PCBP1_HUMAN/PCBP1_HUMAN_full_b01_distance_map_monomer",
        "RNH_ECOLI": f"{args.data_root}/RNH_ECOLI/RNH_ECOLI_full_b04_distance_map_monomer",
        "THIO_ALIAC": f"{args.data_root}/THIO_ALIAC/THIO_ALIAC_full_b09_distance_map_monomer",
        "TRY2_RAT": f"{args.data_root}/TRY2_RAT/TRY2_RAT_full_b02_distance_map_monomer",
    }[args.data]
    distmap_intra = load_distmap(intra_file)

    multimer_file = {
        "BPT1_BOVIN": f"{args.data_root}/BPT1_BOVIN/BPT1_BOVIN_full_b03_distance_map_multimer",
        "CADH1_HUMAN": f"{args.data_root}/CADH1_HUMAN/CADH1_HUMAN_full_b02_distance_map_multimer",
        "CHEY_ECOLI": f"{args.data_root}/CHEY_ECOLI/CHEY_ECOLI_full_b09_distance_map_multimer",
        "ELAV4_HUMAN": f"{args.data_root}/ELAV4_HUMAN/ELAV4_HUMAN_full_b02_distance_map_multimer",
        "O45418_CAEEL": f"{args.data_root}/O45418_CAEEL/O45418_CAEEL_full_b02_distance_map_multimer",
        "OMPR_ECOLI": f"{args.data_root}/OMPR_ECOLI/OMPR_ECOLI_full_b08_distance_map_multimer",
        "OPSD_BOVIN": None,
        "PCBP1_HUMAN": f"{args.data_root}/PCBP1_HUMAN/PCBP1_HUMAN_full_b01_distance_map_multimer",
        "RNH_ECOLI": f"{args.data_root}/RNH_ECOLI/RNH_ECOLI_full_b04_distance_map_multimer",
        "THIO_ALIAC": f"{args.data_root}/THIO_ALIAC/THIO_ALIAC_full_b09_distance_map_multimer",
        "TRY2_RAT": None,
    }[args.data]
    if multimer_file is None:
        distmap_multimer = None
    else:
        distmap_multimer = load_distmap(multimer_file)

    num_ecs = {
        "BPT1_BOVIN": 57,
        "CADH1_HUMAN": 324,
        "CHEY_ECOLI": 114,
        "ELAV4_HUMAN": 140,
        "O45418_CAEEL": 234,
        "OMPR_ECOLI": 220,
        "OPSD_BOVIN": 266,
        "PCBP1_HUMAN": 190,
        "RNH_ECOLI": 133,
        "THIO_ALIAC": 95,
        "TRY2_RAT": 209,
    }[args.data]
    print("Done")
    print("Pulling data")
    L = aln.L
    D = len(aln.alphabet)
    print("Raw Data size {}".format((L, D)))


    dca = MyDCA(aln)
    #dca.alignment.set_weights()
    #print(dca.alignment.weights.sum(), "MY DCA SUM")
    L = dca.alignment.L
    D = len(dca.alignment.alphabet)
    x_int = torch.from_numpy(dca.int_matrix()).float()
    x_oh = torch.nn.functional.one_hot(x_int.long(), D).float()
    print("Filtered Data size {}".format((L, D)))


    J = -distmap_intra.dist_matrix
    J = J + args.contact_cutoff
    J[J < 0] = 0.
    J[np.isnan(J)] = 0.  # treat unobserved values as just having no contact
    ind = np.diag_indices(J.shape[0])
    J[ind] = 0.
    C = np.copy(J)
    C[C > 0] = 1.
    C[C <= 0] = 0.
    print("Done")
    print("J size = {}".format(J.shape))

    weight_file = f"{args.data_root}/{args.data}/weights.pkl"
    if not os.path.exists(weight_file):
        print("Generating weights")
        dca.alignment.set_weights()
        weights = dca.alignment.weights
        with open(weight_file, 'wb') as f:
            pickle.dump(weights, f)
    else:
        print("Loading weights")
        with open(weight_file, 'rb') as f:
            weights = pickle.load(f)

    weights = torch.tensor(weights).float()
    print("Done")
    print("Dataset has {} examples, sum weights are {}".format(weights.size(0), weights.sum()))
    print("Scaling up by {}".format(float(weights.size(0)) / weights.sum()))
    weights *= float(weights.size(0)) / weights.sum()
    print("Distmap size {}".format(J.shape))

    train_data = TensorDataset(x_oh, weights)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    # pull indices from distance map
    #dm_indices = list(torch.tensor(np.array(distmap_intra.residues_j.id).astype(int) - 1).numpy())
    dm_indices = list(torch.tensor(np.array(distmap_intra.residues_j.id).astype(int)).numpy())
    print(dm_indices)
    dca_indices = dca.index_list
    print(dca_indices)
    int_indices = list(set(dm_indices).intersection(set(dca_indices)))
    dm_int_indices = []
    for i, ind in enumerate(dm_indices):
        if ind in int_indices:
            dm_int_indices.append(i)

    dca_int_indices = []
    for i, ind in enumerate(dca_indices):
        if ind in int_indices:
            dca_int_indices.append(i)

    print(dm_int_indices)
    print(dca_int_indices)

    print(len(dm_int_indices))
    print(len(dca_int_indices))

    print("Removing indices from C and J")
    print("Old size: {}".format(C.shape))
    C = C[dm_int_indices][:, dm_int_indices]
    J = J[dm_int_indices][:, dm_int_indices]
    print("New size: {}".format(C.shape))
    dca_int_indices = torch.tensor(dca_int_indices).long()
    print(dca_int_indices)
    return train_loader, test_loader, x_oh, num_ecs, torch.tensor(J), torch.tensor(C), dca_int_indices

def load_ingraham(args):
    from data_utils import Alignment, map_matrix
    with open("{}/PF00018.a2m".format(args.data_root), "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")

    L = aln.L
    D = len(aln.alphabet)
    x_int = torch.from_numpy(map_matrix(aln.matrix, aln.alphabet_map))
    n_out = D
    x_oh = torch.nn.functional.one_hot(x_int.long(), n_out).float()

    print(L, D, x_oh.size())

    aln.set_weights()
    weights = torch.tensor(aln.weights).float()
    print("Dataset has {} examples, sum weights are {}".format(weights.size(0), weights.sum()))
    print("Scaling up by {}".format(float(weights.size(0)) / weights.sum()))
    weights *= float(weights.size(0)) / weights.sum()

    with open("{}/PF00018_summary.txt".format(args.data_root), 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        X = [int(line[0]) for line in lines]
        Y = [int(line[1]) for line in lines]
        M = [float(line[5]) for line in lines]
        D = np.zeros((48, 48))
        for x, y, m in zip(X, Y, M):
            D[x - 1, y - 1] = m
            J = -(D + D.T)
            J = J + args.contact_cutoff
            J[J < 0] = 0.
            ind = np.diag_indices(J.shape[0])
            J[ind] = 0.
            C = np.copy(J)
            C[C > 0] = 1.
            C[C <= 0] = 0.

    print("Distmap size {}".format(J.shape))

    train_data = TensorDataset(x_oh, weights)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    return train_loader, test_loader, x_oh, 200, torch.tensor(J), torch.tensor(C)


#for toy data handling:
def get_last_n_levels(path, n=6):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Check if the path has at least six levels
    n = min(len(parts),n)
    
    # Get the last six levels and join them back into a string
    last_n_levels = parts[-n:]
    result = '_'.join(last_n_levels)
    
    return result

def remove_last_n_levels(path, n=6):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Check if the path has at least six levels
    n = min(len(parts),n)
    
    # Get the last six levels and join them back into a string
    without_last_n_levels = parts[:len(parts)-n]
    result = '/'.join(without_last_n_levels)
    
    return result


def has_n_levels(path, n=6):
    # Split the path into components
    parts = path.split(os.sep)
    
    # Check if the path has at least six levels
    return len(parts) >= n


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
  db = toy_data_lib.OnlineToyDataset(args.dataset_name)
  args.int_scale = float(db.int_scale)
  args.plot_size = float(db.f_scale)
  return db, bm, inv_bm

def our_setup_data(args):
  db = toy_data_lib.OurPosiOnlineToyDataset(args.dataset_name, args.vocab_size, args.discrete_dim)
  args.int_scale = float(db.int_scale)
  args.f_scale = float(db.f_scale)
  args.plot_size = float(db.f_scale)
  return db

def plot_samples(samples, out_name, im_size=0, axis=False, im_fmt='png', highlighted=None):
  """Plot samples."""
  if highlighted is None:
    plt.scatter(samples[:, 0], samples[:, 1], marker='.',c='blue')
  else:
    plt.scatter(samples[~highlighted][:, 0], samples[~highlighted][:, 1], marker='.')
    plt.scatter(samples[highlighted][:, 0], samples[highlighted][:, 1], marker='.',c='red')
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

def get_x0(B,D,S,args):
    if args.source == 'mask':
        M = S - 1
        x0 = torch.ones((B,D)).to(args.device).long() * M
    elif args.source == 'uniform':
        x0 = torch.randint(0, S, (B, D)).to(args.device)     
    else:
        raise NotImplementedError("This dataset-source combination is not supported")

    return x0

def get_batch_data(db, args, batch_size = None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def plot(path, samples, args):
    samples = samples.detach().cpu().numpy()
    if args.vocab_size == 2:
        float_samples = utils.bin2float(samples.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
    else:
        float_samples = utils.ourbase2float(samples.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    utils.plot_samples(float_samples, path, im_size=4.1, im_fmt='png')

