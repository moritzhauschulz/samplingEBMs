from utils import utils
from utils.eval import compute_mmd_base_stats
import numpy as np
import torch

class dummy_args:
    def __init__(self, discrete_dim, data_name):
        self.discrete_dim = discrete_dim
        self.data_name = data_name
        self.int_scale = 0
        self.plot_size = 0
        self.vocab_size = 2
        self.batch_size = 128
        self.bm = None
        self.inv_bm = None

data_name = input('Enter dataset name to compute mmd mean and variance for: ')
N = int(input('Sample size N (e.g. 20): '))

my_args = dummy_args(32, data_name)
db, my_args.bm, my_args.inv_bm = utils.setup_data(my_args)


mean_var, mmd_var = compute_mmd_base_stats(my_args, N, db)
