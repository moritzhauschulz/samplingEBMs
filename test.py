from methods.dataq_dfs.model import MixtureModel
from utils import utils
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

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

my_args = dummy_args(32, '2spirals')

db, my_args.bm, my_args.inv_bm = utils.setup_data(my_args)

samples = utils.get_batch_data(db, my_args, batch_size=10000)
float_samples = utils.bin2float(samples.astype(np.int32), my_args.inv_bm, my_args.discrete_dim, my_args.int_scale)
utils.plot_samples(float_samples, './samples_plot.png')

mean = np.mean(samples, axis=0)
print(mean)
q_dist = MixtureModel(samples, mean, 0.75, device='cuda')

mixed_samples = q_dist.sample(1000).detach().cpu().numpy()
mixed_float_samples = utils.bin2float(mixed_samples.astype(np.int32), my_args.inv_bm, my_args.discrete_dim, my_args.int_scale)

utils.plot_samples(mixed_float_samples, './mixed_samples_plot.png')
