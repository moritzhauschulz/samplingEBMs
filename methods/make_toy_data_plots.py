
import matplotlib.pyplot as plt
from utils.toy_data_lib import get_db
from utils.utils import plot as toy_plot
from utils.utils import get_batch_data
import torch

print('finished imports')

toy_dataset_list = ['swissroll','circles','moons','8gaussians','pinwheel','2spirals','checkerboard'] 

plot = lambda p, x: toy_plot(p, x, args)

class my_args:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.batch_size = 1000
        self.sample_path = './final_report_utils'
        self.vocab_size = 2
        self.data_dir = './methods/datasets'


for data_name in toy_dataset_list:
    args = my_args(data_name)
    db = get_db(args)
    x = get_batch_data(db, args)
    plot(f'{args.sample_path}/source_{data_name}.png', torch.tensor(x).float())




