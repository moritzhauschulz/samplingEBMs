import pandas as pd
import matplotlib.pyplot as plt
from utils.eval import make_plots
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--csv_file', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--last_n', type=int, default=5)
    parser.add_argument('--mmd_reference_value', type=float, default=None)
    parser.add_argument('--nll_reference_value', type=float, default=None)



    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    # Read the CSV file
    make_plots(args.csv_file, last_n=args.last_n, output_dir=args.output, mmd_reference_value=args.mmd_reference_value, nll_reference_value=args.nll_reference_value)
