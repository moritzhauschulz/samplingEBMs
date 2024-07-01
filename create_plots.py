import pandas as pd
import matplotlib.pyplot as plt
from utils.eval import make_plots
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--csv_file', type=str, default='')
    parser.add_argument('--output', type=str, default='')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    # Read the CSV file
    make_plots(args.csv_file, output_dir=args.output)
