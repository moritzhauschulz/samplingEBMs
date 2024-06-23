import pandas as pd
import matplotlib.pyplot as plt
from utils.eval import make_plots


# Read the CSV file
csv_file = '/vol/bitbucket/meh23/samplingEBMs/methods/cd_runi_inter/experiments/2spirals/2spirals_cd_runi_inter_2spirals_1/log.csv'  # replace with the path to your CSV file
make_plots(csv_file)
