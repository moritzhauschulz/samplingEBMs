import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--csv_file', type=str, default='')
    parser.add_argument('--save_prefix', type=str, default='')

    args = parser.parse_args()

    return args

def make_plots(csv_file, save_prefix):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract the time component from the model1 name
    df['time'] = df['model1'].apply(lambda x: int(x.replace('.pt', '').split('_')[-1]))

    # Unique bandwidth values
    bandwidth_values = df['bandwidth'].unique()

    # Create a separate graph for each bandwidth value
    for bw in bandwidth_values:
        subset = df[df['bandwidth'] == bw]
        
        plt.figure()
        # for model in subset['model1'].unique():
            # model_subset = subset[subset['model1'] == model]
        plt.plot(subset['time'], subset['hamming_mmd'], label='hamming_mmd')
        
        plt.xlabel('Time')
        plt.ylabel('Hamming MMD')
        plt.title(f'Hamming MMD over Time (Bandwidth = {round(bw,3)})')
        plt.legend(title='Model')
        plt.grid(True)
        plt.savefig(f'mmd_results/hamming_mmd_bandwidth_{round(bw,3)}_{save_prefix}.png')
        plt.close()
    print('finished hamming')

    # Unique sigma values
    sigma_values = df['sigma'].unique()

    # Create a separate graph for each sigma value
    for sigma in sigma_values:
        subset = df[df['sigma'] == sigma]
        
        plt.figure()
        # for model in subset['model1'].unique():
        #     model_subset = subset[subset['model1'] == model]
        plt.plot(subset['time'], subset['euclidean_mmd'], label='euclidean_mmd')
        
        plt.xlabel('Time')
        plt.ylabel('Euclidean MMD')
        plt.title(f'Euclidean MMD over Time (Sigma = {round(sigma,3)})')
        plt.legend(title='Model')
        plt.grid(True)
        plt.savefig(f'mmd_results/euclidean_mmd_sigma_{round(sigma,3)}_{save_prefix}.png')
        plt.close()
    print('finished euclidean')


if __name__ == '__main__':
    args = get_args()
    # Read the CSV file
    make_plots(args.csv_file, args.save_prefix)
