import pandas as pd
import matplotlib.pyplot as plt
import sys

def create_comparison_plots_without_hamming(file1, file2, file3):
    # Load the three CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Create subplots (1x2 grid since Hamming MMD plots are removed)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Define line styles and colors for colorblind-friendly visualization
    colors = ['#0072B2', '#D55E00', '#009E73']  # Blue, Red, Green (colorblind-friendly)
    linestyles = ['-', '--', (0, (1, 1))]  # Solid, Dashed, Longer Dots for Green

    # Plot 1: sampler_ebm_euclidean_mmd
    axs[0].plot(df2['itr'], df2['sampler_ebm_euclidean_mmd'], color=colors[0], label='10 DFS per EBM (lr=0.0001)', linestyle=linestyles[0])
    axs[0].plot(df1['itr'], df1['sampler_ebm_euclidean_mmd'], color=colors[1], label='1 DFS per EBM (lr=0.0001)', linestyle=linestyles[1])
    axs[0].plot(df3['itr'], df3['sampler_ebm_euclidean_mmd'], color=colors[2], label='1 DFS per EBM (lr=0.001)', linestyle=linestyles[2])
    axs[0].set_title('E-MMD between DFS and EBM')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Euclidean MMD')
    axs[0]

    # Plot 2: ebm_euclidean_mmd
    axs[1].plot(df2['itr'], df2['ebm_euclidean_mmd'], color=colors[0], label='10 DFS per EBM (lr=0.0001)', linestyle=linestyles[0])
    axs[1].plot(df1['itr'], df1['ebm_euclidean_mmd'], color=colors[1], label='1 DFS per EBM (lr=0.0001)', linestyle=linestyles[1])
    axs[1].plot(df3['itr'], df3['ebm_euclidean_mmd'], color=colors[2], label='1 DFS per EBM (lr=0.001)', linestyle=linestyles[2])
    axs[1].set_title('EBM E-MMD')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Euclidean MMD')
    axs[1].legend()

    # Adjust the layout and font sizes for publication
    plt.tight_layout()

    # Save the plot in the same directory as the first CSV file
    save_directory = '/'.join(file1.split('/')[:-1])  # Parent directory of the first file
    save_path = f"{save_directory}/comparison_plot_without_hamming_with_third.csv.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")

# If the script is called from the command line, handle the arguments
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python make_ablation1and2_plots.py <path_to_first_csv> <path_to_second_csv> <path_to_third_csv>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    create_comparison_plots_without_hamming(file1, file2, file3)
