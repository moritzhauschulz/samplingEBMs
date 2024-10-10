#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

# Extract the file name without the extension
script_path="$0"
filename=$(basename "$script_path" .pt)


# Create the new directory name by appending '_output'
output="./${filename}_output/"

# Create the new directory in the current directory
mkdir -p $output

# Ensure correct PATH to your virtual environment
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate


# Move to the parent directory
cd ..

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output_date="./scripts/${filename}_output/${CURRENT_DATE}/"
mkdir -p $output_date

echo "Starting job ${filename} at $(date)"

############# SPECIFY JOB BELOW ################
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 1 --sigma 1

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 1 --sigma 1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 1 --sigma 1

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 0.1 --sigma 10

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 0.1 --sigma 10
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 0.1 --sigma 10

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 10 --sigma 0.1

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 10 --sigma 0.1
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 10 --sigma 0.1

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 0.5 --sigma 5

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 0.5 --sigma 5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 0.5 --sigma 5

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 5 --sigma 0.5

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 5 --sigma 0.5
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 5 --sigma 0.5

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 100 --sigma 0.01

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 100 --sigma 0.01
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 100 --sigma 0.01

# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 0.001 --sigma 0.00000000001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 0.001 --sigma 0.00000000001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 0.001 --sigma 0.00000000001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 0.001 --sigma 0.00000000001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 0.001 --sigma 0.00000000001

python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 0.001 --sigma 0.00000000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 0.001 --sigma 0.00000000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 0.001 --sigma 0.00000000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 0.001 --sigma 0.00000000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 0.001 --sigma 0.00000000001

python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 0.0001 --sigma 0.00000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 0.0001 --sigma 0.00000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 0.0001 --sigma 0.00000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 0.0001 --sigma 0.00000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 0.0001 --sigma 0.00000001

python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_100000.pt --model2 2spirals --bandwidth 0.0001 --sigma 0.0000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_200000.pt --model2 2spirals --bandwidth 0.0001 --sigma 0.0000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_300000.pt --model2 2spirals --bandwidth 0.0001 --sigma 0.0000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_400000.pt --model2 2spirals --bandwidth 0.0001 --sigma 0.0000001
python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 2spirals --bandwidth 0.0001 --sigma 0.0000001


# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --model2 checkerboard --bandwidth 0.01 --sigma 0.00001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_200000.pt --model2 checkerboard --bandwidth 0.01 --sigma 0.00001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_300000.pt --model2 checkerboard --bandwidth 0.01 --sigma 0.00001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_400000.pt --model2 checkerboard --bandwidth 0.01 --sigma 0.00001
# python -u compute_mmd.py --model_types dfs_data --model1  /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/checkerboard/checkerboard_0/ckpts/model_500000.pt --model2 checkerboard --bandwidth 0.01 --sigma 0.00001

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
