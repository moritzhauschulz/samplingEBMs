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
python compute_mmd.py  --delta_t 0.001 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output1.log 2>&1 &
python compute_mmd.py  --delta_t 0.01 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_2/experiments/2spirals/2spirals_44/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output2.log 2>&1 &

python compute_mmd.py  --delta_t 0.001 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_5/experiments/2spirals/2spirals_18/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output3.log 2>&1 &
python compute_mmd.py  --delta_t 0.01 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_5/experiments/2spirals/2spirals_18/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output4.log 2>&1 &

python compute_mmd.py  --delta_t 0.001 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_ce_2/experiments/2spirals/2spirals_6/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output5.log 2>&1 &
python compute_mmd.py  --delta_t 0.01 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_ce_2/experiments/2spirals/2spirals_6/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output6.log 2>&1 &

python compute_mmd.py  --delta_t 0.001 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_ce_forced/experiments/2spirals/2spirals_14/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output7.log 2>&1 &
python compute_mmd.py  --delta_t 0.01 --model_types dfs_ebm --model1 /vol/bitbucket/meh23/samplingEBMs/methods/mask_dfs_ce_forced/experiments/2spirals/2spirals_14/ckpts/model_500000.pt --model2 /vol/bitbucket/meh23/samplingEBMs/methods/ed_ebm/experiments/2spirals/2spirals_3/ckpts/model_100000.pt > ${output_date}output8.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
