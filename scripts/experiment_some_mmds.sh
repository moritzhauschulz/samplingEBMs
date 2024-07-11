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
python compute_mmd.py --data_name 2spirals --model_types gfn-data --model1 /vol/bitbucket/meh23/samplingEBMs/methods/eb_gfn/experiments/2spirals/2spirals_eb_gfn_2spirals_11/ckpts/gfn_model_5000.pt --model2 2spirals > ${output_date}output1.log 2>&1 &
python compute_mmd.py --data_name 2spirals --model_types gfn-data --model1 /vol/bitbucket/meh23/samplingEBMs/methods/eb_gfn/experiments/2spirals/2spirals_eb_gfn_2spirals_11/ckpts/gfn_model_15000.pt --model2 2spirals > ${output_date}output2.log 2>&1 &
python compute_mmd.py --data_name 2spirals --model_types gfn-data --model1 /vol/bitbucket/meh23/samplingEBMs/methods/eb_gfn/experiments/2spirals/2spirals_eb_gfn_2spirals_11/ckpts/gfn_model_20000.pt --model2 2spirals > ${output_date}output3.log 2>&1 &

python compute_mmd.py --data_name 2spirals --model_types dfs-data --model1 /vol/bitbucket/meh23/samplingEBMs/methods/dataq_dfs/experiments/2spirals/2spirals_13/ckpts/model_7500.pt --model2 2spirals > ${output_date}output4.log 2>&1 &
python compute_mmd.py --data_name 2spirals --model_types dfs-data --model1 /vol/bitbucket/meh23/samplingEBMs/methods/dataq_dfs/experiments/2spirals/2spirals_13/ckpts/model_20000.pt --model2 2spirals > ${output_date}output5.log 2>&1 &
python compute_mmd.py --data_name 2spirals --model_types dfs-data --model1 /vol/bitbucket/meh23/samplingEBMs/methods/dataq_dfs/experiments/2spirals/2spirals_13/ckpts/model_75000.pt --model2 2spirals > ${output_date}output6.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
