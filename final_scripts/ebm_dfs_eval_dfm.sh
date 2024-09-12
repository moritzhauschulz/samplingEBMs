
#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

# Extract the file name without the extension
script_path="$0"
filename=$(basename "$script_path" .pt)

# Ensure correct PATH to your virtual environment
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate


# Move to the parent directory
cd ..

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output_date="./final_scripts/output/${filename}_output/${CURRENT_DATE}/"
mkdir -p $output_date

echo "Starting job ${filename} at $(date)"


# Initialize the counter
counter=1

############# SPECIFY JOB BELOW ################
#dfm
python -u methods/temp_toy_dfs_ebm_eval.py  --ebm_model /vol/bitbucket/meh23/samplingEBMs/methods/velo_bootstrap_ebm/experiments/2spirals/2spirals_140/ckpts/ebm_model_200000.pt > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/temp_toy_dfs_ebm_eval.py  --ebm_model /vol/bitbucket/meh23/samplingEBMs/methods/velo_bootstrap_ebm/experiments/checkerboard/checkerboard_21/ckpts/ebm_model_200000.pt > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/temp_toy_dfs_ebm_eval.py  --ebm_model /vol/bitbucket/meh23/samplingEBMs/methods/velo_bootstrap_ebm/experiments/circles/circles_13/ckpts/ebm_model_200000.pt > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/temp_toy_dfs_ebm_eval.py  --ebm_model /vol/bitbucket/meh23/samplingEBMs/methods/velo_bootstrap_ebm/experiments/moons/moons_13/ckpts/ebm_model_200000.pt > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/temp_toy_dfs_ebm_eval.py  --ebm_model /vol/bitbucket/meh23/samplingEBMs/methods/velo_bootstrap_ebm/experiments/pinwheel/pinwheel_13/ckpts/ebm_model_200000.pt > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/temp_toy_dfs_ebm_eval.py  --ebm_model /vol/bitbucket/meh23/samplingEBMs/methods/velo_bootstrap_ebm/experiments/swissroll/swissroll_13/ckpts/ebm_model_200000.pt > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/temp_toy_dfs_ebm_eval.py  --ebm_model /vol/bitbucket/meh23/samplingEBMs/methods/velo_bootstrap_ebm/experiments/8gaussians/8gaussians_13/ckpts/ebm_model_200000.pt > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait
############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
