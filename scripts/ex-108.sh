
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

output_date="./scripts/output/${filename}_output/${CURRENT_DATE}/"
mkdir -p $output_date

echo "Starting job ${filename} at $(date)"


# Initialize the counter
counter=1

############# SPECIFY JOB BELOW ################

python -u methods/main.py --MCMC_refinement 10000 --step_size 1  --enable_backward 1 --dataset_name 2spirals --delta_t 0.01 --methods velo_edfm --scheduler_type linear --source mask --num_itr 0 --eval_every 50000 --t 0.9 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --MCMC_refinement 1000 --step_size 1  --enable_backward 1 --dataset_name 2spirals --delta_t 0.01 --methods velo_edfm --scheduler_type linear --source mask --num_itr 0 --eval_every 50000 --t 0.9 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --MCMC_refinement 100 --step_size 1  --enable_backward 1 --dataset_name 2spirals --delta_t 0.01 --methods velo_edfm --scheduler_type linear --source mask --num_itr 0 --eval_every 50000 --t 0.9 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --MCMC_refinement 10 --step_size 1  --enable_backward 1 --dataset_name 2spirals --delta_t 0.01 --methods velo_edfm --scheduler_type linear --source mask --num_itr 0 --eval_every 50000 --t 0.9 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --MCMC_refinement 1 --step_size 1  --enable_backward 1 --dataset_name 2spirals --delta_t 0.01 --methods velo_edfm --scheduler_type linear --source mask --num_itr 0 --eval_every 50000 --t 0.9 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
