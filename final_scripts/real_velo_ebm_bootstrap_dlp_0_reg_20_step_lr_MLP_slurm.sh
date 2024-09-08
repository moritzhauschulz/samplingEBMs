#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=40G 
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

JOB_BASE_NAME="${SLURM_JOB_NAME%.sh}"

# Ensure correct PATH to your virtual environment
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate


# Move to the parent directory
cd ..

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output_date="./final_scripts/output/${JOB_BASE_NAME}_output/${SLURM_JOB_ID}_${CURRENT_DATE}/"
mkdir -p $output_date

echo $output

# Redirect output and error to the desired files
exec >"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.out" 2>"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.err"

echo "Starting job ${JOB_BASE_NAME} with ID $SLURM_JOB_ID"
echo "Job started at $(date)"


# Initialize the counter
counter=1

############# SPECIFY JOB BELOW ################

#on dfm
python -u methods/main.py --dfs 0 --l2 0 --ebm_model mlp-256 --step_size_start 0.2 --adaptive_step_size 1 --dfs_warmup_iter 2000 --sampler dmala --MCMC_refinement_dfs 40 --recycle_dfs_sample 1  --dataset_name static_mnist --batch_size 100 --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source uniform --num_itr 50000 --eval_every 100000 --itr_save 2000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --dfs 1 --l2 0 --ebm_model mlp-256 --step_size_start 0.2 --adaptive_step_size 1 --dfs_warmup_iter 2000 --sampler dmala --MCMC_refinement_dfs 40 --recycle_dfs_sample 1  --dataset_name static_mnist --batch_size 100 --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type cubic --source mask --num_itr 50000 --eval_every 100000 --itr_save 2000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --dfs 0 --l2 0.001 --ebm_model mlp-256 --step_size_start 0.2 --adaptive_step_size 1 --dfs_warmup_iter 2000 --sampler dmala --MCMC_refinement_dfs 20 --recycle_dfs_sample 1  --dataset_name static_mnist --batch_size 100 --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source uniform --num_itr 50000 --eval_every 100000 --itr_save 2000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --dfs 1 --l2 0.001 --ebm_model mlp-256 --step_size_start 0.2 --adaptive_step_size 1 --dfs_warmup_iter 2000 --sampler dmala --MCMC_refinement_dfs 20 --recycle_dfs_sample 1  --dataset_name static_mnist --batch_size 100 --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type cubic --source mask --num_itr 50000 --eval_every 100000 --itr_save 2000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
