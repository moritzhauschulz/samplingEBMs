
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

#on dfm
python -u methods/main.py --dfs 0 --l2 0.1 --step_size_start 1 --adaptive_step_size 1 --sampler dmala --MCMC_refinement_dfs 5 --recycle_dfs_sample 1 --dataset_name 2spirals --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 1000 --eval_nll_every 100000 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --dfs 0 --l2 0.1 --step_size_start 1 --adaptive_step_size 1  --sampler dmala --MCMC_refinement_dfs 5 --recycle_dfs_sample 1 --dataset_name checkerboard --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --dfs 0 --l2 0.1 --step_size_start 1 --adaptive_step_size 1  --sampler dmala --MCMC_refinement_dfs 5 --recycle_dfs_sample 1 --dataset_name circles --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --dfs 0 --l2 0.1 --step_size_start 1 --adaptive_step_size 1  --sampler dmala --MCMC_refinement_dfs 5 --recycle_dfs_sample 1 --dataset_name moons --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait


# #on dfm
# python -u methods/main.py --dfs 0 --l2 0.1 --step_size 1 --sampler dmala --MCMC_refinement 5 --dataset_name 2spirals --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --dfs 0 --l2 0.1 --step_size 1 --sampler dmala --MCMC_refinement 5 --dataset_name checkerboard --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --dfs 0 --l2 0.1 --step_size 1 --sampler dmala --MCMC_refinement 5 --dataset_name circles --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --dfs 0 --l2 0.1 --step_size 1 --sampler dmala --MCMC_refinement 5 --dataset_name moons --delta_t 0.05 --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 1000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))


# wait 

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
