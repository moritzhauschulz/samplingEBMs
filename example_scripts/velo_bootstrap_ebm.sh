
#!/bin/bash

# Extract the file name without the extension
script_path="$0"
filename=$(basename "$script_path" .pt)

# Ensure correct PATH to your virtual environment
# export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
# source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output_date="./example_scripts/output/${filename}_output/${CURRENT_DATE}/"
mkdir -p $output_date

cd..

echo "Starting job ${filename} at $(date)"

# Initialize the counter
counter=1

############# SPECIFY JOB BELOW ################

#on dfm
python -u methods/main.py --dfs 0 --l2 0.1 --step_size_start 1 --adaptive_step_size 1 --sampler dmala --MCMC_refinement_dfs 10 --recycle_dfs_sample 1 --dataset_name 2spirals --methods velo_bootstrap_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 200000 --itr_save 5000 --dfs_per_ebm 1  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
