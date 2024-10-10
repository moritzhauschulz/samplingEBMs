
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

output_date="./final_scripts/output/${filename}_output/${CURRENT_DATE}/"
mkdir -p $output_date

echo "Starting job ${filename} at $(date)"


# Initialize the counter
counter=1

############# SPECIFY JOB BELOW ################

#on dfm
# python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name 2spirals --delta_t 0.01 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name 2spirals --delta_t 0.01 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name 2spirals --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name 2spirals --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name checkerboard --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name checkerboard --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name circles --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name circles --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

# python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name moons --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name moons --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name pinwheel --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name pinwheel --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name swissroll --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name swissroll --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# python -u methods/main.py --l2 0.1 --dfs 0 --ebm_init_mean 0 --dataset_name 8gaussians --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --l2 0.1 --dfs 1 --ebm_init_mean 0 --dataset_name 8gaussians --delta_t 0.05 --methods velo_ebm --scheduler_type linear --source mask --num_itr 200000 --eval_every 100000 --itr_save 5000 --dfs_per_ebm 10  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# wait 

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
