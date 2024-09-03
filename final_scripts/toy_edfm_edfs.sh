
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

#2spirals dfm
python -u methods/main.py --q data_mean --dataset_name 2spirals --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --q random --dataset_name 2spirals --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#2spirals dfs
python -u methods/main.py --q data_mean --dataset_name 2spirals --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --q random --dataset_name 2spirals --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#checkerboard dfm
python -u methods/main.py --q data_mean --dataset_name checkerboard --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --q random --dataset_name checkerboard --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#checkerboard dfs
python -u methods/main.py --q data_mean --dataset_name checkerboard --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --q random --dataset_name checkerboard --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#circles dfm
python -u methods/main.py --q data_mean --dataset_name circles --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#circles dfs
python -u methods/main.py --q data_mean --dataset_name circles --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#swissroll dfm
python -u methods/main.py --q data_mean --dataset_name swissroll --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#swissroll dfs
python -u methods/main.py --q data_mean --dataset_name swissroll --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#moons dfm
python -u methods/main.py --q data_mean --dataset_name moons --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#moons dfs
python -u methods/main.py --q data_mean --dataset_name moons --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#pinwheel dfm
python -u methods/main.py --q data_mean --dataset_name pinwheel --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#pinwheel dfs
python -u methods/main.py --q data_mean --dataset_name pinwheel --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#8gaussians dfm
python -u methods/main.py --q data_mean --dataset_name 8gaussians --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#8gaussians dfs
python -u methods/main.py --q data_mean --dataset_name 8gaussians --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type linear --source mask --num_itr 200000 --eval_every 50000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
