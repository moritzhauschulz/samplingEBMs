
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

# static mnist
# python -u methods/main.py --q data_mean --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type cubic --source mask --num_itr 100000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --q data_mean --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source uniform --num_itr 100000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# # dynamic mnist
# python -u methods/main.py --q data_mean --batch_size 100 --dataset_name dynamic_mnist --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type cubic --source mask --num_itr 100000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u methods/main.py --q data_mean --batch_size 100 --dataset_name dynamic_mnist --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source uniform --num_itr 100000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# wait

# omniglot
python -u methods/main.py --q data_mean --batch_size 100 --dataset_name omniglot --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type cubic --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --q data_mean --batch_size 100 --dataset_name omniglot --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

# caltech
python -u methods/main.py --q data_mean --batch_size 100 --dataset_name caltech --lr 0.0001 --delta_t 0.05 --methods velo_edfs --scheduler_type cubic --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --q data_mean --batch_size 100 --dataset_name caltech --lr 0.0001 --delta_t 0.05 --methods velo_edfm --scheduler_type linear --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
