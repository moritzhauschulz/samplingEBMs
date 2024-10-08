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

# 2spirals dfs
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type linear --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type linear --source uniform --num_itr 50000 --eval_every 10000  --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type linear --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic_noise --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic_noise --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic_noise --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type cubic --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type cubic --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfs --scheduler_type cubic --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait 

#2spirals dfm
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type linear --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type linear --source uniform --num_itr 50000 --eval_every 10000  --itr_save 5000 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type linear --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic_noise --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic_noise --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic_noise --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))

python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type cubic --source mask --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type cubic --source uniform --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --batch_size 100 --dataset_name static_mnist --lr 0.0001 --delta_t 0.05 --methods velo_dfm --scheduler_type cubic --source data --num_itr 50000 --eval_every 10000 --itr_save 5000  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
