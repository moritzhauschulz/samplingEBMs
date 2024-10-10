#!/bin/bash

#SBATCH --gres=gpu:1
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

output_date="./scripts/${JOB_BASE_NAME}_output/${SLURM_JOB_ID}_${CURRENT_DATE}/"
mkdir -p $output_date

echo $output

# Redirect output and error to the desired files
exec >"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.out" 2>"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.err"

echo "Starting job ${JOB_BASE_NAME} with ID $SLURM_JOB_ID"
echo "Job started at $(date)"

# Initialize the counter
counter=1

############# SPECIFY JOB BELOW ################

#lin t
python -u methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1 --epoch_save 2   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source uniform  --lin_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#rand t
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1  --epoch_save 2   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source uniform  --rand_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#fixed t=0.5
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1  --epoch_save 2   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source uniform  --t 0.75 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#lin t
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1 --epoch_save 2   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source mask  --lin_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#rand t
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1 --epoch_save 2    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source mask  --rand_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#fixed t=0.5
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1  --epoch_save 2   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source mask  --t 0.75 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#lin t
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1 --epoch_save 2    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 2000 --source data  --lin_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#rand t
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1 --epoch_save 2    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 2000 --source data --rand_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#fixed t=0.5
python -u  methods_MNIST/main.py --p_control 1e-3 --l2 1e-4 --methods velo_dfm_baf_ebm --lr 0.0005 --num_epochs 50  --warmup_baf 50  --enable_backward 1 --epoch_save 2    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 2000 --source data  --t 0.75 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
