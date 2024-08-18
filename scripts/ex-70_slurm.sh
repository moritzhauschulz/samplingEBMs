#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

JOB_BASE_NAME="${SLURM_JOB_NAME%.sh}"

log_dir="./${JOB_BASE_NAME}_output"
mkdir -p $log_dir

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

############# SPECIFY JOB BELOW ################

#same as 68, just recycling dfs sample by default...
python -u methods_MNIST/our_main.py --alpha 0 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --epoch_save 1 --num_epochs 25 > ${output_date}/output1.log 2>&1 &
python -u methods_MNIST/our_main.py --alpha 0 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output2.log 2>&1 &
python -u methods_MNIST/our_main.py --alpha 0 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --optimal_temp 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output3.log 2>&1 &

python -u methods_MNIST/our_main.py --alpha 0.1 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --epoch_save 1 --num_epochs 25 > ${output_date}/output4.log 2>&1 &
python -u methods_MNIST/our_main.py --alpha 0.1 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output5.log 2>&1 &
python -u methods_MNIST/our_main.py --alpha 0.1 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --optimal_temp 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output6.log 2>&1 &

python -u methods_MNIST/our_main.py --recycle_dfs_sample 0 --alpha 0 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --epoch_save 1 --num_epochs 25 > ${output_date}/output7.log 2>&1 &
python -u methods_MNIST/our_main.py --recycle_dfs_sample 0 --alpha 0 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output8.log 2>&1 &
python -u methods_MNIST/our_main.py --recycle_dfs_sample 0 --alpha 0 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --optimal_temp 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output9.log 2>&1 &

python -u methods_MNIST/our_main.py --recycle_dfs_sample 0 --alpha 0.1 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --epoch_save 1 --num_epochs 25 > ${output_date}/output10.log 2>&1 &
python -u methods_MNIST/our_main.py --recycle_dfs_sample 0 --alpha 0.1 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output11.log 2>&1 &
python -u methods_MNIST/our_main.py --recycle_dfs_sample 0 --alpha 0.1 --itr_save 100 --methods velo_efm_ebm_bootstrap_2 --scheduler_type linear --source uniform --optional_step 1 --optimal_temp 1 --epoch_save 1 --num_epochs 25 > ${output_date}/output12.log 2>&1 &

wait



############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
