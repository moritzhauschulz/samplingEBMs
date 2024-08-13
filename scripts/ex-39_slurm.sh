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
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0 --start_temp 100 --end_temp 100 --num_epochs 500 --dataset_name static_mnist --methods velo_mask_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output1.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0 --start_temp 100 --end_temp 1 --num_epochs 500 --dataset_name static_mnist --methods velo_mask_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output2.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0.5 --start_temp 100 --end_temp 100 --num_epochs 500 --dataset_name static_mnist --methods velo_mask_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output3.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0.5 --start_temp 100 --end_temp 1 --num_epochs 500 --dataset_name static_mnist --methods velo_mask_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output4.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 1 --start_temp 100 --end_temp 100 --num_epochs 500 --dataset_name static_mnist --methods velo_mask_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output5.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 1 --start_temp 100 --end_temp 1 --num_epochs 500 --dataset_name static_mnist --methods velo_mask_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output6.log 2>&1 &

python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0 --start_temp 100 --end_temp 100 --num_epochs 500 --dataset_name static_mnist --methods velo_uni_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output7.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0 --start_temp 100 --end_temp 1 --num_epochs 500 --dataset_name static_mnist --methods velo_uni_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output8.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0.5 --start_temp 100 --end_temp 100 --num_epochs 500 --dataset_name static_mnist --methods velo_uni_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output9.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 0.5 --start_temp 100 --end_temp 1 --num_epochs 500 --dataset_name static_mnist --methods velo_uni_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output10.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 1 --start_temp 100 --end_temp 100 --num_epochs 500 --dataset_name static_mnist --methods velo_uni_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output11.log 2>&1 &
python -u methods_MNIST/our_main.py --lr 0.0001 --q_weight 1 --start_temp 100 --end_temp 1 --num_epochs 500 --dataset_name static_mnist --methods velo_uni_edfs --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output12.log 2>&1 &
wait
############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime