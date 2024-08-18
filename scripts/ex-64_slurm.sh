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
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type linear --source mask --num_epoch 300 --epoch_save 25 > ${output_date}/output1.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type linear --source uniform --num_epoch 300 --epoch_save 25 > ${output_date}/output2.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type linear --source data --num_epoch 300 --epoch_save 25 > ${output_date}/output3.log 2>&1 &

python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic --source mask --num_epoch 300 --epoch_save 25 > ${output_date}/output4.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output5.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output6.log 2>&1 &

python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic_noise --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output7.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic_noise --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output8.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic_noise --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output9.log 2>&1 &

python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type linear --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output10.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type linear --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output11.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type linear --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output12.log 2>&1 &


python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output13.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output14.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output15.log 2>&1 &


python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic_noise --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output16.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic_noise --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output17.log 2>&1 &
python -u methods_MNIST/our_main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic_noise --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output18.log 2>&1 &

wait

python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type linear --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output19.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type linear --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output20.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type linear --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output21.log 2>&1 &

python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type quadratic --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output22.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type quadratic --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output23.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type quadratic --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output24.log 2>&1 &

python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type quadratic_noise --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output25.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type quadratic_noise --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output26.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfm --scheduler_type quadratic_noise --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output27.log 2>&1 &

python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type linear --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output28.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type linear --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output29.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type linear --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output39.log 2>&1 &

python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type quadratic --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output31.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type quadratic --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output32.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type quadratic --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output33.log 2>&1 &

python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type quadratic_noise --source mask --epoch_save 25 --num_epoch 300 > ${output_date}/output34.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type quadratic_noise --source uniform --epoch_save 25 --num_epoch 300 > ${output_date}/output35.log 2>&1 &
python -u methods_MNIST/our_main.py --methods velo_dfs --scheduler_type quadratic_noise --source data --epoch_save 25 --num_epoch 300 > ${output_date}/output36.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
