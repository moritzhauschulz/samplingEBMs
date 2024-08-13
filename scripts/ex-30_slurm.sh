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
python -u methods_toy/our_main.py --data_name 2spirals --methods velo_mask_dfs_ce_2 --pretrained_ebm methods_toy/ed_ebm/experiments/2spirals/2spirals_2/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 5000 --num_epochs 500000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001  > ${output_date}/output1.log 2>&1 &
python -u methods_toy/our_main.py --data_name checkerboard --methods velo_mask_dfs_ce_2 --pretrained_ebm methods_toy/ed_ebm/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 5000 --num_epochs 500000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001 > ${output_date}/output2.log 2>&1 &
wait

python -u methods_toy/our_main.py --data_name pinwheel --methods velo_mask_dfs_ce_2 --pretrained_ebm methods_toy/ed_ebm/experiments/pinwheel/pinwheel_0/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 5000 --num_epochs 500000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001  > ${output_date}/output3.log 2>&1 &
python -u methods_toy/our_main.py --data_name swissroll --methods velo_mask_dfs_ce_2 --pretrained_ebm methods_toy/ed_ebm/experiments/swissroll/swissroll_0/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 5000 --num_epochs 500000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001 > ${output_date}/output4.log 2>&1 &
wait

python -u methods_toy/our_main.py --data_name moons --methods velo_mask_dfs_ce_2 --pretrained_ebm methods_toy/ed_ebm/experiments/moons/moons_8/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 5000 --num_epochs 500000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001  > ${output_date}/output5.log 2>&1 &
python -u methods_toy/our_main.py --data_name 8gaussians --methods velo_mask_dfs_ce_2 --pretrained_ebm methods_toy/ed_ebm/experiments/8gaussians/8gaussians_0/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 5000 --num_epochs 500000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001  > ${output_date}/output6.log 2>&1 &
wait

python -u methods_toy/our_main.py --data_name circles --methods velo_mask_dfs_ce_2 --pretrained_ebm methods_toy/ed_ebm/experiments/circles/circles_0/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 5000 --num_epochs 500000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001  > ${output_date}/output7.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime