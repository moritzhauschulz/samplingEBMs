#!/bin/bash

# Move to the parent directory
cd ..


#########################################################
#SLURM HEADER START

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

JOB_BASE_NAME="${SLURM_JOB_NAME%.sh}"

# Ensure correct PATH to your virtual environment
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output_date="./scripts/${JOB_BASE_NAME}_output/${SLURM_JOB_ID}_${CURRENT_DATE}/"
mkdir -p $output_date

echo $output

# Redirect output and error to the desired files
exec >"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.out" 2>"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.err"

echo "Starting job ${JOB_BASE_NAME} with ID $SLURM_JOB_ID"
echo "Job started at $(date)"

#SLURM HEADER END
#########################################################


python -u pcd_ebm_ema.py \
    --dataset_name static_mnist \
    --sampler dmala \
    --step_size 0.1 \
    --sampling_steps 40 \
    --viz_every 100 \
    --model resnet-64 \
    --print_every 10 \
    --lr .0001 \
    --warmup_iters 10000 \
    --buffer_size 10000 \
    --n_iters 50000 \
    --buffer_init mean \
    --base_dist \
    --reinit_freq 0.0 \
    --eval_every 5000 \
    --eval_sampling_steps 10000 \
    --save_dir ./figs/ebm_stati_MNIST \
    > ${output_date}/output.log;

echo "Job finished at $(date)"

uptime