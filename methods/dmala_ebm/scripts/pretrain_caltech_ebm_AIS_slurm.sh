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

python -u eval_ais.py \
    --dataset_name caltech \
    --eval_sampler dmala \
    --eval_step_size 0.1 \
    --sampling_steps 40 \
    --model resnet-64 \
    --buffer_size 1000 \
    --n_iters 300000 \
    --base_dist \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --ema \
    --viz_every 1000 \
    --save_dir ./figs/ebm_caltech \
    > ${output_date}/output.log;


echo "Job finished at $(date)"

uptime