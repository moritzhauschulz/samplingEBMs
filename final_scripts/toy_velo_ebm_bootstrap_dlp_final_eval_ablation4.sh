
#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

# Extract the file name without the extension
script_path="$0"
filename=$(basename "$script_path" .pt)

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
#static mnist
python -u methods/eval_ais.py \
    --ebm_model methods/velo_bootstrap_ebm/experiments/static_mnist/static_mnist_56/ckpts/best_ema_ebm_ckpt_static_mnist_dmala_0.1_35000.pt \
    --sampler dmala \
    --eval_step_size 0.1 \
    --sampling_steps 40 \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --viz_every 10;

#static mnist
python -u methods/eval_ais.py \
    --ebm_model methods/velo_bootstrap_ebm/experiments/static_mnist/static_mnist_57/ckpts/best_ema_ebm_ckpt_static_mnist_dmala_0.1_45000.pt \
    --sampler dmala \
    --eval_step_size 0.1 \
    --sampling_steps 40 \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --viz_every 10;

#static mnist
python -u methods/eval_ais.py \
    --ebm_model methods/velo_bootstrap_ebm/experiments/static_mnist/static_mnist_58/ckpts/best_ema_ebm_ckpt_static_mnist_dmala_0.1_45000.pt \
    --sampler dmala \
    --eval_step_size 0.1 \
    --sampling_steps 40 \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --viz_every 10;

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
