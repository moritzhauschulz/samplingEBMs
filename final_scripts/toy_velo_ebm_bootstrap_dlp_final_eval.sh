
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

python -u methods/eval_ais.py \
    --ebm_model methods/velo_bootstrap_ebm/experiments/static_mnist/static_mnist_54/ckpts/best_ema_ebm_ckpt_static_mnist_dmala_0.1_45000.pt \
    --sampler dmala \
    --eval_step_size 0.2 \
    --sampling_steps 40 \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --viz_every 10;

python -u methods/eval_ais.py \
    --ebm_model methods/velo_bootstrap_ebm/experiments/static_mnist/static_mnist_54/ckpts/best_ema_ebm_ckpt_static_mnist_dmala_0.1_40000.pt \
    --sampler dmala \
    --eval_step_size 0.2 \
    --sampling_steps 40 \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --viz_every 10;


# #dynamic mnist
# python -u methods/eval_ais.py \
#     --ebm_model methods/velo_bootstrap_ebm/experiments/dynamic_mnist/dynamic_mnist_0/ckpts/best_ema_ebm_ckpt_dynamic_mnist_dmala_0.1_45000.pt \
#     --sampler dmala \
#     --eval_step_size 0.1 \
#     --sampling_steps 40 \
#     --n_samples 500 \
#     --eval_sampling_steps 300000 \
#     --viz_every 10;

# #omniglot
# python -u methods/eval_ais.py \
#     --ebm_model methods/velo_bootstrap_ebm/experiments/omniglot/omniglot_0/ckpts/best_ema_ebm_ckpt_omniglot_dmala_0.1_30000.pt \
#     --sampler dmala \
#     --eval_step_size 0.1 \
#     --sampling_steps 40 \
#     --n_samples 500 \
#     --eval_sampling_steps 300000 \
#     --viz_every 10;

# #caltech
# python -u methods/eval_ais.py \
#     --ebm_model methods/velo_bootstrap_ebm/experiments/caltech/caltech_0/ckpts/best_ema_ebm_ckpt_caltech_dmala_0.1_15000.pt \
#     --sampler dmala \
#     --eval_step_size 0.1 \
#     --sampling_steps 40 \
#     --n_samples 500 \
#     --eval_sampling_steps 300000 \
#     --viz_every 10;


############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
