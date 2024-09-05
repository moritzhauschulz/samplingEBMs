#!/bin/bash
#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

# Extract the file name without the extension
script_path="$0"
filename=$(basename "$script_path" .pt)


# Create the new directory name by appending '_output'
output="./${filename}_output/"

# Create the new directory in the current directory
mkdir -p $output

# Ensure correct PATH to your virtual environment
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate


# Move to the parent directory
cd ..

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output_date="./scripts/${filename}_output/${CURRENT_DATE}/"
mkdir -p $output_date

echo "Starting job ${filename} at $(date)"

# Initialize the counter
counter=1

#SLURM HEADER END
#########################################################

python -u eval_ais.py \
    --dataset_name dynamic_mnist \
    --eval_sampler dmala \
    --eval_step_size 0.1 \
    --sampling_steps 40 \
    --model resnet-64 \
    --buffer_size 10000 \
    --n_iters 300000 \
    --base_dist \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --ema \
    --viz_every 1000 \
    --save_dir ./figs/ebm_dynamic_mnist \
    > ${output_date}/output.log 2>&1;

echo "Job finished at $(date)"

uptime