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

output_date="./scripts/${filename}_output/${CURRENT_DATE}"
mkdir -p $output_date

echo "Starting job ${filename} at $(date)"

############# SPECIFY JOB BELOW ################
python baselines_main.py --data_name 2spirals --n_iters 100000 --gfn_iter_per_epoch 1 --ebm_iter_per_epoch 0 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --plot_every 1000 --eval_every 10000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 0 --with_mh 0  > $output_date/output1.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime