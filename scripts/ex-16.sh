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

############# SPECIFY JOB BELOW ################
python -u our_main.py --data_name 2spirals --methods mask_dfs_5 --pretrained_ebm methods/ed_ebm/experiments/2spirals/2spirals_2/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 2500 --num_epochs 1000000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001 --eta 0 > ${output_date}/output1.log 2>&1 &
python -u our_main.py --data_name 2spirals --methods mask_dfs_5 --pretrained_ebm methods/ed_ebm/experiments/2spirals/2spirals_2/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 2500 --num_epochs 1000000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001 --eta 1 > ${output_date}/output2.log 2>&1 &
python -u our_main.py --data_name 2spirals --methods mask_dfs_5 --pretrained_ebm methods/ed_ebm/experiments/2spirals/2spirals_2/ckpts/model_100000.pt --gpu 0 --vocab_size 2 --eval_every 100000 --plot_every 2500 --num_epochs 1000000 --batch_size 128 --delta_t 0.01 --dfs_lr 0.0001 --eta 5 > ${output_date}/output3.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
