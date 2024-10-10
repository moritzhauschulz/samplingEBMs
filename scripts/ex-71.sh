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

############# SPECIFY JOB BELOW ################
python -u methods_MNIST/main.py --delta_t 0.05 --methods velo_dfm --scheduler_type linear --source omniglot --num_epoch 300 --epoch_save 50 > ${output_date}/output1.log 2>&1 &
python -u methods_MNIST/main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic --source omniglot --epoch_save 50 --num_epoch 300 > ${output_date}/output2.log 2>&1 &
python -u methods_MNIST/main.py --delta_t 0.05 --methods velo_dfm --scheduler_type quadratic_noise --source omniglot --epoch_save 50 --num_epoch 300 > ${output_date}/output3.log 2>&1 &
wait 

python -u methods_MNIST/main.py --delta_t 0.05 --methods velo_dfs --scheduler_type linear --source omniglot --num_epoch 300 --epoch_save 50 > ${output_date}/output4.log 2>&1 &
python -u methods_MNIST/main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic --source omniglot --epoch_save 50 --num_epoch 300 > ${output_date}/output5.log 2>&1 &
python -u methods_MNIST/main.py --delta_t 0.05 --methods velo_dfs --scheduler_type quadratic_noise --source omniglot --epoch_save 50 --num_epoch 300 > ${output_date}/output6.log 2>&1 &
wait 




############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
