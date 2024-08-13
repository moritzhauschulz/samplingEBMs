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
python -u methods_MNIST/our_main.py --optimal_temp 0 --delta_t 0.05 --optimal_temp_use_median 1 --lr 0.0001 --q_weight 1 --start_temp 100 --end_temp 100 --num_epochs 300 --dataset_name static_mnist --methods velo_efm --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output1.log 2>&1 &
python -u methods_MNIST/our_main.py --optimal_temp 1 --delta_t 0.05 --optimal_temp_use_median 1 --lr 0.0001 --q_weight 1 --start_temp 100 --end_temp 100 --num_epochs 300 --dataset_name static_mnist --methods velo_efm --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output2.log 2>&1 &
python -u methods_MNIST/our_main.py --optimal_temp 1 --delta_t 0.05 --optimal_temp_use_median 0 --lr 0.0001 --q_weight 1 --start_temp 100 --end_temp 100 --num_epochs 300 --dataset_name static_mnist --methods velo_efm --norm_by_max 1 --norm_by_sum 0 > ${output_date}/output3.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
