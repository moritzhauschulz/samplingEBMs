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

############# SPECIFY JOB BELOW ################
python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.00001 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source data  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.00001 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source uniform  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

# python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.0001 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source data  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.0001 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source uniform  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# wait

# python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.001 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source data  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.001 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source uniform  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# wait 

# python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.01 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source data  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# python -u  methods_MNIST/main.py --methods velo_dfm --lr 0.01 --num_epochs 25 --enable_backward 1 --epoch_save 1 --source uniform  > ${output_date}/output${counter}.log 2>&1 & ((counter++))
# wait 

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
