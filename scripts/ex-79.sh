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


#lin t
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1 --epoch_save 1   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source uniform  --lin_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#rand t
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1  --epoch_save 1   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source uniform  --rand_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#fixed t=0.5
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1  --epoch_save 1   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source uniform  --t 0.75 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

#lin t
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1 --epoch_save 1   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source mask  --lin_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#rand t
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1 --epoch_save 1    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source mask  --rand_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#fixed t=0.5
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1  --epoch_save 1   --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 1000 --source mask  --t 0.75 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

#lin t
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1 --epoch_save 1    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 2000 --source data  --lin_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#rand t
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1 --epoch_save 1    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 2000 --source data --rand_t 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))

#fixed t=0.5
python -u  methods_MNIST/main.py --methods velo_dfm_baf_ebm --with_mh 0 --num_epochs 25  --warmup_baf 25  --enable_backward 1 --epoch_save 1    --eval_on 0 --dfs_per_ebm 1 --dfs_warmup_iter 2000 --source data  --t 0.75 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
