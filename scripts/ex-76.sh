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

#below code collapses dfs after ca. 80 iterations
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 1 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source uniform --eval_on 0 > ${output_date}/output1.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 1 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source mask --eval_on 0 > ${output_date}/output2.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 1 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source data --eval_on 0 > ${output_date}/output3.log 2>&1 &

# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 1 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source uniform --eval_on 0 > ${output_date}/output4.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 1 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source mask --eval_on 0 > ${output_date}/output5.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 1 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source data --eval_on 0 > ${output_date}/output6.log 2>&1 &
# wait

#below code collapses dfs after ca. 80 iterations

# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source uniform --eval_on 0 > ${output_date}/output7.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source mask --eval_on 0 > ${output_date}/output8.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source data --eval_on 0 > ${output_date}/output9.log 2>&1 &

# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source uniform --eval_on 0 > ${output_date}/output10.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source mask --eval_on 0 > ${output_date}/output11.log 2>&1 &
# python methods_MNIST/main.py --methods velo_edfm_ebm  --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source data --eval_on 0 > ${output_date}/output12.log 2>&1 &
# wait
python methods_MNIST/main.py --methods velo_edfm_ebm  --p_control 1e-2 --l2 1e-3 --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source uniform --eval_on 0 > ${output_date}/output7.log 2>&1 &
python methods_MNIST/main.py --methods velo_edfm_ebm  --p_control 1e-2 --l2 1e-3 --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source mask --eval_on 0 > ${output_date}/output8.log 2>&1 &
python methods_MNIST/main.py --methods velo_edfm_ebm  --p_control 1e-2 --l2 1e-3 --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type linear --source data --eval_on 0 > ${output_date}/output9.log 2>&1 &

python methods_MNIST/main.py --methods velo_edfm_ebm  --p_control 1e-2 --l2 1e-3 --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source uniform --eval_on 0 > ${output_date}/output10.log 2>&1 &
python methods_MNIST/main.py --methods velo_edfm_ebm  --p_control 1e-2 --l2 1e-3 --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source mask --eval_on 0 > ${output_date}/output11.log 2>&1 &
python methods_MNIST/main.py --methods velo_edfm_ebm  --p_control 1e-2 --l2 1e-3 --batch_size 256 --dfs_per_ebm 100 --num_epoch 10 --epoch_save 1 --scheduler_type quadratic_noise --source data --eval_on 0 > ${output_date}/output12.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
