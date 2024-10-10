#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=40G 
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

JOB_BASE_NAME="${SLURM_JOB_NAME%.sh}"

# Ensure correct PATH to your virtual environment
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate


# Move to the parent directory
cd ..

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output_date="./final_scripts/output/${JOB_BASE_NAME}_output/${SLURM_JOB_ID}_${CURRENT_DATE}/"
mkdir -p $output_date

echo $output

# Redirect output and error to the desired files
exec >"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.out" 2>"${output_date}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.err"

echo "Starting job ${JOB_BASE_NAME} with ID $SLURM_JOB_ID"
echo "Job started at $(date)"

# Initialize the counter
counter=1

############# SPECIFY JOB BELOW ################

python -u methods/main.py --methods eb_gfn --dataset_name 2spirals --ebm_lr 0.001 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 0.001 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --num_epochs 200000 --epoch_save 10000 --eval_every 25000 --with_mh 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --methods eb_gfn --dataset_name checkerboard --ebm_lr 0.001 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 0.001 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --num_epochs 200000 --epoch_save 10000 --eval_every 25000 --with_mh 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --methods eb_gfn --dataset_name circles --ebm_lr 0.001 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 0.001 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --num_epochs 200000 --epoch_save 10000 --eval_every 25000 --with_mh 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --methods eb_gfn --dataset_name swissroll --ebm_lr 0.001 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 0.001 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --num_epochs 200000 --epoch_save 10000 --eval_every 25000 --with_mh 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait

python -u methods/main.py --methods eb_gfn --dataset_name moons --ebm_lr 0.001 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 0.001 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --num_epochs 200000 --epoch_save 10000 --eval_every 25000 --with_mh 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --methods eb_gfn --dataset_name pinwheel --ebm_lr 0.001 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 0.001 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --num_epochs 200000 --epoch_save 10000 --eval_every 25000 --with_mh 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
python -u methods/main.py --methods eb_gfn --dataset_name 8gaussians --ebm_lr 0.001 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 0.001 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --num_epochs 200000 --epoch_save 10000 --eval_every 25000 --with_mh 1 > ${output_date}/output${counter}.log 2>&1 & ((counter++))
wait


############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
