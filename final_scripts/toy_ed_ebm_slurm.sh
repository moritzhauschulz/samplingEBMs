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


############# SPECIFY JOB BELOW ################
python methods/main.py --dataset_name swissroll --methods ed_ebm --gpu 0 --vocab_size 2 --lr 0.002 --num_epochs 100000 --eval_every 100000 > ${output_date}/output1.log 2>&1 &
python methods/main.py --dataset_name circles --methods ed_ebm --gpu 0 --vocab_size 2 --lr 0.002 --num_epochs 100000 --eval_every 100000 > ${output_date}/output2.log 2>&1 &
python methods/main.py --dataset_name moons --methods ed_ebm --gpu 0 --vocab_size 2 --lr 0.002 --num_epochs 100000 --eval_every 100000 > ${output_date}/output3.log 2>&1 &
python methods/main.py --dataset_name 8gaussians --methods ed_ebm --gpu 0 --vocab_size 2 --lr 0.002 --num_epochs 100000 --eval_every 100000 > ${output_date}/output4.log 2>&1 &
python methods/main.py --dataset_name pinwheel --methods ed_ebm --gpu 0 --vocab_size 2 --lr 0.002 --num_epochs 100000 --eval_every 100000 > ${output_date}/output5.log 2>&1 &
python methods/main.py --dataset_name 2spirals --methods ed_ebm --gpu 0 --vocab_size 2 --lr 0.002 --num_epochs 100000 --eval_every 100000 > ${output_date}/output6.log 2>&1 &
python methods/main.py --dataset_name checkerboard --methods ed_ebm --gpu 0 --vocab_size 2 --lr 0.002 --num_epochs 100000 --eval_every 100000 > ${output_date}/output7.log 2>&1 &
wait
############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
