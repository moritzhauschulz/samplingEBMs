#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifications
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifications - please replace <your_username> with your college login name or email address
#SBATCH --output=/dev/null # Temporarily send output to /dev/null
#SBATCH --error=/dev/null # Temporarily send error to /dev/null

JOB_BASE_NAME="${SLURM_JOB_NAME%.sh}"

log_dir="./${JOB_BASE_NAME}_output"
mkdir -p $log_dir

# Ensure correct PATH to your virtual environment
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin:$PATH
source /vol/bitbucket/${USER}/samplingEBMs/.venv/bin/activate


# Move to the parent directory
cd ..

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

output="./scripts/${JOB_BASE_NAME}_output/${SLURM_JOB_ID}_${CURRENT_DATE}/"
output_files="${output}/${SLURM_JOB_ID}"
mkdir -p $output

echo $output

# Redirect output and error to the desired files
exec >"${output}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.out" 2>"${output}/${JOB_BASE_NAME}_${SLURM_JOB_ID}.err"


echo "Starting job ${JOB_BASE_NAME} with ID $SLURM_JOB_ID"
echo "Job started at $(date)"

############# SPECIFY JOB BELOW ################

python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/2spirals/2spirals_2/ckpts/model_100000.pt --data_name 2spirals --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2500 --eval_every 2500 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_files}_output1.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################

echo "Job finished at $(date)"

uptime
