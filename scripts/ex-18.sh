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
python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/2spirals/2spirals_2/ckpts/model_100000.pt --data_name 2spirals --n_iters 500000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 5000  --plot_every 5000 --eval_every 20000 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_date}output1.log 2>&1 &
python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/checkerboard/checkerboard_0/ckpts/model_100000.pt --data_name checkerboard --n_iters 500000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 5000  --plot_every 5000 --eval_every 20000 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_date}output2.log 2>&1 &
python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/pinwheel/pinwheel_0/ckpts/model_100000.pt --data_name pinwheel --n_iters 500000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 5000  --plot_every 5000 --eval_every 20000 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_date}output3.log 2>&1 &
python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/swissroll/swissroll_0/ckpts/model_100000.pt --data_name swissroll --n_iters 500000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 5000  --plot_every 5000 --eval_every 20000 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_date}output4.log 2>&1 &
wait

python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/moons/moons_8/ckpts/model_100000.pt --data_name moons --n_iters 500000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 5000  --plot_every 5000 --eval_every 20000 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_date}output5.log 2>&1 &
python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/8gaussians/8gaussians_0/ckpts/model_100000.pt --data_name 8gaussians --n_iters 500000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 5000  --plot_every 5000 --eval_every 20000 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_date}output6.log 2>&1 &
python baselines_main.py --method gfn --pretrained_ebm methods/ed_ebm/experiments/circles/circles_0/ckpts/model_100000.pt --data_name circles --n_iters 500000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 5000  --plot_every 5000 --eval_every 20000 --glr 1e-3 --zlr 1 --rand_coef 0 > ${output_date}output7.log 2>&1 &
wait

############# SPECIFY JOB ABOVE ################
echo "Job finished at $(date)"

uptime
