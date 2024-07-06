#baseline EB-GFN on different toy datasets
log_dir="./run_our_main_logs"
mkdir -p $log_dir

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=meh23@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/samplingEBMs/.venv/bin/:$PATH
source activate
python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 1 --num_epochs 1 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --ebm_lr 0.01 --dfs_lr 0.001 --lin_k 0 --warmup_k 4 --with_mh 1 --final_ais_samples 1 --intermediate_ais_samples 1 --final_ais_num_steps 1 --intermediate_ais_num_steps 1 --gibbs_num_rounds 1


uptime