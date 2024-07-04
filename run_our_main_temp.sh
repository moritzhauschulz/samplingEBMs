# log_dir="./run_our_main_temp_logs"
# mkdir -p $log_dir

# run interleaved EBM training with discrete flow sampler on pinwheel
# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 250 --num_epochs 10000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 --lin_k 1 --warmup_k 5000 --with_mh 1

python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2 --num_epochs 4 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 --lin_k 0 --warmup_k 4 --with_mh 1 --final_ais_samples 1 --intermediate_ais_samples 1 --final_ais_num_steps 1 --intermediate_ais_num_steps 1 --gibbs_num_rounds 1

