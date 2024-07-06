log_dir="./run_baselines_main_temp_logs"
mkdir -p $log_dir

# run EB-GFN on moons data
python baselines_main.py --data_name moons --n_iters 100 --eval_every 10 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1 --final_ais_samples 10 --intermediate_ais_samples 10 --final_ais_num_steps 10 --intermediate_ais_num_steps 10

