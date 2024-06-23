#baseline EB-GFN on different toy datasets
log_dir="./run_baselines_main_logs"
mkdir -p $log_dir

#baseline with mh acceptance

#2spirals
python baselines_main.py --data_name 2spirals --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1 #> $log_dir/output1.log 2>&1 &

python baselines_main.py --data_name 2spirals --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name 8gaussians --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name 8gaussians --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name circles --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name circles --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name moons --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name moons --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name pinwheel --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name pinwheel --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name swissroll --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name swissroll --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name checkerboard --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &

# python baselines_main.py --data_name checkerboard --n_iters 100000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 2000 --eval_every 2000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 0 #> $log_dir/output1.log 2>&1 &


