#baseline EB-GFN on different toy datasets
log_dir="./run_our_main_logs"
mkdir -p $log_dir

#our DFS-EBM 1;1

#2spirals
python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001  #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# #2spirals - with warmup
# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 --warmup_k 1e5 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 --warmup_k 1e5 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 --warmup_k 1e5 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 --warmup_k 1e5 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 --warmup_k 1e5 #>  $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 --warmup_k 1e5 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 --warmup_k 1e5 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 --warmup_k 1e5 #> $log_dir/output1.log 2>&1 &


# #8gaussians
# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &



# #circles
# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# #moons
# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &


# #pinwheel
# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &


# #swissroll
# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &


# #checkerboard
# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.01 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 10 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 100 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --eval_every 2000 --num_epochs 100000 --surrogate_iter_per_epoch 1000 --ebm_iter_per_epoch 1 --batch_size 128 --delta_t 0.001 --lr 0.001 #> $log_dir/output1.log 2>&1 &

# #after this, run back and forth with 100000 