#baseline EB-GFN on different toy datasets

#2spirals
python baselines_main.py --data_name 2spirals --n_iters 1000000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1

#8gaussians
python baselines_main.py --data_name 8gaussians --n_iters 1000000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1

#circles
python baselines_main.py --data_name circles --n_iters 1000000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1

#moons
python baselines_main.py --data_name moons --n_iters 1000000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1

#pinwheel
python baselines_main.py --data_name pinwheel --n_iters 1000000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1

#swissroll
python baselines_main.py --data_name swissroll --n_iters 1000000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1

#checkerboard
python baselines_main.py --data_name checkerboard --n_iters 1000000 --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 1000 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1
