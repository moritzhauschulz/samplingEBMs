#our DFS-EBM on different toy datasets

#2spirals
python our_main.py --data_name 2spirals --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --num_epochs 100000  --batch_size 128 --delta_t 0.01 --lr 0.001

#8gaussians
python our_main.py --data_name 8gaussians --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --num_epochs 100000  --batch_size 128 --delta_t 0.01 --lr 0.001

#circles
python our_main.py --data_name circles --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --num_epochs 100000  --batch_size 128 --delta_t 0.01 --lr 0.001

#moons
python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --num_epochs 100000  --batch_size 128 --delta_t 0.01 --lr 0.001

#pinwheel
python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --num_epochs 100000  --batch_size 128 --delta_t 0.01 --lr 0.001

#swissroll
python our_main.py --data_name swissroll --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --num_epochs 100000  --batch_size 128 --delta_t 0.01 --lr 0.001

#checkerboard
python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --num_epochs 100000  --batch_size 128 --delta_t 0.01 --lr 0.001
