# python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 100 --num_epochs 10 --batch_size 128 --delta_t 0.01

# run interleaved EBM training with discrete flow sampler on pinwheel
python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 1000 --num_epochs 100000 --batch_size 128 --delta_t 0.01 --lr 0.001

