# run interleaved EBM training with discrete flow sampler on moons
python our_main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --batch_size 128 --delta_t 0.01

# run interleaved EBM training with discrete flow sampler on pinwheel
python our_main.py --data_name pinwheel --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --batch_size 128 --delta_t 0.01

# run interleaved EBM training with discrete flow sampler on checkerboard
python our_main.py --data_name checkerboard --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --batch_size 128 --delta_t 0.01
