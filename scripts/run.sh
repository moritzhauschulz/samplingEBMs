
# train flow model with conditional distribution modelling on Binary dataset
python main.py --data_name moons --methods punidb --gpu 0 --vocab_size 2
# train flow model with conditional distribution modelling on Binary dataset (no noise!)
python main.py --data_name moons --methods punidb --gpu 0 --vocab_size 2
# train flow model with conditional distribution modelling on Categorical dataset
python main.py --data_name moons --methods punidb --gpu 0 --vocab_size 5

# train flow model with rate matrix modelling on Binary dataset
python main.py --data_name moons --methods runidb --gpu 0 --vocab_size 2
# train flow model with conditional distribution modelling on Categorical dataset
python main.py --data_name moons --methods runidb --gpu 0 --vocab_size 5

# train energy discrepancy on Binary dataset
python main.py --data_name moons --methods ed_ebm --gpu 0 --vocab_size 2
# train flow model with rate matrix on known energy without dataset
python main.py --data_name moons --methods ebm_runidb --gpu 0 --vocab_size 2

# train contrastive divergence on Binary dataset
python main.py --data_name moons --methods cd_ebm --gpu 0 --vocab_size 2 --epoch_save 5

# run interleaved EBM training with discrete flow sampler on moons
python main.py --data_name moons --methods cd_runi_inter --gpu 0 --vocab_size 2 --epoch_save 25 --batch_size 128 --delta_t 0.01

