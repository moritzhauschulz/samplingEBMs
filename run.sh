
# train flow model with conditional distribution modelling on Binary dataset
python main.py --data_name moons --methods punidb --gpu 0 --vocab_size 2
# train flow model with conditional distribution modelling on Binary dataset (no noise!)
python main.py --data_name moons --methods punidb --gpu 0 --vocab_size 2 --noise 0
# train flow model with conditional distribution modelling on Categorical dataset
python main.py --data_name moons --methods punidb --gpu 0 --vocab_size 5

# train flow model with rate matrix modelling on Binary dataset
python main.py --data_name moons --methods runidb --gpu 0 --vocab_size 2
# train flow model with conditional distribution modelling on Categorical dataset
python main.py --data_name moons --methods runidb --gpu 0 --vocab_size 5

# train energy discrepancy on Binary dataset
#python main.py --data_name moons --methods ed --gpu 0 --vocab_size 2
# train flow model with rate matrix on known energy without dataset
python main.py --data_name moons --methods ebm_runidb --gpu 0 --vocab_size 2