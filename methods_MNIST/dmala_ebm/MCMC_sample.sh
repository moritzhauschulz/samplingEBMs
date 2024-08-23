#!/bin/bash

python -u MCMC_sample.py \
    --dataset_name static_mnist \
    --sampler dula \
    --step_size 0.1 \
    --sampling_steps 40 \
    --viz_every 100 \
    --model resnet-64 \
    --print_every 10 \
    --lr .0001 \
    --warmup_iters 10000 \
    --buffer_size 10000 \
    --n_iters 50000 \
    --buffer_init mean \
    --base_dist \
    --reinit_freq 0.0 \
    --eval_every 5000 \
    --eval_sampling_steps 10000
    --ckpt_path methods_MNIST/dl_ebm/figs/ebm/ckpt_static_mnist_dula_0.1.pt;