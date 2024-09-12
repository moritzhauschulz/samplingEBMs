# Training Energy Based Models by Learning to Sample with Discrete Flows

## Abstract

Training energy-based models with contrastive divergence is a challenging problem due to the requirement to generate negative samples from the model distribution during training. This is particularly challenging in discrete domains, since conventional MCMC methods cannot exploit gradient information well and convergence is difficult to assess. Discrete flows are generative models that transform a source distribution to a data distribution through continuous time Markov chains, whose rate matrix or generating velocity is parameterized by a neural network. This work attempts to address the challenges in the training of energy based models by using discrete flows. We contribute to the literature on discrete flows, specifically that on discrete flow matching, by proposing to directly learn the generating velocity through a regression loss instead of constructing it at sampling time from learned posteriors. We further employ importance sampling techniques to adapt the method to a data-free setting where beyond the target energy model, only a proposal distribution is available. Equipped with new tools that enable discrete flows to act as learned samplers, we propose and assess three different ways of applying them in the training of energy-based models. We demonstrate that all three have the capacity to learn on relatively low-dimensional synthetic benchmark problems, but find that without the use of MCMC-based refinements, they are not competitive in higher dimensions where suitable proposal distributions are difficult to construct.

## Author

- **Moritz Elias Hauschulz** - Imperial College London

## Supervisors

- **Yingzhen Li** - Imperial College London
- **Zijing Ou** - Imperial College London


## Methods
- my methods:
    - velo_dfm (implementation of https://arxiv.org/abs/2407.15595, where no code is available)
    - velo_dfs ('EDFM' in the report)
    - velo_dfs ('CVM' in the report)
    - velo_edfs ('ECVM' in the report)
    - velo_ebm ('DFS-EBM' in the report)
    - velo_bootstrap_ebm ('DFS-EBM with bootstrapped proposal' in the report)
    - velo_baf_ebm ('DFS-EBM with heuristic MH' in the report)
    - velo_bootstrap_v2_ebm ('EBM-DFS' in the appendix of the report)
- baseline methods (not my code): 
    - dmala_ebm (https://arxiv.org/abs/2206.09914)
    - eb_gfn (https://arxiv.org/abs/2202.01361)
    - ed_ebm (https://arxiv.org/abs/2307.07595)

## Installation and Example

To reproduce the results or use the model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/moritzhauschulz/samplingEBMs.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download datasets:
    - The discrete image datasets are available here: https://github.com/jmtomczak/vae_vampprior/tree/master/datasets
    - Set appropriate paths in `methods/utils/vamp_utils.py`
    - The synthetic datasets are from the `scikit-learn` library and do not require download

4. Run example codes (here specified only for toy data):
    ```bash 
    bash example_scripts/velo_dfm.sh
    ```
    ```bash
    bash example_scripts/velo_dfs.sh
    ```
    ```bash
    bash example_scripts/velo_ebm.sh
    ```
    ```bash
    bash example_scripts/velo_bootstrap_ebm.sh
    ```
    ```bash
    bash example_scripts/velo_baf_ebm.sh
    ```
    - Output will appear in nested folders under the respective method for each run.

