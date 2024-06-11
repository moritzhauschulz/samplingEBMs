from utils.utils import annealed_importance_sampling
import torch

# scorer = lambda x: -torch.sum(torch.log(((1 - 0.75) ** x) * (0.75 ** (1-x))), dim=-1)

# latent_dim = 32
# num_samples = 1000
# num_intermediate = 10000
# num_mcmc_steps = 25

# log_partition_ratio, _, _ = annealed_importance_sampling(scorer, num_samples, num_intermediate, num_mcmc_steps, latent_dim)

# z = torch.exp(log_partition_ratio)

# print(z)

x = torch.zeros([1])

print(x)

y = x.detach()
print(y.item())

