import torch
import torch.nn as nn
import numpy as np

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class MixtureModel:
    def __init__(self, empirical_samples, bernoulli_mean, weight, device='cpu'):
        assert 0 <= weight <= 1, "Weight parameter must be between 0 and 1."
        
        self.empirical_samples = torch.tensor(empirical_samples, device=device)
        self.bernoulli_dist = torch.distributions.Bernoulli(probs=torch.tensor(bernoulli_mean, device=device) * (1. - 2 * 1e-2) + 1e-2)
        self.weight = weight
        self.device = device
        
    def sample(self, num_samples):
        num_empirical = int((1 - self.weight) * num_samples)
        num_bernoulli = num_samples - num_empirical
        
        # Sample from empirical distribution
        empirical_indices = np.random.choice(self.empirical_samples.shape[0], size=num_empirical, replace=True)
        empirical_samples = self.empirical_samples[empirical_indices]
        print(empirical_samples.shape)
        
        # Sample from Bernoulli distribution
        bernoulli_samples = self.bernoulli_dist.sample((num_bernoulli,))
        print(bernoulli_samples.shape)
        
        # Combine the samples
        combined_samples = torch.cat([empirical_samples, bernoulli_samples])
        indices = torch.randperm(combined_samples.size(0))
        combined_samples = combined_samples[indices]
        
        return combined_samples
    
    def empirical_likelihood(self, sample):
        count = torch.sum(torch.all(self.empirical_samples == sample, dim=1)).item()
        return count / len(self.empirical_samples)
    
    def bernoulli_likelihood(self, sample):
        sample_tensor = sample.to(self.device)
        return torch.exp(self.bernoulli_dist.log_prob(sample_tensor)).prod().item()
    
    def likelihood(self, sample):
        sample_tensor = sample.to(self.device)
        empirical_prob = self.empirical_likelihood(sample_tensor)
        bernoulli_prob = self.bernoulli_likelihood(sample_tensor)
        return torch.tensor((1 - self.weight) * empirical_prob + self.weight * bernoulli_prob, device=self.device)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class MLPModel(nn.Module):
    def __init__(self, args):
        super(MLPModel, self).__init__()

        D = args.discrete_dim
        S = args.vocab_size
        self.embedding = nn.Embedding(args.vocab_size, 16)
        self.net = nn.Sequential(
            nn.Linear((16+1) * D, 1024),
            Swish(),
            nn.Linear(1024, 1024),
            Swish(),
            nn.Linear(1024, 1024),
            Swish(),
            nn.Linear(1024, S * D),
            nn.ReLU()
        )
    
    def forward(self, x, t):
        B, D = x.shape

        x_emb = self.embedding(x)   # (B, D, 16)
        net_input = torch.cat([x_emb, t[:, None, None].repeat(1, D, 1)], dim=-1).reshape(B, -1) # (B, D * 17)
        return self.net(net_input).reshape(B, D, -1)   # (B, D, S)
