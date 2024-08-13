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
        
        # Sample from Bernoulli distribution
        bernoulli_samples = self.bernoulli_dist.sample((num_bernoulli,))
        
        # Combine the samples
        combined_samples = torch.cat([empirical_samples, bernoulli_samples])
        indices = torch.randperm(combined_samples.size(0))
        combined_samples = combined_samples[indices]
        
        return combined_samples
    
    def empirical_likelihood(self, samples):
        samples = samples.to(self.device)
        matches = (self.empirical_samples.unsqueeze(0) == samples.unsqueeze(1)).all(dim=-1).float()
        count = matches.sum(dim=1)  # Count the matches for each sample in the batch
        return count / len(self.empirical_samples)
    
    def bernoulli_likelihood(self, sample):
        sample_tensor = sample.to(self.device)
        return torch.exp(self.bernoulli_dist.log_prob(sample_tensor).sum(dim=-1))
    
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
        super(MLP, self).__init__()
        self.act_last = act_last
        self.nonlinearity = nonlinearity
        self.input_dim = input_dim
        self.bn = bn

        if isinstance(hidden_dims, str):
            hidden_dims = list(map(int, hidden_dims.split("-")))
        assert len(hidden_dims)
        hidden_dims = [input_dim] + hidden_dims
        self.output_size = hidden_dims[-1]

        list_layers = []

        for i in range(1, len(hidden_dims)):
            list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i + 1 < len(hidden_dims):  # not the last layer
                if self.bn:
                    bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
                    list_layers.append(bnorm_layer)
                list_layers.append(NONLINEARITIES[self.nonlinearity])
                if dropout > 0:
                    list_layers.append(nn.Dropout(dropout))
            else:
                if act_last is not None:
                    list_layers.append(NONLINEARITIES[act_last])

        self.main = nn.Sequential(*list_layers)

    def forward(self, z):
        x = self.main(z)
        return x

class MLPScore(nn.Module):
    def __init__(self, input_dim, hidden_dims, scale=1.0, nonlinearity='swish', act_last=None, bn=False, dropout=-1, bound=-1):
        super(MLPScore, self).__init__()
        self.scale = scale
        self.bound = bound
        self.mlp = MLP(input_dim, hidden_dims, nonlinearity, act_last, bn, dropout)

    def forward(self, z):
        raw_score = self.mlp(z.float() / self.scale)
        if self.bound > 0:
            raw_score = torch.clamp(raw_score, min=-self.bound, max=self.bound)
        return raw_score

class EBM(nn.Module):
    def __init__(self, net, mean=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def update_mean(self, mean):
        self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        '''
        we define p(x) = exp(-f(x)) / Z, the output of net is f(x)
        '''
        if self.mean is None:
            bd = 0.
        else:
            base_dist = torch.distributions.Bernoulli(probs=self.mean)
            bd = base_dist.log_prob((x > 0.5).float()).sum(-1)

        logp = self.net(x).squeeze()
        return logp - bd
