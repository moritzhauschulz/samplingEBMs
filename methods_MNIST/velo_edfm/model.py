import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Dataq:
    def __init__(self, args, dataset, bernoulli_mean=None):
        assert 0 <= args.q_weight <= 1, "Q Weight parameter must be between 0 and 1."
        
        sampler =  torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=(len(dataset) // args.batch_size) *  args.batch_size, generator=None)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)

        self.loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
        self.weight = args.q_weight
        if bernoulli_mean is not None and self.weight > 0:
            self.bernoulli_dist = torch.distributions.Bernoulli(probs=bernoulli_mean)
        else:
            self.weight = 0
            print(f'Bernoulli weight set to zero.')
        self.device = args.device
        self.batch_size = args.batch_size
        self.len = len(dataset)
        
    def sample(self, empirical_samples):
        num_empirical = int((1 - self.weight) * self.batch_size)
        num_bernoulli = self.batch_size - num_empirical
        
        # Sample from empirical distribution
        if num_empirical > 0 and num_empirical < self.batch_size:
            empirical_indices = np.random.choice(self.batch_size, size=num_empirical, replace=False)
            empirical_samples = empirical_samples[empirical_indices].to(self.device)
            is_empirical = torch.ones((num_empirical,)).to(self.device)
            empirical_log_likelihood = (torch.ones((self.num_empirical,)) * torch.log(torch.tensor(1/self.batch_size))).to(self.device) + torch.log(torch.tensor(1 - self.weight)).to(self.device)
            bernoulli_samples = self.bernoulli_dist.sample((num_bernoulli,)).to(self.device)
            bernoulli_log_likelihood = self.bernoulli_dist.log_prob(bernoulli_samples).sum(dim=-1).to(self.device) + torch.log(torch.tensor(self.weight)).to(self.device)
            is_bernoulli = torch.zeros((num_bernoulli,)).to(self.device)
            combined_samples = torch.cat([empirical_samples, bernoulli_samples])
            indices = torch.randperm(combined_samples.size(0))
            combined_samples = combined_samples[indices]
            self.is_empirical = torch.cat([is_empirical,is_bernoulli])[indices]
            self.last_log_likelihood = torch.cat([empirical_log_likelihood, bernoulli_log_likelihood])[indices]
        elif num_bernoulli == 0:
            empirical_indices = np.random.choice(self.batch_size, size=num_empirical, replace=False)
            combined_samples = empirical_samples[empirical_indices]
            is_empirical = torch.ones((num_empirical,)).to(self.device)
            empirical_log_likelihood = (torch.ones((num_empirical,)) * torch.log(torch.tensor(1/self.len))).to(self.device) + torch.log(torch.tensor(1 - self.weight)).to(self.device)
            self.is_empirical = is_empirical
            self.last_log_likelihood = empirical_log_likelihood
        elif num_empirical == 0:
            combined_samples = self.bernoulli_dist.sample((num_bernoulli,)).to(self.device)
            bernoulli_log_likelihood = self.bernoulli_dist.log_prob(combined_samples).sum(dim=-1).to(self.device) + torch.log(torch.tensor(self.weight)).to(self.device)
            is_bernoulli = torch.zeros((num_bernoulli,)).to(self.device)
            self.is_empirical = is_bernoulli
            self.last_log_likelihood = bernoulli_log_likelihood

        return combined_samples
    
    def get_last_log_likelihood(self):
        return self.last_log_likelihood

    def get_last_is_empirical(self):
        return self.is_empirical

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)

def conv_transpose_3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    if stride < 0:
        return conv_transpose_3x3(in_planes, out_planes, stride=-stride)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_nonlin=True):
        super(BasicBlock, self).__init__()
        self.nonlin1 = Swish()
        self.nonlin2 = Swish()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.out_nonlin = out_nonlin

        self.shortcut_conv = None
        if stride != 1 or in_planes != self.expansion * planes:
            if stride < 0:
                self.shortcut_conv = nn.ConvTranspose2d(in_planes, self.expansion*planes,
                                                        kernel_size=1, stride=-stride,
                                                        output_padding=1, bias=True)
            else:
                self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes,
                                               kernel_size=1, stride=stride, bias=True)


    def forward(self, x):
        out = self.nonlin1(self.conv1(x))
        out = self.conv2(out)
        if self.shortcut_conv is not None:
            out_sc = self.shortcut_conv(x)
            out += out_sc
        else:
            out += x
        if self.out_nonlin:
            out = self.nonlin2(out)
        return out


class ResNetFlow(nn.Module):
    def __init__(self, n_channels, args):
        super().__init__()
        D = args.discrete_dim
        S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size

        self.x_proj_linear = nn.Sequential(
            nn.Linear(28*28, 1024),
            Swish(),
            nn.Linear(1024, 28*28)
        )
        self.t_proj_linear = nn.Sequential(
            nn.Linear(28*28, 1024),
            Swish(),
            nn.Linear(1024, 28*28)
        )
        self.input_conv = nn.Conv2d(1, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(6)]
        all = downsample + main
        self.net = nn.Sequential(*all)
        self.output_linear = nn.Linear(n_channels, D*S)

    def forward(self, xt, t):
        B, D = xt.shape

        xt = xt.float()
        xt = self.x_proj_linear(xt.view(xt.size(0), -1))
        t = self.t_proj_linear(timestep_embedding(t, 28*28))
        input = xt + t

        input = self.input_conv(input.view(input.size(0), 1, 28, 28))
        h = self.net(input)
        h = h.view(h.size(0), h.size(1), -1).mean(-1)   # mean pooling: (B, C, D) -> (B, C)

        out = self.output_linear(h)  # (B, C) -> (B, D*S)
        #out = F.relu(out)
        return out.reshape(B, D, -1)   # (B, D, S)

class EBM(nn.Module):
    def __init__(self, net, mean=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            base_dist = torch.distributions.Bernoulli(probs=self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp - bd # or +?
