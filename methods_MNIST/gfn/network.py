import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def make_mlp(l, act=nn.LeakyReLU(), tail=[], with_bn=False):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + (([nn.BatchNorm1d(o), act] if with_bn else [act]) if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []
    ) + tail))
    return net


def mlp_ebm(nin, nint=256, nout=1):
    return nn.Sequential(
        nn.Linear(nin, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nout),
    )

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

# class EnergyModel(T.nn.Module):

#     def __init__(self, s, mid_size):
#         super(EnergyModel, self).__init__()

#         self.m = T.nn.Sequential(T.nn.Linear(s, mid_size),
#                                  T.nn.ELU(),
#                                  T.nn.Linear(mid_size, mid_size),
#                                  T.nn.ELU(),
#                                  T.nn.Linear(mid_size, mid_size),
#                                  T.nn.ELU(),
#                                  T.nn.Linear(mid_size, 1))

#     def forward(self, x):
#         x = x.view((x.shape[0], -1))
#         x = self.m(x)

#         return x[:, -1]