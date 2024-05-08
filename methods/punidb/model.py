import torch
import torch.nn as nn

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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
            nn.Linear(1024, S * D)
        )
    
    def forward(self, x, t):
        B, D = x.shape

        x_emb = self.embedding(x)   # (B, D, 16)
        net_input = torch.cat([x_emb, t[:, None, None].repeat(1, D, 1)], dim=-1).reshape(B, -1) # (B, D * 17)
        return self.net(net_input).reshape(B, D, -1)   # (B, D, S)
