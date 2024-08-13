import torch
import torch.nn as nn

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}

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

class alt_MLPModel(nn.Module):
    def __init__(self, args):
        super(alt_MLPModel, self).__init__()

        D = args.discrete_dim
        self.args = args
        self.S = args.vocab_size_with_mask
        self.embedding = nn.Embedding(self.S, 16)
        self.net = nn.Sequential(
            nn.Linear((16+1) * D, 1024),
            Swish(),
            nn.Linear(1024, 1024),
            Swish(),
            nn.Linear(1024, 1024),
            Swish(),
            nn.Linear(1024, (self.S - 1) * D), #note we reduce output state number (diagonal is imputed later)
            nn.ReLU()
        )
    
    def forward(self, x, t):
        B, D = x.shape

        x_emb = self.embedding(x)   # (B, D, 16)
        t_expanded = t[:, None, None].repeat(1, D, 1) 
        net_input = torch.cat([x_emb, t_expanded], dim=-1).reshape(B, -1) # (B, D * 17)
        net_output = self.net(net_input).reshape(B, D, -1)
        # Create a mask to identify positions to exclude
        mask = torch.ones((B, D, self.S), dtype=bool).to(self.args.device)
        batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2)
        dim_indices = torch.arange(D).unsqueeze(0).unsqueeze(2)
        mask[batch_indices, dim_indices, x.unsqueeze(-1)] = False
        expanded_net_output = torch.zeros((B, D, self.S)).to(self.args.device)
        net_output_flat = net_output.view(B, D * (self.S - 1))
        expanded_net_output[mask] = net_output_flat.flatten()
        expanded_net_output[~mask] = -1 * torch.sum(expanded_net_output,-1).flatten()

        return expanded_net_output   # (B, D, S)

class MLPModel(nn.Module):
    def __init__(self, args):
        super(MLPModel, self).__init__()

        D = args.discrete_dim
        S = args.vocab_size
        self.embedding = nn.Embedding(args.vocab_size_with_mask, 16)
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
        t_expanded = t[:, None, None].repeat(1, D, 1) 

        net_input = torch.cat([x_emb, t_expanded], dim=-1).reshape(B, -1) # (B, D * 17)
        return self.net(net_input).reshape(B, D, -1)   # (B, D, S)


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
