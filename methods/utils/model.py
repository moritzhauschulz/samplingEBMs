import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.model_has_noise = args.model_has_noise
        self.enable_backward = args.enable_backward
        if self.model_has_noise:
            S_noise = S
        if self.enable_backward:
            S_source = S
        self.relu = args.relu

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
        if self.model_has_noise:
            self.output_linear_noise = nn.Linear(n_channels, D*S_noise)
        if self.enable_backward:
            self.output_linear_backward = nn.Linear(n_channels, D * S_source)
            
        


    def forward(self, xt, t):
        B, D = xt.shape

        xt = xt.float()
        xt = self.x_proj_linear(xt.view(xt.size(0), -1))
        t = self.t_proj_linear(timestep_embedding(t, 28*28))
        input = xt + t

        input = self.input_conv(input.view(input.size(0), 1, 28, 28))
        h = self.net(input)
        h = h.view(h.size(0), h.size(1), -1).mean(-1)   # mean pooling: (B, C, D) -> (B, C)

        S_out = self.output_linear(h).reshape(B, D, -1)  # (B, C) -> (B, D*S)
        if self.model_has_noise:
            S_noise_out = self.output_linear_noise(h).reshape(B, D, -1)
        else:
            S_noise_out = None
        if self.enable_backward:
            S_back_out = self.output_linear_backward(h).reshape(B, D, -1)
            if self.relu:
                S_back_out = F.relu(S_back_out)
        else:
            S_back_out = None
        return S_out, S_back_out, S_noise_out    # (B, D, S)


class MLPModel(nn.Module):
    def __init__(self, args):
        super(MLPModel, self).__init__()

        D = args.discrete_dim
        S = args.vocab_size_with_mask if args.source == 'mask' else args.vocab_size
        self.model_has_noise = args.model_has_noise
        self.enable_backward = args.enable_backward
        if self.model_has_noise:
            S_noise = S
        if self.enable_backward:
            S_source = S
        self.relu = args.relu
        
        self.embedding = nn.Embedding(args.vocab_size_with_mask, 16)
        self.net = nn.Sequential(
            nn.Linear((16+1) * D, 1024),
            Swish(),
            nn.Linear(1024, 1024),
            Swish(),
            nn.Linear(1024, 1024),
            Swish(),
        )

        self.output_linear = nn.Sequential(
            nn.Linear(1024, D * S)
        )
        if self.model_has_noise:
            self.output_linear_noise = nn.Sequential(
                nn.Linear(1024, D * S_noise),
            )
        if self.enable_backward:
            self.output_linear_backward = nn.Sequential(
                nn.Linear(1024, D * S_source),
            )
    
    def forward(self, x, t):
        B, D = x.shape

        x_emb = self.embedding(x)   # (B, D, 16)
        t_expanded = t[:, None, None].repeat(1, D, 1) 

        net_input = torch.cat([x_emb, t_expanded], dim=-1).reshape(B, -1) # (B, D * 17)
        
        h = self.net(net_input)

        S_out = self.output_linear(h).reshape(B, D, -1)  # (B, C) -> (B, D*S)
        if self.relu:
            S_out = F.relu(S_out)
        if self.model_has_noise:
            S_noise_out = self.output_linear_noise(h).reshape(B, D, -1)
            if self.relu:
                S_noise_out = F.relu(S_noise_out)
        else:
            S_noise_out = None
        if self.enable_backward:
            S_back_out = self.output_linear_backward(h).reshape(B, D, -1)
            if self.relu:
                S_back_out = F.relu(S_back_out)
        else:
            S_back_out = None
        return S_out, S_back_out, S_noise_out    # (B, D, S)


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
        return -logp - bd # or +?

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