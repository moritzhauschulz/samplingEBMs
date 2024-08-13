import torch
import torch.distributions as dists
import numpy as np

""" Three methods:  1. Uniform perturbation + uninformed proposal
                    2. Structured perturbation + uninformed proposal
                    3. Structured perturbation + informed proposal
"""

def ed_categorical(energy_net, samples, K = 5, epsilon = 1., m_particles = 32, w_stable = 1.):
    """ Perturbation assumes periodic structure on discrete values"""
    device = samples.device
    bs, dim = samples.shape

    y = (samples + (epsilon * torch.randn_like(samples)).int())%K   # [bs, dim]

    neg_data = (y.unsqueeze(1) + (epsilon * torch.randn(bs, m_particles, dim).int()))%K   # [bs, m_particles, dim]

    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(neg_data.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    noise = torch.randn(10000).int()

    plt.hist(noise, bins='auto')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Noise')
    plt.show()

    