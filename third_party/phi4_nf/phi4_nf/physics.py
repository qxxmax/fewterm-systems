import torch


def phi4_action_density(cfgs, lam=0.022, kappa=0.27, k=None):
    """
    Per-site action density.

    Parameters
    ----------
    cfgs : torch.Tensor, shape [batch, Lx, Ly]
    lam : float
    kappa : float  (alias *k* also accepted for backward compat)
    """
    if k is not None:
        kappa = k
    density = (1 - 2 * lam) * cfgs**2 + lam * cfgs**4
    for mu in range(1, len(cfgs.shape)):
        density -= 2 * kappa * cfgs * torch.roll(cfgs, shifts=1, dims=mu)
    return density


def phi4_action(cfgs, lam=0.022, kappa=0.27, k=None):
    """Total action per configuration, shape [batch]."""
    if k is not None:
        kappa = k
    density = phi4_action_density(cfgs, lam=lam, kappa=kappa)
    return density.sum(dim=tuple(range(1, len(cfgs.shape))))


def phi4_force(cfgs, lam=0.022, kappa=0.27, k=None):
    """
    Force F = -dS/dphi.
    F(x) = -[2(1-2*lam)*phi + 4*lam*phi^3] + 2*kappa * sum_mu [phi(x+mu) + phi(x-mu)]
    """
    if k is not None:
        kappa = k
    grad_S = 2 * (1 - 2 * lam) * cfgs + 4 * lam * cfgs**3
    neighbor_sum = torch.zeros_like(cfgs)
    for mu in range(1, len(cfgs.shape)):
        neighbor_sum += torch.roll(cfgs, shifts=1, dims=mu)
        neighbor_sum += torch.roll(cfgs, shifts=-1, dims=mu)
    grad_S -= 2 * kappa * neighbor_sum
    return -grad_S
