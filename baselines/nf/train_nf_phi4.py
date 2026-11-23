"""Reverse-KL normalizing flow for phi^4 lattice scalar field theory.

Adapted from Albergo, Boyda, Hackett, Kanwar, Cranmer, Racaniere,
Rezende, Shanahan - arXiv:2101.08176 (tutorial notebook).

The paper uses the Z2 mixture version in train_nf_phi4_mixture.py.

Uses our action convention:
  S = sum_x [(1-2*lam)*phi^2 + lam*phi^4 - 2*kappa*sum_mu phi(x)*phi(x+mu)]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import packaging.version
import torch
import torch.nn as nn

# ── action ──────────────────────────────────────────────────────────

class ScalarPhi4Action:
    def __init__(self, kappa: float, lam: float):
        self.kappa = kappa
        self.lam = lam

    def __call__(self, cfgs: torch.Tensor) -> torch.Tensor:
        density = (1 - 2 * self.lam) * cfgs**2 + self.lam * cfgs**4
        Nd = len(cfgs.shape) - 1
        for mu in range(1, Nd + 1):
            density -= 2 * self.kappa * cfgs * torch.roll(cfgs, -1, mu)
        return torch.sum(density, dim=tuple(range(1, Nd + 1)))


# ── prior ───────────────────────────────────────────────────────────

class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(loc, var)
        self.shape = loc.shape

    def log_prob(self, x):
        return self.dist.log_prob(x).sum(dim=tuple(range(1, len(x.shape))))

    def sample_n(self, batch_size):
        return self.dist.sample((batch_size,))


# ── checkerboard mask ───────────────────────────────────────────────

def make_checker_mask(shape, parity, device='cpu'):
    checker = torch.ones(shape, dtype=torch.uint8, device=device) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker


# ── coupling layer ──────────────────────────────────────────────────

class AffineCoupling(nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity):
        super().__init__()
        self.register_buffer(
            'mask', make_checker_mask(mask_shape, mask_parity))
        self.net = net

    def forward(self, x):
        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return fx, logJ

    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        x = (1 - self.mask) * (fx_active - t) * torch.exp(-s) + fx_frozen
        axes = range(1, len(s.size()))
        logJ = -torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return x, logJ


# ── CNN conditioner ─────────────────────────────────────────────────

def make_conv_net(*, hidden_sizes, kernel_size, in_channels,
                  out_channels, use_final_tanh):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    padding_size = kernel_size // 2
    net = []
    for i in range(len(sizes) - 1):
        net.append(nn.Conv2d(
            sizes[i], sizes[i + 1], kernel_size,
            padding=padding_size, stride=1, padding_mode='circular'))
        if i != len(sizes) - 2:
            net.append(nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(nn.Tanh())
    return nn.Sequential(*net)


# ── assemble flow ───────────────────────────────────────────────────

def make_phi4_affine_layers(*, n_layers, lattice_shape,
                            hidden_sizes, kernel_size):
    layers = []
    for i in range(n_layers):
        parity = i % 2
        net = make_conv_net(
            in_channels=1, out_channels=2,
            hidden_sizes=hidden_sizes,
            kernel_size=kernel_size, use_final_tanh=True)
        coupling = AffineCoupling(
            net, mask_shape=lattice_shape, mask_parity=parity)
        layers.append(coupling)
    return nn.ModuleList(layers)


# ── forward pass ────────────────────────────────────────────────────

def apply_flow_to_prior(prior, coupling_layers, *, batch_size):
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer(x)
        logq = logq - logJ
    return x, logq


def apply_flow_to_prior_z2(prior, coupling_layers, *, batch_size):
    """Z2-symmetrized: q(phi) = 0.5*q_psi(phi) + 0.5*q_psi(-phi)."""
    x, logq = apply_flow_to_prior(prior, coupling_layers,
                                  batch_size=batch_size)
    sign = 2 * torch.randint(0, 2, (batch_size,),
                             device=x.device).float() - 1
    for _ in range(len(x.shape) - 1):
        sign = sign.unsqueeze(-1)
    x = x * sign
    return x, logq


# ── metrics ─────────────────────────────────────────────────────────

def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2 * torch.logsumexp(logw, dim=0) - \
        torch.logsumexp(2 * logw, dim=0)
    return torch.exp(log_ess) / len(logw)


def jackknife_mean_err(samples):
    N = len(samples)
    means = np.array([np.delete(samples, i).mean() for i in range(N)])
    m = means.mean()
    err = np.sqrt((N - 1) * np.mean((means - m) ** 2))
    return float(m), float(err)


def compute_observables(cfgs_np):
    N, L, _ = cfgs_np.shape
    V = L * L
    M = cfgs_np.mean(axis=(1, 2))
    absM = np.abs(M)
    chi = V * np.var(M)
    phi2 = (cfgs_np ** 2).mean(axis=(1, 2))
    M2_bar = (M ** 2).mean()
    M4_bar = (M ** 4).mean()
    U4 = 1.0 - M4_bar / (3.0 * M2_bar ** 2) if M2_bar > 0 else 0.0
    P_Mpos = float((M > 0).mean())
    return {
        'M': jackknife_mean_err(M),
        'absM': jackknife_mean_err(absM),
        'chi': (float(chi), 0.0),
        'phi2': jackknife_mean_err(phi2),
        'U4': (float(U4), 0.0),
        'P_Mpos': (float(P_Mpos), 0.0),
    }


# ── training ────────────────────────────────────────────────────────

def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    L = args.L
    lattice_shape = (L, L)
    action = ScalarPhi4Action(kappa=args.kappa, lam=args.lam)

    prior = SimpleNormal(
        torch.zeros(lattice_shape, device=device),
        torch.ones(lattice_shape, device=device))

    layers = make_phi4_affine_layers(
        n_layers=args.n_layers,
        lattice_shape=lattice_shape,
        hidden_sizes=args.hidden_sizes,
        kernel_size=args.kernel_size).to(device)

    n_params = sum(p.numel() for p in layers.parameters())
    print(f"[arch] n_layers={args.n_layers} hidden={args.hidden_sizes} "
          f"kernel={args.kernel_size} -> {n_params / 1e6:.3f}M params")

    optimizer = torch.optim.Adam(layers.parameters(), lr=args.lr)

    z2_mode = args.z2
    forward_fn = apply_flow_to_prior_z2 if z2_mode else apply_flow_to_prior
    print(f"[train] kappa={args.kappa} lam={args.lam} L={L} "
          f"z2={z2_mode} epochs={args.epochs} bs={args.batch_size}")

    history = {'loss': [], 'ess': []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        x, logq = forward_fn(prior, layers, batch_size=args.batch_size)
        logp = -action(x)
        loss = (logq - logp).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ess = compute_ess(logp, logq).item()
        history['loss'].append(loss.item())
        history['ess'].append(ess)

        if epoch == 1 or epoch % args.log_every == 0:
            elapsed = (time.time() - t0) / 60
            avg_ess = np.mean(history['ess'][-args.log_every:])
            avg_loss = np.mean(history['loss'][-args.log_every:])
            print(f"  ep {epoch}/{args.epochs} "
                  f"loss={avg_loss:.4f} ESS={avg_ess:.4f} "
                  f"({elapsed:.1f} min)")

        if epoch % max(1, args.log_every) == 0:
            torch.save(layers.state_dict(), args.out)

    elapsed = (time.time() - t0) / 60
    torch.save(layers.state_dict(), args.out)
    print(f"[done] {elapsed:.1f} min")
    print(f"[save] {args.out}")

    # Sample and evaluate
    print("[eval] sampling 8192 configs ...")
    layers.eval()
    all_cfgs = []
    with torch.no_grad():
        for _ in range(8192 // args.batch_size + 1):
            x, logq = forward_fn(prior, layers,
                                 batch_size=args.batch_size)
            all_cfgs.append(x.detach().cpu().numpy())
    cfgs = np.concatenate(all_cfgs, axis=0)[:8192]
    obs = compute_observables(cfgs)

    print("=" * 60)
    for k, (m, e) in obs.items():
        print(f"  {k:8s}: {m:+.5f}  (err={e:.5f})")
    print("=" * 60)

    meta = {
        'kappa': args.kappa, 'lam': args.lam, 'L': L,
        'z2': z2_mode,
        'n_layers': args.n_layers,
        'hidden_sizes': args.hidden_sizes,
        'kernel_size': args.kernel_size,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'seed': args.seed,
        'n_params': n_params,
        'final_loss': history['loss'][-1],
        'final_ess': history['ess'][-1],
        'loss_history': history['loss'],
        'ess_history': history['ess'],
        'observables': {k: {'mean': m, 'err': e}
                        for k, (m, e) in obs.items()},
        'wall_min': elapsed,
    }
    meta_path = args.out.replace('.pt', '.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {meta_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kappa', type=float, default=0.27)
    p.add_argument('--lam', type=float, default=0.022)
    p.add_argument('--L', type=int, default=8)
    p.add_argument('--z2', action='store_true', default=False,
                   help='Use Z2-symmetrized mixture flow')
    p.add_argument('--n_layers', type=int, default=16)
    p.add_argument('--hidden_sizes', type=int, nargs='+', default=[8, 8])
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--epochs', type=int, default=2500)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--out', type=str, required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
