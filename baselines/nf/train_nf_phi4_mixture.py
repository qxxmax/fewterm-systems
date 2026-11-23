"""Reverse-KL training of the convolutional flow of the paper with the Z2 mixture density

    q_mix(phi) = [q_psi(phi) + q_psi(-phi)] / 2,

    L_mix = E_{phi ~ q_mix} [log q_mix(phi) - log p(phi)],

where log q_psi(-phi) is obtained by running the inverse of the affine couplings on -phi.
Architecture as in train_nf_phi4.py (16 affine couplings, hidden [8, 8], kernel 3, circular padding),
adapted from the tutorial of Albergo et al., arXiv:2101.08176.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nf_phi4 import (  # noqa: E402
        AffineCoupling,  # noqa: F401
        SimpleNormal,
        ScalarPhi4Action,
        apply_flow_to_prior,
        compute_ess,
        compute_observables,
        make_phi4_affine_layers,
    )


# ── log q_psi(phi) for arbitrary phi (not necessarily flow output) ────

def log_q_psi(phi: torch.Tensor,
              layers: nn.ModuleList,
              prior: SimpleNormal) -> torch.Tensor:
    """Compute log q_psi(phi) = log prior(z) + sum_layers (-logJ_layer)
    where z is recovered by running the inverse coupling stack on phi.

    Each AffineCoupling.reverse(fx) returns (x_prev, -sum mask*s) so that
    log q_psi(phi) under the change of variables formula is

        log q_psi(phi) = log prior(z) - sum_layers logJ_forward(z->phi)
                      = log prior(z) + sum_layers (-logJ_forward).

    Since reverse() returns logJ_reverse = -logJ_forward, we ADD reverse
    logJs as we walk the inverse stack.
    """
    z = phi
    logJ_total = torch.zeros(phi.size(0), device=phi.device,
                             dtype=phi.dtype)
    # walk layers in reverse order; reverse() returns -logJ_forward for
    # the layer it's inverting, which is exactly what we add.
    for layer in reversed(list(layers)):
        z, logJ_rev = layer.reverse(z)
        logJ_total = logJ_total + logJ_rev
    return prior.log_prob(z) + logJ_total


def apply_flow_mixture(prior: SimpleNormal,
                       layers: nn.ModuleList,
                       *, batch_size: int):
    """Sample from q_mix and return (phi, log q_mix(phi)) under the
    proper mixture density.

    Sampling: z~N, phi_pos = f(z); s ~ Uniform{+1,-1}; phi = s * phi_pos.
    log q_psi(phi_pos) is the standard forward-pass log density.
    log q_mix(phi) = logsumexp(log q_psi(phi) + log 1/2,
                                log q_psi(-phi) + log 1/2).
    For the s = +1 branch we already have log q_psi(phi); for the
    s = -1 branch the sample equals -phi_pos so log q_psi(phi) = log q_psi(-phi_pos),
    which we recompute via inverse map. We always recompute log q_psi(-phi)
    so that the gradient flows through *both* branches at every step.
    """
    phi_pos, logq_psi_pos = apply_flow_to_prior(prior, layers,
                                                batch_size=batch_size)
    sign = 2 * torch.randint(0, 2, (batch_size,),
                             device=phi_pos.device).float() - 1
    sign_full = sign
    for _ in range(len(phi_pos.shape) - 1):
        sign_full = sign_full.unsqueeze(-1)
    phi = phi_pos * sign_full

    # log q_psi(phi)
    # For the +1 branch phi == phi_pos so log q_psi(phi) == logq_psi_pos.
    # For the -1 branch phi == -phi_pos and we need log q_psi(-phi_pos)
    # which is generically different (the conditioner is not odd).
    # We unify the two cases by recomputing log q_psi(phi) via inverse.
    logq_psi_phi = log_q_psi(phi, layers, prior)
    # log q_psi(-phi)
    logq_psi_neg = log_q_psi(-phi, layers, prior)

    log_half = float(np.log(0.5))
    logq_mix = torch.logsumexp(
        torch.stack([logq_psi_phi + log_half,
                     logq_psi_neg + log_half], dim=0), dim=0)
    return phi, logq_mix


# ── training ───────────────────────────────────────────────────────────

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
          f"kernel={args.kernel_size} -> {n_params / 1e6:.3f}M params "
          f"(should match train_nf_phi4.py)")

    optimizer = torch.optim.Adam(layers.parameters(), lr=args.lr)

    print(f"[train] Z2-mixture objective: "
          f"kappa={args.kappa} lam={args.lam} L={L} "
          f"epochs={args.epochs} bs={args.batch_size}")

    history = {'loss': [], 'ess': []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        phi, logq_mix = apply_flow_mixture(prior, layers,
                                           batch_size=args.batch_size)
        logp = -action(phi)
        loss = (logq_mix - logp).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ess = compute_ess(logp, logq_mix).item()
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

    # ── final eval ──
    print("[eval] sampling 8192 configs from q_mix ...")
    layers.eval()
    all_cfgs = []
    with torch.no_grad():
        for _ in range(8192 // args.batch_size + 1):
            phi, _ = apply_flow_mixture(prior, layers,
                                        batch_size=args.batch_size)
            all_cfgs.append(phi.detach().cpu().numpy())
    cfgs = np.concatenate(all_cfgs, axis=0)[:8192]
    obs = compute_observables(cfgs)

    print("=" * 60)
    for k, (m, e) in obs.items():
        print(f"  {k:8s}: {m:+.5f}  (err={e:.5f})")
    print("=" * 60)

    meta = {
        'kappa': args.kappa, 'lam': args.lam, 'L': L,
        'flow': 'C_mixture_proper',
        'z2_training': True,
        'mixture_objective': True,
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
