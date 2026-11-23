"""d-dimensional HMC + Wolff reference sampler and pool generator for phi^4.

Generalizes
  phi4-wolff-main/src/hmc_wolff_tau_phi4_fullsave.py  _build_nb_rect, _wolff_one, _wolff_batch
                                            (vendored unchanged in third_party/phi4_wolff, see SOURCES.md)
  PKG/phi4_nf/kernels.py                    IdentityMass leapfrog / hmc_step (kinetic sum and accept mask over d axes)
  TQ/run_followup.py                        initial, fit_pool (1024 chains x 4096 updates, Wolff every 5th)
  S0/gen_pools.py                           pool j uses seed base + 100 j
  S0/e18_csd.py                             mcmc (4 chains, hmc or hmcwolff, traces of M)
Neighbour table: C-order flat index of an L^d lattice, columns (+e_{d-1}, -e_{d-1}, ..., +e_0, -e_0); for d = 2
this is exactly _build_nb_rect(L, L), so the 2D Wolff random stream is reproduced bit-for-bit.
Step size: eps0 = 0.1 min(1, sqrt(16/L)) for d = 2 (original); eps0 = 0.1 min(1, (256/V)^(1/4)) in general,
which is the same V^(-1/4) rule; n_leap = ceil(1/eps0), eps = 1/n_leap (trajectory length 1)."""
import json
import os
import math
from pathlib import Path
import time

import numpy as np
from numba import njit
import torch

from .phi4 import (DT, LAM, KEYS, axes, phi4_action, phi4_force, observables, estimates, influence_series,
                   classical_minimum, write, encode)


# ----------------------------------------------------------------------------- Wolff (numba)
@njit(cache=True)
def build_neighbors(L, d):
    V = L ** d
    stride = np.empty(d, dtype=np.int64)
    stride[d - 1] = 1
    for mu in range(d - 2, -1, -1):
        stride[mu] = stride[mu + 1] * L
    nb = np.empty((V, 2 * d), dtype=np.int32)
    for i in range(V):
        k = 0
        for mu in range(d - 1, -1, -1):
            coord = (i // stride[mu]) % L
            base = i - coord * stride[mu]
            nb[i, k] = base + ((coord + 1) % L) * stride[mu]
            nb[i, k + 1] = base + ((coord - 1 + L) % L) * stride[mu]
            k += 2
    return nb


@njit(cache=True)
def wolff_one(phi_flat, nb, kappa, V, bond_factor=4.0):
    """Single-cluster Wolff flip phi -> -phi on the cluster; bond probability 1 - exp(-4 kappa phi_i phi_j).
    bond_factor (default 4, the correct value for this action) exists only for the mutation tests."""
    seed = np.random.randint(0, V)
    included = np.zeros(V, dtype=np.int32)
    stack = np.empty(V, dtype=np.int32)
    included[seed] = 1
    stack[0] = seed
    sp = 1
    cluster_size = 1
    nn = nb.shape[1]
    while sp > 0:
        sp -= 1
        i = stack[sp]
        for jpos in range(nn):
            j = nb[i, jpos]
            if included[j] == 0:
                bond = phi_flat[i] * phi_flat[j]
                if bond > 0.0 and np.random.random() < 1.0 - np.exp(-bond_factor * kappa * bond):
                    included[j] = 1
                    stack[sp] = j
                    sp += 1
                    cluster_size += 1
    for i in range(V):
        if included[i]:
            phi_flat[i] = -phi_flat[i]
    return cluster_size


@njit(cache=True)
def wolff_batch(phi_batch, nb, kappa, B, V, bond_factor=4.0):
    sizes = np.empty(B, dtype=np.int32)
    for b in range(B):
        sizes[b] = wolff_one(phi_batch[b], nb, kappa, V, bond_factor)
    return sizes


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


def wolff_update(phi, nb, kappa, bond_factor=4.0):
    """In-place-free Wolff update of a [B, L, ..., L] float64 CPU tensor; returns (phi, cluster sizes)."""
    B = phi.shape[0]
    V = phi[0].numel()
    flat = phi.numpy().reshape(B, V).copy()
    sizes = wolff_batch(flat, nb, float(kappa), B, V, float(bond_factor))
    return torch.from_numpy(flat.reshape(phi.shape)), sizes


# ----------------------------------------------------------------------------- HMC (kernels.hmc_step, identity mass)
def step_size(L, d, eps_factor=1.):
    eps = .1 * min(1., math.sqrt(16 / L)) if d == 2 else .1 * min(1., (256 / L ** d) ** .25)
    n_leap = math.ceil(1 / (eps * eps_factor))
    return 1 / n_leap, n_leap


def leapfrog(phi, momentum, epsilon, n_leap, lam, kappa):
    q = phi + (0.5 * epsilon) * momentum
    p = momentum.clone()
    for _ in range(n_leap - 1):
        p = p + epsilon * phi4_force(q, lam=lam, kappa=kappa)
        q = q + epsilon * p
    p = p + epsilon * phi4_force(q, lam=lam, kappa=kappa)
    q = q + (0.5 * epsilon) * p
    return q, p


@torch.no_grad()
def hmc_step(phi, epsilon, n_leap, lam=LAM, kappa=.27, noise=None, uniform=None):
    """kernels.hmc_step with IdentityMass on [B, L, ..., L]; noise/uniform supplied by the caller."""
    d = phi.dim() - 1
    ax = axes(d)
    momentum = noise
    h_old = phi4_action(phi, lam, kappa) + 0.5 * momentum.square().sum(ax)
    if not bool(torch.isfinite(h_old).all()):
        raise ValueError("initial Hamiltonian is not finite")
    q, p = leapfrog(phi, momentum, epsilon, n_leap, lam, kappa)
    h_new = phi4_action(q, lam, kappa) + 0.5 * p.square().sum(ax)
    delta_h = h_new - h_old
    finite = torch.isfinite(delta_h) & torch.isfinite(q).all(ax) & torch.isfinite(p).all(ax)
    accepted = finite & (uniform.log() < -delta_h)
    return torch.where(accepted.reshape((-1,) + (1,) * d), q, phi), {"accepted": accepted, "delta_h": delta_h}


def metropolis_sweep(phi, lam, kappa, generator, delta=1.):
    """One checkerboard single-site Metropolis sweep (exact for the phi4 action; L even).  Repairs chains that HMC
    can no longer move, e.g. hot starts with a site at |phi| ~ 4.5 where the leapfrog step is unstable."""
    d = phi.dim() - 1
    ax = axes(d)
    parity = sum(torch.arange(phi.shape[-1]).reshape((-1,) + (1,) * (d - 1 - i)) for i in range(d)) % 2
    for colour in (0, 1):
        h = sum(torch.roll(phi, s, mu) for mu in ax for s in (1, -1))
        new = phi + delta * torch.randn(phi.shape, dtype=phi.dtype, generator=generator)
        dens = lambda f: (1 - 2 * lam) * f ** 2 + lam * f ** 4 - 2 * kappa * f * h
        u = torch.rand(phi.shape, dtype=phi.dtype, generator=generator)
        ok = (u.log() < dens(phi) - dens(new)) & (parity == colour)
        phi = torch.where(ok, new, phi)
    return phi


# ----------------------------------------------------------------------------- pools (run_followup.fit_pool)
def initial(L, k, n, seed, d=2, lam=LAM):
    """Half independent Gaussian, half signed classical minima plus 0.1 Gaussian noise."""
    rng = torch.Generator().manual_seed(seed)
    phi = torch.randn((n,) + (L,) * d, dtype=torch.float64, generator=rng)
    v = classical_minimum(k, lam, d) if d != 2 else math.sqrt(max(0., (4 * k - (1 - 2 * lam)) / (2 * lam)))
    signs = torch.where(torch.arange(n - n // 2) % 2 == 0, 1., -1.)       # n - n//2 = n//2 for even n
    phi[n // 2:] = signs.reshape((-1,) + (1,) * d) * v + .1 * phi[n // 2:]
    return phi


def endpoint_stats(phi, lam, kappa, d):
    obs = observables(phi, lam, kappa, d)
    V = phi[0].numel()
    est = estimates([obs[k].mean() for k in KEYS], V)
    inf = influence_series(obs, V)
    return {k: {'mean': v, 'se_iid': float(inf[k].std(ddof=1) / math.sqrt(len(phi)))} for k, v in est.items()}


@torch.no_grad()
def fit_pool(L, k, seed, folder=None, d=2, lam=LAM, n=1024, warmup=4096, wolff_every=5, report_every=256,
             eps_factor=1., local_every=0):
    """run_followup.fit_pool in d dimensions: n chains, `warmup` HMC updates, one Wolff cluster every 5th;
    the final state of every chain is the pool (no filtering).  Hot/cold z-scores as a finite-sample check.
    local_every > 0: also one checkerboard Metropolis sweep every local_every-th update (fixes frozen hot chains).
    report['frozen_chains'] = chains without an accepted HMC proposal in the last 256 updates."""
    phi = initial(L, k, n, seed, d, lam)
    recent = torch.zeros(n)
    generator = torch.Generator().manual_seed(seed + 1)
    nb = build_neighbors(L, d)
    seed_numba(seed + 2)
    epsilon, n_leap = step_size(L, d, eps_factor)
    trace, acceptances, cluster = [], [], []
    started = time.perf_counter()
    V = L ** d
    for i in range(warmup):
        noise = torch.randn(phi.shape, dtype=torch.float64, generator=generator)
        uniform = torch.rand(n, dtype=torch.float64, generator=generator)
        phi, info = hmc_step(phi, epsilon, n_leap, lam=lam, kappa=k, noise=noise, uniform=uniform)
        acceptances.append(float(info['accepted'].double().mean()))
        if i >= warmup - 256:
            recent += info['accepted'].double()
        if local_every and (i + 1) % local_every == 0:
            phi = metropolis_sweep(phi, lam, k, generator)
        if (i + 1) % wolff_every == 0:
            phi, sizes = wolff_update(phi, nb, k)
            cluster.append(float(sizes.mean() / V))
        if (i + 1) % report_every == 0 and L >= 4:
            groups = [endpoint_stats(x, lam, k, d) for x in phi.chunk(2)]
            trace.append({'step': i + 1, 'hot': groups[0], 'cold': groups[1]})
    report = {'seed': seed, 'd': d, 'L': L, 'kappa': k, 'lambda': lam, 'N': n, 'updates': warmup,
              'epsilon': epsilon, 'n_leap': n_leap,
              'kernel': f'HMC followed by one Wolff single-cluster update every {wolff_every} steps',
              'initialization': 'half independent Gaussian, half signed classical minima plus Gaussian noise',
              'fit_sample': 'one final state per independently initialized chain; no filtering',
              'acceptance_last256': float(np.mean(acceptances[-256:])), 'local_every': local_every,
              'frozen_chains': int((recent == 0).sum()), 'frozen_hot': int((recent[:n // 2] == 0).sum()),
              'cluster_fraction': float(np.mean(cluster)) if cluster else None, 'trace': trace,
              'seconds': time.perf_counter() - started,
              'caution': 'Agreement of starts is a finite-sample diagnostic, not proof of equilibration.'}
    if L >= 4:
        final = [endpoint_stats(x, lam, k, d) for x in phi.chunk(2)]
        diff = {m: (final[0][m]['mean'] - final[1][m]['mean']) / math.hypot(final[0][m]['se_iid'], final[1][m]['se_iid'])
                for m in ('absM', 'U4', 'chi', 'action_density', 'Smin')}
        report.update(hot_cold_z=diff, hot_cold_max_abs_z=max(map(abs, diff.values())))
    if folder is not None:
        torch.save(phi, Path(folder) / 'fit_pool.pt')
        write(Path(folder) / 'fit_pool_setup.json', report)
    return phi, report


# ----------------------------------------------------------------------------- chains (e18_csd.mcmc)
@torch.no_grad()
def mcmc(L, kappa, kernel, seed, d=2, lam=LAM, n_chains=4, warm=4096, n_rec=65536, record=None, bond_factor=4.0,
         eps_factor=1., local_every=0):
    """4 chains of the pool kernel (kernel 'hmc' or 'hmcwolff'); returns traces of M (and phi2, energy).
    bond_factor != 4 gives a deliberately wrong Wolff kernel (mutation tests only)."""
    phi = initial(L, kappa, n_chains, seed, d, lam)
    gen = torch.Generator().manual_seed(seed + 1)
    if kernel == 'hmcwolff':
        nb = build_neighbors(L, d)
        seed_numba(seed + 2)
    eps, n_leap = step_size(L, d, eps_factor)
    ax = axes(d)
    rec = {'M': [], 'phi2': [], 'energy': []}
    accs = []
    t0 = None
    for i in range(warm + n_rec):
        if i == warm:
            t0 = time.perf_counter()
        noise = torch.randn(phi.shape, dtype=torch.float64, generator=gen)
        uniform = torch.rand(n_chains, dtype=torch.float64, generator=gen)
        phi, info = hmc_step(phi, eps, n_leap, lam=lam, kappa=kappa, noise=noise, uniform=uniform)
        if local_every and (i + 1) % local_every == 0:
            phi = metropolis_sweep(phi, lam, kappa, gen)
        if kernel == 'hmcwolff' and (i + 1) % 5 == 0:
            phi, _ = wolff_update(phi, nb, kappa, bond_factor)
        if i >= warm:
            e = phi * phi.roll(1, -1)
            for mu in range(2, d + 1):
                e = e + phi * phi.roll(1, -mu)
            rec['M'].append(phi.mean(ax).numpy().copy())
            rec['phi2'].append(phi.square().mean(ax).numpy().copy())
            rec['energy'].append(e.mean(ax).numpy().copy())
            accs.append(info['accepted'].numpy().copy())
    sec = (time.perf_counter() - t0) / max(1, n_rec)
    out = {k: np.array(v) for k, v in rec.items()}
    out['absM'], out['M2'], out['M4'] = np.abs(out['M']), out['M'] ** 2, out['M'] ** 4
    out['accepted'] = np.array(accs)
    info = {'d': d, 'L': L, 'kappa': kappa, 'lambda': lam, 'kernel': kernel, 'epsilon': eps, 'n_leap': n_leap,
            'seconds_per_update': sec, 'N_recorded': n_rec, 'local_every': local_every, 'acceptance': float(np.mean(accs)), 'seed': seed}
    if record is not None:
        Path(record).mkdir(parents=True)
        with (Path(record) / 'chains.npz').open('xb') as fh:
            np.savez_compressed(fh, **out)
        write(Path(record) / 'record.json', info)
    return out, info


def main():
    import argparse
    import hashlib
    p = argparse.ArgumentParser(description='d-dim HMC+Wolff pools (gen_pools protocol)')
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, default=LAM)
    p.add_argument('--pools', type=int, default=3)
    p.add_argument('--base-seed', type=int, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--updates', type=int, default=4096)
    p.add_argument('--eps-factor', type=float, default=1., help='HMC step = default step * factor (1 = original)')
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('FEWTERM_THREADS', '1')))
    folder = a.root / f'd{a.d}_L{a.L}_k{a.kappa:.4f}_l{a.lam:.4f}'
    folder.mkdir(parents=True, exist_ok=True)
    src = Path(__file__)
    (folder / f'sources_{a.base_seed}.json').write_text(json.dumps(
        {str(src): hashlib.sha256(src.read_bytes()).hexdigest()}, indent=1), encoding='utf-8')
    for j in range(a.pools):
        out = folder / f'pool_{j}'
        out.mkdir()
        _, report = fit_pool(a.L, a.kappa, a.base_seed + 100 * j, out, a.d, a.lam, a.n, a.updates,
                             eps_factor=a.eps_factor)
        print(json.dumps({'pool': str(out), 'hot_cold_max_abs_z': report.get('hot_cold_max_abs_z'),
                          'acceptance': report['acceptance_last256'], 'seconds': report['seconds']},
                         default=encode), flush=True)
    (folder / 'complete.json').write_text(json.dumps({'complete': True, 'pools': a.pools}), encoding='utf-8')


if __name__ == '__main__':
    main()
