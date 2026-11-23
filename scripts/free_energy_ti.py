"""Reference free energy density by thermodynamic integration in kappa with HMC+Wolff, and its wall time.

    log Z(kappa) = log Z(0) + int_0^kappa dk' 2 <sum_x sum_mu phi_x phi_{x+mu}>_{k'}
    log Z(0)     = V log int dphi exp(-(1-2 lam) phi^2 - lam phi^4)          (1D quadrature)

Uniform kappa grid (step --dk) from 0 to --kmax; at every node --chains HMC+Wolff chains (local_every 5), --warm
then --n recorded updates; node mean and error of the link sum by the Gamma method.  The integral is cumulative
trapezoid; the Simpson value on the even nodes is stored as a quadrature check.  Per node the wall time is
recorded, so the cost of F(kappa) at any grid kappa is the cumulative time up to it.

    python scripts/free_energy_ti.py --L 16 --out FILE
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyerrors as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewterm.wolff import mcmc  # noqa: E402


def logz0_site(lam, lim=12., n=400001):
    u = np.linspace(-lim, lim, n)
    f = np.exp(-(1 - 2 * lam) * u * u - lam * u ** 4)
    return math.log(((f[1:] + f[:-1]) * .5 * (u[1] - u[0])).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, default=2)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--lam', type=float, default=.022)
    p.add_argument('--kmax', type=float, default=.30)
    p.add_argument('--dk', type=float, default=.0025)
    p.add_argument('--chains', type=int, default=16)
    p.add_argument('--warm', type=int, default=1024)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    V = a.L ** a.d
    ks = np.round(np.arange(0, a.kmax + a.dk / 2, a.dk), 6)
    rows = []
    for j, k in enumerate(ks):
        t = time.perf_counter()
        ch, info = mcmc(a.L, float(k), 'hmcwolff', a.seed + 1000 * j, a.d, a.lam, a.chains, a.warm, a.n, local_every=5)
        link = 2 * V * ch['energy']                         # d log Z / d kappa per configuration, [T, chains]
        o = pe.Obs([link[:, c] for c in range(a.chains)], [f'k{j}|r{c}' for c in range(a.chains)])
        o.gamma_method(S=2.0)
        rows.append({'kappa': float(k), 'dlogZ': float(o.value), 'err': float(o.dvalue),
                     'seconds': time.perf_counter() - t})
        print(json.dumps(rows[-1]), flush=True)
    g = np.array([r['dlogZ'] for r in rows]); e = np.array([r['err'] for r in rows])
    sec = np.cumsum([r['seconds'] for r in rows])
    lz0 = V * logz0_site(a.lam)
    trap = np.concatenate([[0.], np.cumsum(.5 * a.dk * (g[1:] + g[:-1]))])
    # error of the cumulative trapezoid (independent nodes): weights dk/2 at both ends, dk inside
    var = np.zeros(len(ks))
    for i in range(1, len(ks)):
        w = np.ones(i + 1); w[0] = w[-1] = .5
        var[i] = ((w * a.dk) ** 2 * e[:i + 1] ** 2).sum()
    simp = np.full(len(ks), np.nan)
    for i in range(2, len(ks), 2):
        simp[i] = a.dk / 3 * (g[0] + g[i] + 4 * g[1:i:2].sum() + 2 * g[2:i - 1:2].sum())
    out = {'L': a.L, 'd': a.d, 'lambda': a.lam, 'dk': a.dk, 'chains': a.chains, 'warm': a.warm, 'n': a.n,
           'logZ0': lz0, 'nodes': rows,
           'grid': [{'kappa': float(k), 'F_over_V': -(lz0 + tr) / V, 'err': math.sqrt(v) / V,
                     'simpson_minus_trap_over_V': (None if np.isnan(s) else float((s - tr) / V)),
                     'cum_seconds': float(c)} for k, tr, v, s, c in zip(ks, trap, var, simp, sec)]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
