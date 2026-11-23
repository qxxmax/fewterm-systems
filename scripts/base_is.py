"""Direct importance sampling from the base distributions alone (no flow), same pool as landau_compare.

    python scripts/base_is.py --d 2 --L 8 --kappa 0.28 --lam 0.022 --out FILE

Variants (q = base, w = exp(-S(x)) / q(x)):
  white_gauss   unit Gaussian on every site (the CFM base)
  white_landau  unit Gaussian on the non-zero modes, Landau zero mode (fewterm.landau)
  spec_gauss    Gaussian, diagonal in momentum, variance = pool spectrum averaged over each k^2 shell (zero mode: <y^2>)
  spec_landau   spec_gauss on the non-zero modes, Landau zero mode
Pool: HMC+Wolff (fewterm.wolff.fit_pool, local_every 5), seed 1 = the same pool as landau_compare / out/lscan.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewterm import landau, phi4  # noqa: E402
from fewterm.phi4 import DT, action  # noqa: E402
from fewterm.wolff import fit_pool  # noqa: E402

torch.set_num_threads(int(os.environ.get('FEWTERM_THREADS', '1')))


def k2_grid(L, d):
    s = 4 * torch.sin(math.pi * torch.arange(L, dtype=DT) / L) ** 2
    k2 = sum(s.reshape((-1,) + (1,) * (d - 1 - mu)) for mu in range(d))
    return torch.round(k2 * 1e9) / 1e9


class SpecBase:
    """x = ifft(sqrt(P) fft(w)), w white: E|fft(x)_k|^2 / V = P_k; zero mode optionally replaced by a 1D base."""

    def __init__(self, P, zero=None):
        self.P, self.zero = P, zero
        self.V, self.d = P.numel(), P.dim()
        self.ax = tuple(range(1, self.d + 1))
        m = torch.ones_like(P, dtype=torch.bool)
        if zero is not None:
            m[(0,) * self.d] = False
        self.mask = m

    def sample(self, shape, generator):
        w = torch.randn(shape, dtype=DT, generator=generator)
        x = torch.fft.ifftn(self.P.sqrt() * torch.fft.fftn(w, dim=self.ax), dim=self.ax).real
        if self.zero is not None:
            rt = math.sqrt(self.V)
            y_old = x.sum(self.ax) / rt
            x = x + ((self.zero.sample_y(shape[0], generator) - y_old) / rt).reshape((-1,) + (1,) * self.d)
        return x

    def logp(self, x):
        q = torch.fft.fftn(x, dim=self.ax).abs().square() / (self.V * self.P)
        n = int(self.mask.sum())
        out = -.5 * (q[:, self.mask].sum(1) + self.P[self.mask].log().sum() + n * math.log(2 * math.pi))
        if self.zero is not None:
            out = out + self.zero.logp_y(x.sum(self.ax) / math.sqrt(self.V))
        return out


class WhiteBase:
    def sample(self, shape, generator):
        return torch.randn(shape, dtype=DT, generator=generator)

    def logp(self, x):
        return -.5 * (x.square().flatten(1).sum(1) + x[0].numel() * math.log(2 * math.pi))


def run(base, n, seed, lam, kappa, shape, batch=4096):
    g = torch.Generator().manual_seed(seed)
    lw, ys = [], []
    for off in range(0, n, batch):
        x = base.sample((min(batch, n - off),) + shape, g)
        lw.append((-action(x, lam, kappa) - base.logp(x)).numpy())
        ys.append((x.flatten(1).sum(1) / math.sqrt(x[0].numel())).numpy())
    lw, y = np.concatenate(lw), np.concatenate(ys)
    w = np.exp(lw - lw.max())
    return {'ESS': float(w.sum() ** 2 / (len(w) * (w * w).sum())), 'Var_logw_per_site': float(lw.var() / np.prod(shape)),
            'logw_max_minus_median': float(lw.max() - np.median(lw)), 'raw_y2': float((y ** 2).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--pool', type=int, default=1024)
    p.add_argument('--n', type=int, default=262144)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    t0 = time.perf_counter()
    res = {'d': a.d, 'L': a.L, 'kappa': a.kappa, 'lambda': a.lam, 'n': a.n}
    fields, rep = fit_pool(a.L, a.kappa, a.seed, None, a.d, a.lam, a.pool, local_every=5)
    res['pool_frozen_chains'] = rep['frozen_chains']
    res['pool_seconds'] = time.perf_counter() - t0
    shape = (a.L,) * a.d
    ax = tuple(range(1, a.d + 1))
    Pk = torch.fft.fftn(fields, dim=ax).abs().square().mean(0) / fields[0].numel()
    k2 = k2_grid(a.L, a.d)
    P = torch.empty_like(Pk)
    for v in torch.unique(k2):
        P[k2 == v] = Pk[k2 == v].mean()
    y = fields.sum(ax) / math.sqrt(fields[0].numel())
    res['pool_y2'] = float((y ** 2).mean())
    zb = landau.from_pool(fields)
    res['landau'] = zb.info()
    # LandauBase itself = unit Gaussian non-zero modes + Landau zero mode
    for name, b in (('white_gauss', WhiteBase()), ('white_landau', zb),
                    ('spec_gauss', SpecBase(P)), ('spec_landau', SpecBase(P, zb))):
        t = time.perf_counter()
        out = run(b, a.n, a.seed + 50, a.lam, a.kappa, shape)
        out['seconds_per_proposal'] = (time.perf_counter() - t) / a.n
        res[name] = out
        print(name, json.dumps(out), flush=True)
    res['seconds'] = time.perf_counter() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=phi4.encode))


if __name__ == '__main__':
    main()
