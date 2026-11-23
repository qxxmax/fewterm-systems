"""Gaussian vs Landau zero-mode base (fewterm.landau) for the coupled flow, same pool, with zero-mode diagnostics.

    python scripts/landau_compare.py --d 2 --L 8 --kappa 0.28 --lam 0.022 --out FILE

Per variant: IS ESS and Var(log w)/V on --n proposals (micro_u retries), share of Var(log w) explained by the
output zero mode (R^2 of the binned conditional mean of log w given |y|, y = sqrt V mean phi), raw-proposal zero-mode
moments vs pool, raw/pool momentum-shell spectrum for the lowest shells, and the zero-mode share of the LSQ
validation residual per node.  Pool: HMC+Wolff (fewterm.wolff.fit_pool, local_every 5).
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
from fewterm import landau, phi4, phi4u  # noqa: E402
from fewterm.phi4 import DT, action  # noqa: E402
from fewterm.wolff import fit_pool  # noqa: E402

torch.set_num_threads(int(os.environ.get('FEWTERM_THREADS', '1')))


def shells(L, d, nshell=5):
    n = torch.arange(L, dtype=DT)
    s = 4 * torch.sin(math.pi * n / L) ** 2
    k2 = sum(s.reshape((-1,) + (1,) * (d - 1 - mu)) for mu in range(d))
    k2 = torch.round(k2 * 1e9) / 1e9
    vals = torch.unique(k2)[:nshell + 1]
    return k2, vals


def spectrum(x, k2, vals):
    d = x.dim() - 1
    P = torch.fft.fftn(x, dim=tuple(range(1, d + 1))).abs().square().mean(0) / x[0].numel()
    return [float(P[k2 == v].mean()) for v in vals]


def zm(x):
    d = x.dim() - 1
    return x.sum(tuple(range(1, d + 1))) / math.sqrt(x[0].numel())


def r2_binned(lw, y, nb=40):
    q = np.quantile(y, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(q, y, side='right') - 1, 0, nb - 1)
    mu = np.array([lw[idx == b].mean() if np.any(idx == b) else 0. for b in range(nb)])
    return float(np.var(mu[idx]) / np.var(lw))


@torch.no_grad()
def propose(model, n, seed, lam, kappa, batch=512):
    L, d = model.L, model.d
    g = torch.Generator().manual_seed(seed)
    lw, xs = [], []
    for off in range(0, n, batch):
        z = phi4u.base_noise((min(batch, n - off),) + (L,) * d, g, getattr(model, 'base', None))
        x, ld = model.transform(z, 32)
        lw.append((-action(x, lam, kappa) - (model.logp(z) - ld)).numpy())
        xs.append(x)
    return np.concatenate(lw), torch.cat(xs)


def run_variant(model, a, pool_stats, k2, vals, keep=None):
    err = None
    for mu_ in (.25, .0625, .015625):
        model.micro_u = mu_
        try:
            t = time.perf_counter()
            lw, x = propose(model, a.n, a.seed + 50, a.lam, a.kappa)
            sec = (time.perf_counter() - t) / a.n
            break
        except RuntimeError as ex:
            err = repr(ex)
    else:
        return {'error': err}
    w = np.exp(lw - lw.max())
    y = zm(x).numpy()
    if keep is not None:
        keep['y'], keep['logw'] = y, lw
    y2, y4 = float((y ** 2).mean()), float((y ** 4).mean())
    wn = w / w.sum()
    sp = spectrum(x, k2, vals)
    return {'ESS': float(w.sum() ** 2 / (len(w) * (w * w).sum())), 'Var_logw_per_site': float(lw.var() / x[0].numel()),
            'micro_u': mu_, 'seconds_per_proposal': sec,
            'R2_logw_given_absy': r2_binned(lw, np.abs(y)), 'R2_logw_given_y': r2_binned(lw, y),
            'raw_y2_rel': y2 / pool_stats['y2'] - 1, 'raw_y4_rel': y4 / pool_stats['y4'] - 1,
            'raw_U4': 1 - y4 / (3 * y2 ** 2), 'pool_U4': pool_stats['U4'],
            'rw_y2_rel': float((wn * y ** 2).sum()) / pool_stats['y2'] - 1,
            'raw_spectrum_over_pool': [s / p for s, p in zip(sp, pool_stats['spec'])],
            'logw_max_minus_median': float(lw.max() - np.median(lw))}


def grid_y2(base):
    """<y^2> under the piecewise-constant density exp(cell_logp) that logp_y evaluates (exact cell integrals)."""
    lo, hi = base.y[:-1], base.y[1:]
    return float((np.exp(base.cell_logp) * (hi ** 3 - lo ** 3) / 3).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--pool', type=int, default=1024)
    p.add_argument('--n', type=int, default=16384)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--variants', default='gauss,gauss_ot,landau,landau_ot,empirical_ot')
    p.add_argument('--save-npz', action='store_true', help='also write OUT.npz: pool zero modes and per-variant proposal zero modes + log w')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    t0 = time.perf_counter()
    res = {'d': a.d, 'L': a.L, 'kappa': a.kappa, 'lambda': a.lam}
    fields, rep = fit_pool(a.L, a.kappa, a.seed, None, a.d, a.lam, a.pool, local_every=5)
    res['pool_hot_cold_max_z'] = rep.get('hot_cold_max_abs_z')
    res['pool_frozen_chains'] = rep['frozen_chains']
    res['pool_seconds'] = time.perf_counter() - t0
    k2, vals = shells(a.L, a.d)
    y = zm(fields)
    ps = {'y2': float((y ** 2).mean()), 'y4': float((y ** 4).mean()), 'spec': spectrum(fields, k2, vals)}
    ps['U4'] = 1 - ps['y4'] / (3 * ps['y2'] ** 2)
    res['pool'] = {**ps, 'shells_k2': vals.tolist()}

    base = landau.from_pool(fields)
    g = torch.Generator().manual_seed(77)
    yb = base.sample_y(200000, g).numpy()
    ld_b = base.logp_y(torch.as_tensor(yb)).numpy()
    res['landau_base'] = {**base.info(), 'sample_y2_rel': float((yb ** 2).mean()) / ps['y2'] - 1,
                          'sample_y4_rel': float((yb ** 4).mean()) / ps['y4'] - 1,
                          'logp_finite': bool(np.isfinite(ld_b).all()),
                          'sampler_vs_logp_y2_rel': float((yb ** 2).mean()) / grid_y2(base) - 1}

    emp = landau.empirical_from_pool(fields)
    ye = emp.sample_y(200000, g).numpy()
    arrays = {'pool_y': y.numpy()}
    res['empirical_base'] = {'nq': len(emp.knots) - 1, 'sample_y2_rel': float((ye ** 2).mean()) / ps['y2'] - 1,
                             'sampler_vs_logp_y2_rel': float((ye ** 2).mean()) / grid_y2(emp) - 1}
    for name, b, ot in (('gauss', None, False), ('gauss_ot', None, True), ('landau', base, False),
                        ('landau_ot', base, True), ('empirical_ot', emp, True), ('landau_ot_nonn', base, True)):
        if name not in a.variants.split(','):
            continue
        t = time.perf_counter()
        if name.endswith('_nonn'):                  # ablation: no nearest-neighbour terms (UnboundedCNF, K = 14)
            model, info = phi4u.fit(fields, a.seed + 1, base=b, zm_ot=ot)
            if b is not None:
                landau.attach(model, b)
        else:
            model, info = phi4u.fit_coupled(fields, a.seed + 1, base=b, zm_ot=ot)
        fr = [r['val_res_zero_frac'] for r in info['rows'] if 'val_res_zero_frac' in r] or [float('nan')]
        keep = {} if a.save_npz else None
        out = run_variant(model, a, ps, k2, vals, keep)
        if keep:
            arrays[f'{name}_y'], arrays[f'{name}_logw'] = keep['y'], keep['logw']
        out.update({'fit_seconds': time.perf_counter() - t, 'val_res_zero_frac_mean': float(np.mean(fr)),
                    'val_res_zero_frac_max': float(np.max(fr)), 'one_over_V': 1 / fields[0].numel(),
                    'max_abs_zero_mode_coeff': float(getattr(model, 'full', model.coeff)[:, 3:6].abs().max())})
        res[name] = out
        print(name, json.dumps({k: v for k, v in out.items() if k != 'raw_spectrum_over_pool'}), flush=True)
    res['seconds'] = time.perf_counter() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=phi4.encode))
    if a.save_npz:
        np.savez_compressed(a.out.with_suffix('.npz'), **arrays)


if __name__ == '__main__':
    main()
