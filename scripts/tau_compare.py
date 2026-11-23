"""Autocorrelation and physics of flow-IMH chains vs HMC+Wolff chains at one phi4 point (both with the repaired pool kernel).

    python scripts/tau_compare.py --d 3 --L 8 --kappa 0.15 --lam 1.0 --eps-factor 0.5 --out FILE

Flow: repaired fit_pool (local_every 5) -> phi4u.fit_coupled -> 4 independence-Metropolis chains (phi4.imh) of --n steps
on fresh proposals (exact for any proposal).  Reference: wolff.mcmc 'hmcwolff', 4 chains of --n updates, local_every 5.
For <|M|>, <M^2>, U4 = 1 - <M^4>/(3<M^2>^2), <E> (nn energy density), <phi^2>: Gamma-method mean, error, tau_int
(pyerrors, S = 2, 4 replica chains), and z = (flow - ref) / combined error.  Cost: seconds per IMH step (one proposal)
and per HMC+Wolff update; tau_int is in units of those steps.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewterm import phi4, phi4u  # noqa: E402
from fewterm import landau  # noqa: E402
from fewterm.wolff import fit_pool, mcmc  # noqa: E402

torch.set_num_threads(int(os.environ.get('FEWTERM_THREADS', '1')))
torch.set_default_dtype(torch.float64)


def gamma(series, tag):
    """series: dict name -> [T, R] (R replica chains); returns mean/err/tau per observable incl. derived U4."""
    R = next(iter(series.values())).shape[1]
    names = [f'{tag}|r{c}' for c in range(R)]
    o = {k: pe.Obs([v[:, c] for c in range(R)], names) for k, v in series.items()}
    o['U4'] = 1 - o['M4'] / (3 * o['M2'] * o['M2'])
    out = {}
    for k, v in o.items():
        v.gamma_method(S=2.0)
        t = v.e_tauint.get(tag, None) if isinstance(v.e_tauint, dict) else None
        dt = v.e_dtauint.get(tag, None) if isinstance(getattr(v, 'e_dtauint', None), dict) else None
        out[k] = {'mean': float(v.value), 'err': float(v.dvalue), 'tau_int': float(t) if t is not None else None,
                  'tau_int_err': float(dt) if dt is not None else None}
    return out


def obs(x, d):
    ax = tuple(range(1, d + 1))
    M = x.mean(ax)
    e = sum(x * x.roll(1, mu) for mu in ax).mean(ax) / d
    return {'absM': M.abs(), 'M2': M ** 2, 'M4': M ** 4, 'E': e, 'phi2': (x ** 2).mean(ax)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--eps-factor', type=float, default=1.)
    p.add_argument('--n', type=int, default=16384)
    p.add_argument('--warm', type=int, default=1024)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--hmc-chains', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--base', choices=('gauss', 'landau'), default='gauss', help='zero-mode base (fewterm.landau)')
    p.add_argument('--zm-ot', action='store_true', help='monotone zero-mode pairing in the LSQ fit')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    t0 = time.perf_counter()
    res = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    pool, rep = fit_pool(a.L, a.kappa, a.seed, None, a.d, a.lam, 1024, eps_factor=a.eps_factor, local_every=5)
    res['pool_frozen_chains'], res['pool_seconds'] = rep['frozen_chains'], rep['seconds']
    # second-moment correlation length of the (equilibrated) pool: xi = sqrt(chi/F - 1) / (2 sin(pi/L))
    ax = tuple(range(1, a.d + 1))
    V = a.L ** a.d
    chi = float((pool.sum(ax) ** 2).mean() / V)
    ph = torch.exp(2j * math.pi * torch.arange(a.L) / a.L)
    F = float(np.mean([(pool * ph.reshape((-1,) + (1,) * (a.d - 1 - mu)).real).sum(ax).pow(2).mean()
                       + (pool * ph.reshape((-1,) + (1,) * (a.d - 1 - mu)).imag).sum(ax).pow(2).mean()
                       for mu in range(a.d)]) / V)
    res['chi_pool'], res['F_kmin_pool'] = chi, F
    res['xi2_pool'] = math.sqrt(max(chi / F - 1, 0)) / (2 * math.sin(math.pi / a.L))
    res['L_over_xi'] = a.L / res['xi2_pool'] if res['xi2_pool'] > 0 else None
    t = time.perf_counter()
    base = landau.from_pool(pool) if a.base == 'landau' else None
    model, _ = phi4u.fit_coupled(pool, a.seed + 1, base=base, zm_ot=a.zm_ot)
    res['fit_seconds'] = time.perf_counter() - t

    N = 4 * (a.n + a.warm + 1)

    def proposals():
        g = torch.Generator().manual_seed(a.seed + 60)
        lw, O = [], {}
        with torch.no_grad():
            for off in range(0, N, 512):
                z = phi4u.base_noise((min(512, N - off),) + (a.L,) * a.d, g, base)
                x, ld = model.transform(z, a.steps)
                lw.append((-phi4.action(x, a.lam, a.kappa) - (model.logp(z) - ld)).numpy())
                for k, v in obs(x, a.d).items():
                    O.setdefault(k, []).append(v.numpy())
        return lw, O

    t = time.perf_counter()
    for mu_ in (.25, .0625, .015625):              # finer sub-cycling if the implicit midpoint fails
        model.micro_u = mu_
        try:
            lw, O = proposals()
            break
        except RuntimeError as ex:
            res.setdefault('midpoint_retries', []).append([mu_, repr(ex)])
    else:
        raise RuntimeError('all sub-cycling levels failed')
    res['micro_u'] = mu_
    res['flow_seconds_per_proposal'] = (time.perf_counter() - t) / N
    lw = np.concatenate(lw)
    O = {k: np.concatenate(v) for k, v in O.items()}
    w = np.exp(lw - lw.max())
    res['flow_ESS'] = float(w.sum() ** 2 / (len(w) * (w * w).sum()))
    idx, acc = phi4.imh(lw, a.n, a.warm, a.seed + 61)
    idx, acc = idx[a.warm:], acc[a.warm:]
    res['imh_acceptance'] = float(acc.mean())
    rej = ~acc
    runs = [np.diff(np.flatnonzero(np.r_[True, acc[:, c], True])).max() for c in range(4)]
    res['imh_longest_rejection_run'] = int(max(runs))
    res['flow_imh'] = gamma({k: v[idx] for k, v in O.items()}, 'imh')

    t = time.perf_counter()
    ch, info = mcmc(a.L, a.kappa, 'hmcwolff', a.seed + 300, a.d, a.lam, a.hmc_chains, a.warm * 4, a.n,
                    eps_factor=a.eps_factor, local_every=5)
    res['hmc_seconds_per_update'] = info['seconds_per_update']      # one update of all hmc_chains chains
    res['hmc_acceptance'] = info['acceptance']
    E = ch['energy'] / a.d
    res['hmc_wolff'] = gamma({'absM': ch['absM'], 'M2': ch['M2'], 'M4': ch['M4'], 'E': E, 'phi2': ch['phi2']}, 'hmc')
    res['z_flow_minus_hmc'] = {k: (res['flow_imh'][k]['mean'] - res['hmc_wolff'][k]['mean'])
                               / math.hypot(res['flow_imh'][k]['err'], res['hmc_wolff'][k]['err']) for k in res['flow_imh']}
    # end-to-end cost per effectively independent sample of |M| (same threads for both):
    # flow: proposals cost s_prop each, IMH chain gives 1 / (2 tau) effective samples per step; training = pool + fit
    tf, th = res['flow_imh']['absM']['tau_int'], res['hmc_wolff']['absM']['tau_int']
    res['flow_seconds_per_Neff'] = 2 * tf * res['flow_seconds_per_proposal']
    res['flow_training_seconds'] = res['pool_seconds'] + res['fit_seconds']
    res['hmc_seconds_per_Neff'] = 2 * th * res['hmc_seconds_per_update'] / a.hmc_chains
    res['seconds'] = time.perf_counter() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
