"""Plain HMC (no Wolff, no local sweeps) vs HMC+Wolff: tau_int of |M|, M^2, E with the Gamma method (same setup as
tau_compare: 64 batched chains, 4 threads).

    python scripts/hmc_tau.py --L 16 --kappa 0.2705 --lam 0.022 --out FILE
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewterm.wolff import mcmc  # noqa: E402
from tau_compare import gamma  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, default=2)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--chains', type=int, default=64)
    p.add_argument('--warm', type=int, default=4096)
    p.add_argument('--n', type=int, default=65536)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save-m', action='store_true', help='also write OUT.npz with the M traces (every --m-stride update)')
    p.add_argument('--m-stride', type=int, default=8)
    p.add_argument('--kernels', default='hmc,hmcwolff', help='comma list of hmc, hmcwolff')
    p.add_argument('--eps', type=float, default=None,
                   help='fixed leapfrog step (n_leap = round(1/eps), trajectory length 1), e.g. 0.1 as in Singha et al.')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    res = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    traces = {}
    eps_factor = 1.
    if a.eps is not None:                                # step_size() gives eps0 = 0.1 min(1, sqrt(16/L)) (d=2)
        import math
        eps0 = .1 * min(1., math.sqrt(16 / a.L)) if a.d == 2 else .1 * min(1., (256 / a.L ** a.d) ** .25)
        eps_factor = a.eps / eps0 * (1 + 1e-9)          # n_leap = ceil(1/(eps0 * factor)) = round(1/eps)
    res['eps_factor'] = eps_factor
    for kernel, local in [kl for kl in (('hmc', 0), ('hmcwolff', 5)) if kl[0] in a.kernels.split(',')]:
        t = time.perf_counter()
        ch, info = mcmc(a.L, a.kappa, kernel, a.seed + 300, a.d, a.lam, a.chains, a.warm, a.n, local_every=local,
                        eps_factor=eps_factor)
        traces[kernel] = ch['M'][::a.m_stride].astype('float32')
        E = ch['energy'] / a.d
        g = gamma({'absM': ch['absM'], 'M': ch['M'], 'M2': ch['M2'], 'M4': ch['M4'], 'E': E, 'phi2': ch['phi2']}, kernel)
        res[kernel] = {'obs': g, 'seconds_per_update': info['seconds_per_update'], 'acceptance': info['acceptance'],
                       'epsilon': info['epsilon'], 'n_leap': info['n_leap'],
                       'local_every': local, 'seconds': time.perf_counter() - t,
                       'seconds_per_Neff_absM': 2 * g['absM']['tau_int'] * info['seconds_per_update'] / a.chains}
        print(kernel, json.dumps({k: v for k, v in res[kernel].items() if k != 'obs'}), g['absM'], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    if a.save_m:
        import numpy as np
        np.savez_compressed(a.out.with_suffix('.npz'), **traces)


if __name__ == '__main__':
    main()
