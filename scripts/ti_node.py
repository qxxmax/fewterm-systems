"""One node of the thermodynamic integration in kappa with a single pure-HMC chain (settings of Singha et al.:
trajectory length 1, 10 leapfrog steps of eps = 0.1):  d log Z / d kappa = 2 < sum_x sum_mu phi_x phi_{x+mu} >,
mean and error from the Gamma method (pyerrors, S = 2).  The integral over the nodes is done in the analysis.

    python scripts/ti_node.py --L 16 --kappa 0.2500 --out FILE
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import pyerrors as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewterm.wolff import mcmc  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, default=2)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, default=.022)
    p.add_argument('--eps', type=float, default=.1)
    p.add_argument('--warm', type=int, default=10000)
    p.add_argument('--n', type=int, default=200000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    V = a.L ** a.d
    eps0 = .1 * min(1., math.sqrt(16 / a.L))
    t = time.perf_counter()
    ch, info = mcmc(a.L, a.kappa, 'hmc', a.seed + int(round(a.kappa * 1e4)), a.d, a.lam, 1, a.warm, a.n,
                    eps_factor=a.eps / eps0 * (1 + 1e-9))
    o = pe.Obs([2 * V * ch['energy'][:, 0]], ['ti|r0'])
    o.gamma_method(S=2.0)
    res = {'L': a.L, 'kappa': a.kappa, 'dlogZ': float(o.value), 'err': float(o.dvalue),
           'tau_int': float(o.e_tauint['ti']), 'epsilon': info['epsilon'], 'n_leap': info['n_leap'],
           'acceptance': info['acceptance'], 'n': a.n, 'warm': a.warm, 'seconds': time.perf_counter() - t}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
