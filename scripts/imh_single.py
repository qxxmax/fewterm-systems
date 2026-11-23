"""IMH autocorrelation exactly as in Singha et al. (arXiv:2604.10209): one IMH chain of 5x10^4 steps after a burn-in
of 2x10^3, built from fresh independent proposals of the fitted few-parameter flow; tau_int of |M|, M, M^2 with the
Gamma method (Wolff 2004 automatic windowing; pyerrors, S = 2).  Input: OUT.npz of landau_compare.py --save-npz
(landau_ot_y, landau_ot_logw, >= 52000 proposals).

    python scripts/imh_single.py --npz FILE --L 16 --seed 1 --out FILE
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_compare import gamma  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--d', type=int, default=2)
    p.add_argument('--burn', type=int, default=2000)
    p.add_argument('--n', type=int, default=50000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    z = np.load(a.npz)
    y, lw = z['landau_ot_y'], z['landau_ot_logw']
    assert len(lw) >= a.burn + a.n, (len(lw), a.burn + a.n)
    rng = np.random.default_rng(a.seed)
    u = np.log(rng.random(a.burn + a.n))
    idx, cur, acc = np.empty(a.burn + a.n, int), 0, 0
    for i in range(a.burn + a.n):                       # candidate i proposed at step i; start from candidate 0
        if i > 0 and u[i] < lw[i] - lw[cur]:
            cur = i; acc += i >= a.burn
        idx[i] = cur
    m = (y[idx] / np.sqrt(a.L ** a.d))[a.burn:]
    g = gamma({'absM': np.abs(m)[:, None], 'M': m[:, None], 'M2': (m ** 2)[:, None], 'M4': (m ** 4)[:, None]}, 'imh')
    res = {'L': a.L, 'burn': a.burn, 'n': a.n, 'acceptance': acc / a.n, 'obs': g, 'npz': str(a.npz)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v['tau_int'] for k, v in g.items()}))


if __name__ == '__main__':
    main()
