"""ESS of the paper flow (CoupledCNF, Landau zero-mode base, sorted zero-mode pairing) vs number of fitting configurations.

    python scripts/datasize_landau.py --L 16 --kappa 0.2705 --seed 1 --out FILE

One pool of --pool chains (HMC with cluster updates, 4096 updates each, as in the paper).  For N in (1024, 4096, 16384):
Landau base from the first N configurations, fit_coupled on them with zm_ot, IS ESS (32 steps) on --n proposals.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewterm import landau, phi4u  # noqa: E402
from fewterm.wolff import fit_pool  # noqa: E402

torch.set_num_threads(int(os.environ.get('FEWTERM_THREADS', '1')))
torch.set_default_dtype(torch.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, default=0.022)
    p.add_argument('--pool', type=int, default=16384)
    p.add_argument('--n', type=int, default=16384)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    res = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    t0 = time.perf_counter()
    pool, rep = fit_pool(a.L, a.kappa, a.seed, None, 2, a.lam, a.pool, local_every=5)
    pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(a.seed + 7))]
    res['pool_seconds'], res['frozen_chains'] = rep['seconds'], rep['frozen_chains']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    N = 1024
    while N <= a.pool:
        t = time.perf_counter()
        fields = pool[:N]
        base = landau.from_pool(fields)
        model, _ = phi4u.fit_coupled(fields, a.seed + 1, base=base, zm_ot=True)
        fit_s = time.perf_counter() - t
        r = {'fit_seconds': fit_s}
        for mu_ in (.25, .0625, .015625):
            model.micro_u = mu_
            try:
                e, lw = phi4u.ess_of(model, 32, a.n, a.seed + 50, a.lam, a.kappa)
                r.update({'ESS': e, 'Var_logw': float(lw.var()), 'micro_u': mu_})
                break
            except RuntimeError as ex:
                r['error'] = repr(ex)
        res[f'N{N}'] = r
        a.out.write_text(json.dumps(res, indent=1))
        print(json.dumps({f'N{N}': r}), flush=True)
        N *= 4
    res['seconds'] = time.perf_counter() - t0
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
