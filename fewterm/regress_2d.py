"""One-off 2D regression of the full pipeline: fewterm.phi4.flow (d = 2) on the stored L = 8, kappa = 0.270, rep-0
pool, compared field by field with the stored S0/e18_csd.py flow run.

    python -m fewterm.regress_2d --out <new folder> [--pool <pool_0>] [--ref-run <stored flow folder>]

Defaults: --pool and --ref-run point to data/phi4_regress_L8_k0.270 (paths.phi4_ref_dir()).  The repo copy holds
only the first 1024 IS proposals (proposals_head1024.npz) and no chains.npz; with the complete stored run as
--ref-run (proposals.npz, chains.npz) every array is compared in full.
"""
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fewterm import phi4 as P
from fewterm.paths import phi4_ref_dir


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = np.abs(b).max()
    return float(np.abs(a - b).max() / den) if den > 0 else float(np.abs(a - b).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pool', type=Path, default=phi4_ref_dir() / 'pool_0')
    p.add_argument('--ref-run', type=Path, default=phi4_ref_dir() / 'flow')
    a = p.parse_args()
    POOL, RUN = a.pool, a.ref_run
    torch.set_num_threads(1)
    seed = json.loads((POOL / 'fit_pool_setup.json').read_text(encoding='utf-8'))['seed']
    fields = torch.load(POOL / 'fit_pool.pt', map_location='cpu', weights_only=True).double()
    a.out.mkdir(parents=True)
    rec = P.flow(fields, seed, .27, a.out)
    old = json.loads((RUN / 'record.json').read_text(encoding='utf-8'))
    cmp = {}
    for path in ('D192/initial_fit.json', 'D192/time_fit/fit.json', 'D192/radial_fit/fit.json', 'D224/fit.json'):
        new = json.loads((a.out / path).read_text(encoding='utf-8'))['coeff']
        cmp['coeff ' + path] = rel(new, json.loads((RUN / path).read_text(encoding='utf-8'))['coeff'])
    for path in ('D192/radial_fit/step_convergence.json', 'D224/step_convergence.json'):
        cmp['steps ' + path] = [json.loads((a.out / path).read_text())['steps'], json.loads((RUN / path).read_text())['steps']]
    for k in ('steps', 'accepted_steps_continuous', 'bound', 'ESS', 'Var_logw_per_site', 'acceptance',
              'ratios', 'sign_changes_per_chain', 'positive_fraction_proposals'):
        if k in old:
            cmp[k] = {'new': rec[k], 'stored': old[k],
                      'rel': rel(rec[k], old[k]) if not isinstance(old[k], int) else rec[k] - old[k]}
    for k in ('ESS_steps', 'ESS_accepted'):
        cmp['step_guard ' + k] = {'new': rec['step_guard'][k], 'stored': old['step_guard'][k],
                                  'rel': rel(rec['step_guard'][k], old['step_guard'][k])}
    full = (RUN / 'proposals.npz').exists()
    with np.load(a.out / 'proposals.npz') as n, np.load(RUN / ('proposals.npz' if full else 'proposals_head1024.npz')) as s:
        cmp['proposals compared'] = 'all' if full else 'first 1024'
        cmp['proposals.npz max rel'] = {k: rel(n[k][:len(s[k])], s[k]) for k in s.files}
    cmp['chains.npz max rel'] = {}
    if (RUN / 'chains.npz').exists():
        with np.load(a.out / 'chains.npz') as n, np.load(RUN / 'chains.npz') as s:
            cmp['chains.npz max rel'] = {k: rel(n[k], s[k]) for k in s.files}
    worst = max([v for k, v in cmp.items() if k.startswith('coeff')]
                + [v['rel'] for k, v in cmp.items() if isinstance(v, dict) and 'rel' in v and not isinstance(v['rel'], int)]
                + list(cmp['proposals.npz max rel'].values()) + list(cmp['chains.npz max rel'].values()))
    cmp['worst_relative'] = worst
    P.write(a.out / 'regression_vs_stored.json', cmp)
    print(json.dumps(cmp, indent=1, default=P.encode), flush=True)


if __name__ == '__main__':
    main()
