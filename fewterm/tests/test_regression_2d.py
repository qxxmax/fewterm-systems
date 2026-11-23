"""Gate 0c: the d-generic code at d = 2 reproduces the stored teacher-free D192/D224 fit at L = 8, kappa = 0.270,
rep 0 (run written by S0/e18_csd.py flow).  The reference (pool, fitted coefficients, record.json and the first
1024 IS proposals) is stored in data/phi4_regress_L8_k0.270 (about 0.9 MB; override with FEWTERM_PHI4_REF).
If it is missing the tests skip; regenerate a pool with the fewterm.wolff CLI (see fewterm/README.md), but note
that only the stored pool reproduces the stored coefficients bit for bit.

Compared: coefficients of every stage (initial 72, time 128, radial D192, D224) and the stored step-guard ESS
(8192 proposals, seed pool+55, 32 steps) plus the first 1024 stored IS log-weights (seed pool+50).  The full
65536-proposal IS ESS and check_extended are in regress_2d.py (run once, see README)."""
import json
import math

import numpy as np
import pytest
import torch

from fewterm import phi4 as P
from fewterm.paths import phi4_ref_dir

POOL = phi4_ref_dir() / 'pool_0'
RUN = phi4_ref_dir() / 'flow'
KAPPA = .27

pytestmark = pytest.mark.skipif(not (POOL / 'fit_pool.pt').exists() or not (RUN / 'record.json').exists(),
                                reason=f'stored 2D reference not found under {phi4_ref_dir()} (data/phi4_regress_L8_k0.270 '
                                       'or FEWTERM_PHI4_REF); regenerate pools with python -m fewterm.wolff')


def stored(path):
    return json.loads((RUN / path).read_text(encoding='utf-8'))


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.fixture(scope='module')
def fitted():
    seed = json.loads((POOL / 'fit_pool_setup.json').read_text(encoding='utf-8'))['seed']
    fields = torch.load(POOL / 'fit_pool.pt', map_location='cpu', weights_only=True).double()
    assert fields.shape == (1024, 8, 8)
    data = P.data_paths(fields, seed)
    m72 = P.initial_fit(data, 8)
    m128 = P.constrained_fit(data, m72, None, False, 8, KAPPA)
    m192 = P.constrained_fit(data, m128, None, True, 8, KAPPA)
    m224 = P.local_stage(m192, data, None, 'local')
    return seed, m72, m128, m192, m224


def test_coefficients(fitted):
    _, m72, m128, m192, m224 = fitted
    pairs = {'initial': (m72, 'D192/initial_fit.json'), 'time': (m128, 'D192/time_fit/fit.json'),
             'D192': (m192, 'D192/radial_fit/fit.json'), 'D224': (m224, 'D224/fit.json')}
    out = {}
    for name, (model, path) in pairs.items():
        spec = stored(path)
        out[name] = rel(model.coeff.numpy(), spec['coeff'])
        assert model.sphi == spec['sphi'] and model.sm == spec['sm']
    print('coefficient max relative difference', out)
    assert max(out.values()) < 1e-6, out


def test_ess_and_logw(fitted):
    seed, _, _, _, model = fitted
    rec = stored('record.json')
    steps = max(32, 16 * math.ceil(model.bound() / 16))
    assert steps == rec['steps'] == 32
    assert abs(model.bound() - rec['bound']) < 1e-6 * rec['bound']
    g, _ = P.generate(model, steps, 8192, seed + 55, KAPPA, with_obs=False)
    e = P.ess(g['logw'])
    print('ESS_steps new/stored', e, rec['step_guard']['ESS_steps'])
    assert abs(e - rec['step_guard']['ESS_steps']) < 1e-6 * rec['step_guard']['ESS_steps']
    f, obs = P.generate(model, steps, 1024, seed + 50, KAPPA)
    with np.load(RUN / 'proposals_head1024.npz') as s:
        for k in ('logw', 'logdet', 'M', 'energy', 'z2'):
            assert np.abs(f[k] - s[k][:1024]).max() < 1e-8 * max(1., np.abs(s[k][:1024]).max()), k
        for k in P.KEYS:
            assert np.abs(obs[k] - s['obs_' + k][:1024]).max() < 1e-8 * max(1., np.abs(s['obs_' + k][:1024]).max()), k
