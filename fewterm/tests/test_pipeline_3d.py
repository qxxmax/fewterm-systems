"""3D end-to-end smoke (code paths of step 1): a small HMC+Wolff pool, the D192/D224 fit with all post-fit checks
(step doubling, inverse, L = 4 autograd log-det, divergence), IS ESS and IMH on a fixed proposal count.
Not part of the fast suite: run with FEWTERM_SLOW=1 (about 55 s, dominated by check_extended)."""
import math
import os

import numpy as np
import pytest

from fewterm import phi4 as P
from fewterm import wolff as W


@pytest.mark.skipif(os.environ.get('FEWTERM_SLOW') != '1',
                    reason='slow (~55 s, post-fit autograd checks); set FEWTERM_SLOW=1')
def test_pipeline_3d_smoke(tmp_path):
    d, L, kappa = 3, 4, .18
    fields, report = W.fit_pool(L, kappa, 4242, tmp_path, d=d, n=256, warmup=200, report_every=100)
    assert fields.shape == (256, 4, 4, 4) and report['acceptance_last256'] > .9
    assert (tmp_path / 'fit_pool.pt').exists()
    m192, s192, model, accepted = P.fit_d224(fields, 4242, kappa, tmp_path)
    assert s192 is not None and accepted is not None
    steps = max(32, 16 * math.ceil(model.bound() / 16))
    f, obs = P.generate(model, steps, 2048, 99, kappa)
    e = P.ess(f['logw'])
    idx, acc = P.imh(f['logw'][:4 * (300 + 100 + 1)], 300, 100, 7)
    print('3D smoke: steps', steps, 'accepted', accepted, 'ESS', e, 'IMH acceptance', acc[100:].mean())
    assert 0 < e <= 1 and np.isfinite(f['logw']).all() and set(obs) == set(P.KEYS)
