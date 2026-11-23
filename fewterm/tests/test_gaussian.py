"""Gate 1 (Gaussian limit), d = 3, L = 4, lambda = 0, kappa = 0.1 (free field, m0^2 = 1 - 2 d kappa = 0.4).

Target p ~ exp(-S), S = sum_k |phi_k|^2 [(1 - 2 lam) - 2 kappa sum_mu cos k_mu], so Var(phi_k) = sigma_k^2 =
1 / (2 (1 - 2 lam - 2 kappa sum_mu cos k_mu)).  On the straight path x_t = (1-t) z + t phi the exact velocity is
linear and diagonal in k:  A_t(k) = (t sigma_k^2 - (1-t)) / ((1-t)^2 + t^2 sigma_k^2)
                                  = alpha_t + beta_t / (s_t + khat^2),
a single rational pole whose position s_t moves with t.  The basis has c0 - c1 khat^2 + four fixed poles
sigma/(sigma + khat^2), sigma in {1/16, 1/4, 1, 4}, plus c3 M for the zero mode, so the exact velocity is
represented to high accuracy but not exactly (only when s_t hits one of the fixed poles).

Tolerances (measured values in README):
 * exact (noise-free) velocity as the regression target: the remaining error is pole approximation + 17-node
   time interpolation + 32-step Strang splitting.  Measured 1 - ESS = 5e-5 (d=3, L=4) and 1e-4 (d=2, L=8);
   the test requires 1 - ESS < 1e-3.
 * conditional target u = phi_1 - z on N exact samples (the production protocol): 1 - ESS ~ Var(log w) is
   dominated by coefficient noise ~ p / N_train; measured 0.024-0.039 at N = 1024 and 0.007-0.012 at N = 4096
   (two seeds).  The test requires ESS > 0.93 at N = 1024, ESS > 0.98 at N = 4096 and ESS(4096) > ESS(1024)."""
import math

import pytest
import torch

from fewterm import phi4 as P

D, L, KAPPA, LAM0 = 3, 4, .1, 0.


def variance(L, d, kappa, lam=0.):
    _, _, k2 = P.lattice(L, d)
    return 1 / (2 * ((1 - 2 * lam) - 2 * kappa * (d - k2 / 2)))


def exact_samples(n, seed, L=L, d=D, kappa=KAPPA):
    ax = P.axes(d)
    xi = torch.randn((n,) + (L,) * d, dtype=P.DT, generator=torch.Generator().manual_seed(seed))
    return torch.fft.ifftn(torch.fft.fftn(xi, dim=ax, norm='ortho') * variance(L, d, kappa).sqrt(),
                           dim=ax, norm='ortho').real


def exact_targets(data, L=L, d=D, kappa=KAPPA):
    var, ax = variance(L, d, kappa), P.axes(d)
    out = []
    for t, x in zip(data['times'], data['paths']):
        A = (t * var - (1 - t)) / ((1 - t) ** 2 + t * t * var)
        out.append(torch.fft.ifftn(torch.fft.fftn(x, dim=ax, norm='ortho') * A, dim=ax, norm='ortho').real)
    return torch.stack(out)


def fit_and_ess(n, seed, exact):
    data = P.data_paths(exact_samples(n, seed), 1000 + seed)
    if exact:
        data['targets'] = exact_targets(data)
    m192, _ = P.fit_stages(data, None, L, KAPPA, check=False, lam=LAM0)
    model = P.local_stage(m192, data, None)
    steps = max(32, 16 * math.ceil(model.bound() / 16))
    g, _ = P.generate(model, steps, 4096, 77, KAPPA, lam=LAM0, with_obs=False)
    return P.ess(g['logw'])


def test_exact_samples_have_target_covariance():
    x = exact_samples(4096, 9)
    power = torch.fft.fftn(x, dim=P.axes(D), norm='ortho').abs().square().mean(0)
    rel = (power / variance(L, D, KAPPA) - 1).abs().max()
    assert float(rel) < 5 * math.sqrt(2 / 4096)  # 5 sd of a chi^2_1 mean (complex modes fluctuate less)


def test_exact_velocity_gives_unit_ess():
    e = fit_and_ess(1024, 5, exact=True)
    print('Gaussian exact-target ESS', e)
    assert 1 - e < 1e-3


def test_conditional_target_ess():
    e1 = fit_and_ess(1024, 5, exact=False)
    e4 = fit_and_ess(4096, 5, exact=False)
    print('Gaussian conditional-target ESS N=1024, 4096:', e1, e4)
    assert e1 > .93 and e4 > .98 and e4 > e1
