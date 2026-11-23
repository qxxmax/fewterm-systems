"""HMC + Wolff in d dimensions: neighbour table, 2D bit-for-bit agreement with the vendor Wolff code and the
package HMC step, d-generic observables vs the 2D diagnostics, and 3D L = 4 stationarity checks:
  * HMC+Wolff vs plain HMC on <phi^2>, <|M|>, <energy> (64 independent chains each, z-scores from chain means);
  * Wolff-only updates applied to HMC-equilibrated configurations (paired before/after z-scores);
  * mutation checks: with the bond factor 2 kappa or 8 kappa instead of 4 kappa both checks must fail."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from fewterm import phi4 as P
from fewterm import wolff as W

from fewterm.paths import THIRD_PARTY

VENDOR = THIRD_PARTY / 'phi4_wolff' / 'src'      # phi4-wolff-main/src (third_party/SOURCES.md)
PKG = THIRD_PARTY / 'phi4_nf'                    # phi4_explicit_nf_20261017 subset


@pytest.mark.parametrize('d,L', [(2, 4), (2, 5), (3, 3), (3, 4)])
def test_neighbors_match_roll(d, L):
    nb = W.build_neighbors(L, d)
    idx = torch.arange(L ** d).reshape((L,) * d)
    k = 0
    for mu in range(d - 1, -1, -1):
        assert np.array_equal(nb[:, k], idx.roll(-1, mu).reshape(-1).numpy())      # site + e_mu
        assert np.array_equal(nb[:, k + 1], idx.roll(1, mu).reshape(-1).numpy())   # site - e_mu
        k += 2


@pytest.mark.skipif(not VENDOR.exists(), reason='vendor Wolff code not available')
def test_wolff_2d_bitwise_vendor():
    sys.path.insert(0, str(VENDOR))
    from hmc_wolff_tau_phi4_fullsave import _wolff_batch, _build_nb_rect
    L, n = 8, 16
    nb = _build_nb_rect(L, L)
    assert np.array_equal(nb, W.build_neighbors(L, 2))
    phi = torch.randn(n, L, L, dtype=P.DT, generator=torch.Generator().manual_seed(3)) + .8
    a = phi.numpy().reshape(n, L * L).copy()
    b = a.copy()
    W.seed_numba(11)
    sa = W.wolff_batch(a, nb, .27, n, L * L)
    W.seed_numba(11)
    sb = _wolff_batch(b, nb, .27, n, L * L)
    assert np.array_equal(a, b) and np.array_equal(sa, sb) and sa.max() > 1


@pytest.mark.skipif(not PKG.exists(), reason='phi4_nf package not available')
def test_hmc_and_observables_2d_bitwise_package():
    sys.path.insert(0, str(PKG))
    from phi4_nf.kernels import IdentityMass, hmc_step
    from phi4_nf.diagnostics import observables
    g = torch.Generator().manual_seed(5)
    phi = W.initial(8, .27, 8, 17)
    eps, n_leap = W.step_size(8, 2)
    assert (eps, n_leap) == (.1, 10)
    for _ in range(3):
        noise = torch.randn(phi.shape, dtype=P.DT, generator=g)
        u = torch.rand(8, dtype=P.DT, generator=g)
        a, ia = W.hmc_step(phi, eps, n_leap, kappa=.27, noise=noise, uniform=u)
        b, ib = hmc_step(phi, IdentityMass(), eps, n_leap, kappa=.27, noise=noise, uniform=u)
        assert torch.equal(a, b) and torch.equal(ia['accepted'], ib['accepted'])
        phi = a
    oa, ob = P.observables(phi, .022, .27, 2), observables(phi, .022, .27)
    for k in P.KEYS:
        assert np.array_equal(oa[k], ob[k]), k


def test_force_is_minus_gradient_3d():
    phi = torch.randn(2, 3, 4, 5, dtype=P.DT, generator=torch.Generator().manual_seed(1), requires_grad=True)
    S = P.phi4_action(phi, .022, .18).sum()
    g = torch.autograd.grad(S, phi)[0]
    assert float((P.phi4_force(phi.detach(), .022, .18) + g).abs().max()) < 1e-12


D3, L3, NCH = 3, 4, 64
OBS = ('phi2', 'absM', 'energy')


@pytest.fixture(scope='module')
def hmc_reference():
    """Plain HMC, 64 chains x 4000 recorded updates at each kappa (shared by the check and its mutations)."""
    return {k: W.mcmc(L3, k, 'hmc', 123, d=D3, n_chains=NCH, warm=200, n_rec=4000) for k in (.17, .19)}


def hmcwolff_z(kappa, ref, bond_factor=4.):
    hw, iw = W.mcmc(L3, kappa, 'hmcwolff', 456, d=D3, n_chains=NCH, warm=200, n_rec=1000, bond_factor=bond_factor)
    hm, ih = ref
    assert iw['acceptance'] > .9 and ih['acceptance'] > .9
    z = {}
    for k in OBS:
        a, b = hw[k].mean(0), hm[k].mean(0)
        z[k] = (a.mean() - b.mean()) / math.hypot(a.std(ddof=1), b.std(ddof=1)) * math.sqrt(NCH)
    m = hw['M'].mean(0)
    z['M_wolff_zero'] = m.mean() / m.std(ddof=1) * math.sqrt(NCH)
    return z


@pytest.mark.parametrize('kappa', [.17, .19])
def test_3d_hmc_wolff_stationarity(kappa, hmc_reference):
    z = hmcwolff_z(kappa, hmc_reference[kappa])
    print('3D L=4 kappa', kappa, 'z', z)
    assert max(abs(v) for v in z.values()) < 4


@pytest.mark.parametrize('bond_factor', [2., 8.])
def test_3d_hmc_wolff_mutation_fails(bond_factor, hmc_reference):
    """Mutation check: the same comparison with a wrong bond probability 1 - exp(-f kappa phi_i phi_j), f = 2 or 8,
    must fail (observed max |z| about 35 and 45 at kappa = 0.17; at 0.19 the f = 8 kernel is not detected)."""
    z = hmcwolff_z(.17, hmc_reference[.17], bond_factor)
    print('mutation f =', bond_factor, 'z', z)
    assert max(abs(z[k]) for k in OBS) > 10


# ----------------------------------------------------------------------------- Wolff-only stationarity
N_EQ = 1024


def nn_correlation(phi, d):
    """(1/(d V)) sum_x sum_mu phi_x phi_{x+mu}."""
    e = phi * phi.roll(1, -1)
    for mu in range(2, d + 1):
        e = e + phi * phi.roll(1, -mu)
    return e.mean(P.axes(d)).numpy() / d


@pytest.fixture(scope='module')
def hmc_equilibrated():
    """1024 independent chains, 300 plain-HMC updates each from the half-hot/half-cold start (Z2-symmetric by
    construction, so Z2-even observables do not need tunnelling)."""
    out = {}
    for kappa in (.17, .19):
        phi = W.initial(L3, kappa, N_EQ, 7, D3)
        g = torch.Generator().manual_seed(8)
        eps, n_leap = W.step_size(L3, D3)
        for _ in range(300):
            phi, _ = W.hmc_step(phi, eps, n_leap, kappa=kappa, noise=torch.randn(phi.shape, dtype=P.DT, generator=g),
                                uniform=torch.rand(N_EQ, dtype=P.DT, generator=g))
        out[kappa] = phi
    return out


def wolff_only_z(phi, kappa, bond_factor=4., n_updates=50):
    """Paired z-scores (after - before) of action density, nn correlation, |M| and M^2 after `n_updates`
    single-cluster updates.  If pi is stationary under the kernel, E[after] = E[before] exactly."""
    nb = W.build_neighbors(L3, D3)
    W.seed_numba(9)
    x = phi.clone()
    for _ in range(n_updates):
        x, _ = W.wolff_update(x, nb, kappa, bond_factor)
    assert torch.equal(x.abs(), phi.abs())          # a cluster flip only changes signs
    feats = {'action_density': lambda u: (P.phi4_action(u, P.LAM, kappa) / L3 ** D3).numpy(),
             'nn_corr': lambda u: nn_correlation(u, D3),
             'absM': lambda u: u.mean(P.axes(D3)).abs().numpy(),
             'M2': lambda u: u.mean(P.axes(D3)).square().numpy()}
    z = {}
    for k, f in feats.items():
        diff = f(x) - f(phi)
        z[k] = diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))
    return z


@pytest.mark.parametrize('kappa', [.17, .19])
def test_3d_wolff_only_stationarity(kappa, hmc_equilibrated):
    z = wolff_only_z(hmc_equilibrated[kappa], kappa)
    print('Wolff-only kappa', kappa, 'z', z)
    assert max(abs(v) for v in z.values()) < 4


@pytest.mark.parametrize('bond_factor', [2., 8.])
def test_3d_wolff_only_mutation_fails(bond_factor, hmc_equilibrated):
    """Wrong bond factor: Wolff-only updates drift away from pi (observed |z| about 36 and 57 on the action at 0.17)."""
    z = wolff_only_z(hmc_equilibrated[.17], .17, bond_factor)
    print('Wolff-only mutation f =', bond_factor, 'z', z)
    assert max(abs(v) for v in z.values()) > 10


@pytest.mark.parametrize('n', [1, 5, 8])
def test_initial_odd_and_even_n(n):
    """initial(): first n//2 Gaussian, remaining n - n//2 signed classical minima (alternating sign) + 0.1 noise."""
    phi = W.initial(4, .19, n, 3, d=3)
    assert phi.shape == (n, 4, 4, 4)
    v = P.classical_minimum(.19, P.LAM, 3)
    means = phi[n // 2:].mean((1, 2, 3))
    expect = torch.tensor([v if i % 2 == 0 else -v for i in range(n - n // 2)], dtype=P.DT)
    assert float((means - expect).abs().max()) < .1
