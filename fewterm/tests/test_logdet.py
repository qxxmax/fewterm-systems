"""Gate 0a/0b: analytic log-det of every substep and of the full map equals the autograd full-Jacobian log-det
(float64, 1e-10); inverse round trip 1e-10; divergence = trace of the velocity Jacobian; symmetries."""
import numpy as np
import pytest
import torch

from fewterm import phi4 as P

TOL = 1e-10
GEOMS = [(2, 4), (2, 5), (3, 3), (3, 4)]
KINDS = ['operator', 'time', 'radial', 'local', 'local_small', 'fourier']


def make(kind, d, L, seed=31117000):
    g = torch.Generator().manual_seed(seed)
    if kind == 'operator':
        c = .12 * torch.randn(9, 8, dtype=P.DT, generator=g)
        return P.OperatorCNF(L, c, 1.1, .7, d=d), g
    n = {'time': 8, 'radial': 12}.get(kind, 14)
    c = .1 * torch.randn(17, n, dtype=P.DT, generator=g)
    c[0] = 0
    c[0, 0] = c[0, 3] = -1
    if kind == 'time':
        return P.TimeCNF(L, c, 1.1, .7, d=d), g
    if kind == 'radial':
        return P.RadialCNF(L, c, 1.1, .7, d=d), g
    return P.ExtCNF(L, c, 1.1, .7, kind, d=d), g


def jac_logdet(fn, x):
    shape = x.shape
    with torch.enable_grad():
        j = torch.autograd.functional.jacobian(lambda u: fn(u.reshape(shape)).reshape(-1), x.reshape(-1))
    sign, ld = torch.linalg.slogdet(j)
    return float(sign), float(ld), j


@pytest.mark.parametrize('d,L', GEOMS)
@pytest.mark.parametrize('kind', KINDS)
def test_substep_logdet(kind, d, L):
    model, g = make(kind, d, L)
    x = 1.5 * torch.randn((1,) + (L,) * d, dtype=P.DT, generator=g)
    c = model.at(.37)
    for part, tau in (('A', .21), ('B', .21), ('C', .43), ('A', -.17), ('B', -.17), ('C', -.3)):
        sign, ld, _ = jac_logdet(lambda u: model.substep(u, tau, c, part)[0], x)
        _, analytic = model.substep(x, tau, c, part)
        assert sign > 0
        assert abs(ld - float(analytic[0])) < TOL, (part, tau, ld, float(analytic[0]))


@pytest.mark.parametrize('d,L', GEOMS)
@pytest.mark.parametrize('kind', KINDS)
def test_full_map_logdet_inverse_divergence(kind, d, L):
    model, g = make(kind, d, L)
    steps = 16
    assert model.bound() / (4 * steps) <= .25
    x = torch.randn((3,) + (L,) * d, dtype=P.DT, generator=g)
    # full map: analytic vs autograd
    sign, ld, _ = jac_logdet(lambda u: model.transform(u, steps)[0], x[:1])
    y, lda = model.transform(x, steps)
    assert sign > 0
    assert abs(ld - float(lda[0])) < TOL
    # divergence = trace of the velocity Jacobian
    _, _, jv = jac_logdet(lambda u: model.velocity(u, .37), x[:1])
    assert abs(float(jv.trace()) - float(model.divergence(x[:1], .37)[0])) < TOL
    # inverse round trip (unit, large and shifted probes)
    probes = torch.cat([x, 10 * x, x + 3])
    yy, l1 = model.transform(probes, steps)
    zz, l2 = model.transform(yy, steps, inverse=True)
    assert float(((zz - probes).abs() / probes.abs().clamp_min(1)).max()) < TOL
    assert float((l1 + l2).abs().max()) < TOL
    assert float((model.logp(probes) - l1 - model.logp(zz) - l2).abs().max()) < TOL


@pytest.mark.parametrize('d,L', [(2, 4), (3, 4)])
def test_symmetries_and_limits(d, L):
    model, g = make('local', d, L)
    x = torch.randn((3,) + (L,) * d, dtype=P.DT, generator=g)
    y, ld = model.transform(x, 16)
    perm = (0,) + tuple(range(d, 0, -1))  # reverse lattice axes (rot90 analogue for d >= 2)
    for t in (lambda u: -u, lambda u: u.roll(1, -1), lambda u: u.roll(-2, 1), lambda u: u.permute(perm),
              lambda u: u.flip(-1)):
        yy, ll = model.transform(t(x), 16)
        assert float((yy - t(y)).abs().max()) < 1e-12
        assert float((ll - ld).abs().max()) < 1e-11
    # initial velocity is -x (prior score) and zero coefficients give the identity
    assert float((model.velocity(x, 0) + x).abs().max()) < 1e-13
    zero = P.ExtCNF(L, torch.zeros(17, 14), 1.1, .7, 'local', d=d)
    yy, ll = zero.transform(x, 8)
    assert float((yy - x).abs().max()) < 1e-14 and float(ll.abs().max()) == 0
    # purely linear rates: closed-form solution exp(rate) in Fourier space, logdet = sum of rates
    c = torch.zeros(17, 12, dtype=P.DT)
    c[:, P.LINEAR] = torch.tensor([.2, .1, .05, .04, .06, -.08, .02, -.01], dtype=P.DT)
    lin = P.RadialCNF(L, c, 1.1, .7, d=d)
    m = x.mean(lin.ax, keepdim=True)
    rate = lin.rates(lin.at(.5), m.square() / (.7 ** 2 + m.square()))
    yy, ll = lin.transform(x, 8)
    expected = m + lin.ifft(lin.fft(x - m) * rate.exp()).real
    assert float((yy - expected).abs().max()) < 1e-12
    assert float((ll - rate.sum(lin.ax)).abs().max()) < 1e-12
    # explicit d-dimensional divergence constants: trace(lap) = -2 d V, rank of the q^2 = 1 shell = 2 d
    assert lin.rank == 2 * d


def test_profiles_and_bounds():
    for f, s in ((P.r3, 1.1), (P.r5, .7)):
        u = torch.linspace(-20, 20, 301, dtype=P.DT, requires_grad=True)
        v, dv = f(u, s)
        assert float((torch.autograd.grad(v.sum(), u)[0] - dv).abs().max()) < 1e-12
    # Bernstein bound dominates the zero-mode derivative on a dense grid (fit_time_constraint selftest)
    model, _ = make('time', 3, 4)
    for c in model.coeff.numpy():
        m = np.linspace(-50, 50, 20001)
        mt = torch.as_tensor(m)
        dd = c[3] + c[4] * P.r3(mt, model.sm)[1].numpy() + c[5] * P.r5(mt, model.sm)[1].numpy()
        assert np.abs(dd).max() <= model.bound()


@pytest.mark.parametrize('d,L', [(2, 4), (2, 5), (2, 8), (3, 3), (3, 4), (3, 6)])
def test_laplacian_is_minus_khat2_in_fourier(d, L):
    """fft(laplacian(x)) = -khat^2 fft(x) with khat^2 = 4 sum_mu sin^2(pi q_mu / L): the real-space
    Laplacian used by the velocity and the Fourier rate used by the exact C substep are the same operator."""
    model, g = make('fourier', d, L)
    x = torch.randn((3,) + (L,) * d, dtype=P.DT, generator=g)
    lhs = model.fft(model.laplacian(x))
    rhs = -model.k2 * model.fft(x)
    assert float((lhs - rhs).abs().max()) < 1e-12


@pytest.mark.parametrize('L', [4, 5, 49, 98, 103, 107])
def test_lattice_frequencies_are_integers(L):
    """fftfreq(L, d=1/L) is not an exact integer for some L (49, 98, 103, ...); lattice() must still give exact
    integer q, so that the q^2 = 1 shell has rank 2d."""
    for d in (2, 3) if L < 50 else (2,):
        q2, mask, _ = P.lattice(L, d)
        assert torch.equal(q2, q2.round())
        assert int(mask.sum()) == 2 * d


def test_lattice_unchanged_for_production_sizes():
    """For L = 6..24 the rounded frequencies equal the raw fftfreq values bit for bit (2D results unchanged)."""
    for L in range(4, 25):
        f = torch.fft.fftfreq(L, d=1 / L, dtype=P.DT)
        assert torch.equal(f, f.round()), L
