"""Few-parameter flow for d-dimensional lattice phi^4: basis functions, splitting integrator and least-squares fits.

The velocity is a linear combination of field functions with time-dependent coefficients.  The classes add the
terms of the paper step by step:
  OperatorCNF   8 terms: psi, Laplacian psi, the projector p on the q^2 = 1 shell and h(m) p (Fourier operator),
                the local r3(phi_x, s_phi), and the zero-mode terms m, r3(m), r5(m)
  TimeCNF       the same terms at 17 time nodes (15 interior nodes plus the two fixed end nodes)
  RadialCNF     + four rational Fourier rates sigma / (sigma + khat^2): 12 terms
  ExtCNF        kind 'local': + r3 at the scales 2 s_phi and 4 s_phi: 14 terms
  phi4u.CoupledCNF   + the two nearest-neighbour terms: the 16-term flow of the paper (fit: phi4u.fit_coupled)
The names D192 and D224 in the fit functions below and in the stored test data are the 12- and 14-term fits with
bounded coefficients of an earlier development stage; they are kept for the bit-for-bit regression test.

Field shape: [batch, L, ..., L] (d lattice axes).  float64; tensors follow the device of the coefficients
(CPU default).  For d = 2 the code reproduces the stored 2D reference fit bit for bit (tests/test_regression_2d.py)."""
import json
import os
import math
from pathlib import Path
import time

import numpy as np
import scipy.linalg
from scipy.optimize import LinearConstraint, minimize
import torch

DT = torch.float64
LAM = .022
BOUND = 31.9
SCALES = (1 / 16, 1 / 4, 1., 4.)           # fit_radial_response.SCALES
LINEAR = [0, 1, 6, 7, 8, 9, 10, 11]          # fit_radial_response.LINEAR
EXTRA_SIGMA = (1 / 64, 16.)                  # e11_local_tail.EXTRA_SIGMA


# ----------------------------------------------------------------------------- io helpers (run_cnf)
def encode(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def write(path, obj):
    """run_cnf.write; a None folder disables writing (used by the tests)."""
    if path is None:
        return
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=encode)


def sub(out, name):
    return None if out is None else Path(out) / name


def as_coeff(c, device):
    if isinstance(c, (list, tuple)):
        c = np.asarray([np.asarray(r.cpu() if isinstance(r, torch.Tensor) else r, dtype=np.float64) for r in c])
    return torch.as_tensor(c, dtype=DT, device=device)


def axes(d):
    return tuple(range(-d, 0))


def corner(x, d, n=4):
    """x[:1, :n, :n] in d dimensions."""
    return x[(slice(0, 1),) + (slice(0, n),) * d]


# ----------------------------------------------------------------------------- scalar profiles (run_cnf)
def r3(u, s):
    w = (u / s).square()
    return u * s ** 2 * w / (1 + w), s ** 2 * w * (3 + w) / (1 + w).square()


def r5(u, s):
    w = (u / s).pow(4)
    return u * s ** 4 * w / (1 + w), s ** 4 * w * (5 + w) / (1 + w).square()


def timeweights(t):
    h = np.zeros(9)
    j = min(7, int(t * 8))
    h[j], h[j + 1] = 1 - (t * 8 - j), t * 8 - j
    return h


def bernstein(power):
    power = np.pad(np.asarray(power), (0, 7 - len(power)))
    return np.array([sum(power[j] * math.comb(k, j) / math.comb(6, j)
                         for j in range(k + 1)) for k in range(7)])


def derivative_matrix(sm, sp):
    """Rows 0-6: Bernstein bounds of the zero-mode map derivative; row 7: bound of the local r3 derivative.
    Dimension independent (both maps are scalar)."""
    mul = np.polynomial.polynomial.polymul
    D = np.array([1., -2., 2.])
    D2 = mul(D, D)
    p5 = np.pad(5 * mul([0., 0., 1.], D), (0, 1))
    p5[4] -= 4
    B = np.zeros((8, 8))
    denom = bernstein(D2)
    assert denom.min() > 0
    B[:7, 3] = 1
    B[:7, 4] = bernstein(sm ** 2 * mul([0., 3., -2.], D2)) / denom
    B[:7, 5] = bernstein(sm ** 4 * p5) / denom
    B[7, 2] = 9 / 8 * sp ** 2
    return B


def errors(pred, target, d=2):
    ax = axes(d)
    e = pred - target
    em = e.mean(ax, keepdims=True)
    ym = target.mean(ax, keepdims=True)
    return {k: {'mse': float((a * a).mean()), 'power': float((b * b).mean())}
            for k, a, b in (('all', e, target), ('zero', em, ym), ('nonzero', e - em, target - ym))}


# ----------------------------------------------------------------------------- lattice geometry
def lattice(L, d, shell=1, device='cpu'):
    """Integer frequencies, q^2-shell mask and khat^2 = 4 sum_mu sin^2(pi q_mu / L) on an L^d lattice."""
    f = torch.fft.fftfreq(L, d=1 / L, dtype=DT, device=device).round()   # exact integers for every L (49, 98, ...)

    def along(v, mu):
        shape = [1] * d
        shape[mu] = L
        return v.reshape(shape)
    q2 = along(f, 0).square()
    s2 = torch.sin(math.pi * along(f, 0) / L).square()
    for mu in range(1, d):
        q2 = q2 + along(f, mu).square()
        s2 = s2 + torch.sin(math.pi * along(f, mu) / L).square()
    return q2, (q2 == shell).double(), 4 * s2


# ----------------------------------------------------------------------------- flow classes
class OperatorCNF:
    """run_cnf.OperatorCNF in d dimensions; coeff (9, 8), linear in t between 9 nodes."""

    def __init__(self, L, coeff, sphi, sm, shell=1, d=2, device='cpu'):
        self.L, self.d, self.V = L, d, L ** d
        self.ax = axes(d)
        self.device = torch.device(device)
        self.coeff = as_coeff(coeff, self.device)
        self.sphi, self.sm = float(sphi), float(sm)
        if min(self.sphi, self.sm) <= 0 or self.coeff.shape != (9, 8):
            raise ValueError('invalid scales or coefficient shape')
        _, self.mask, self.k2 = lattice(L, d, shell, self.device)
        self.rank = int(self.mask.sum())
        self.shell = shell
        self.origin = (0,) * d

    def at(self, t):
        pos = float(t) * 8
        lo = min(7, int(pos))
        w = pos - lo
        return (1 - w) * self.coeff[lo] + w * self.coeff[lo + 1]

    def laplacian(self, psi):
        lap = psi.roll(1, -1) + psi.roll(-1, -1)
        for mu in range(2, self.d + 1):
            lap = lap + psi.roll(1, -mu) + psi.roll(-1, -mu)
        return lap - 2 * self.d * psi

    def fft(self, x):
        return torch.fft.fftn(x, dim=self.ax, norm='ortho')

    def ifft(self, x):
        return torch.fft.ifftn(x, dim=self.ax, norm='ortho')

    def basis(self, x):
        m = x.mean(self.ax, keepdim=True)
        psi = x - m
        lap = self.laplacian(psi)
        p = self.ifft(self.fft(psi) * self.mask).real
        h = m.square() / (self.sm ** 2 + m.square())
        return torch.stack((psi, lap, r3(x, self.sphi)[0], m.expand_as(x),
                            r3(m, self.sm)[0].expand_as(x), r5(m, self.sm)[0].expand_as(x), p, h * p), -1)

    def velocity(self, x, t):
        return (self.basis(x) * self.at(t)).sum(-1)

    def divergence(self, x, t):
        c = self.at(t)
        m = x.mean(self.ax)
        h = m.square() / (self.sm ** 2 + m.square())
        return ((self.V - 1) * c[0] - 2 * self.d * self.V * c[1]
                + c[2] * r3(x, self.sphi)[1].sum(self.ax) + c[3]
                + c[4] * r3(m, self.sm)[1] + c[5] * r5(m, self.sm)[1]
                + self.rank * (c[6] + c[7] * h))

    def bound(self):
        c = self.coeff.abs().max(0).values
        ka = float(c[3] + c[4] * 9 / 8 * self.sm ** 2 + c[5] * 25 / 16 * self.sm ** 4)
        kb = float(c[2] * 9 / 8 * self.sphi ** 2)
        return max(ka, kb)

    def scalar(self, x, tau, c, local):
        """Implicit-midpoint step of a scalar (site-wise or zero-mode) ODE; log-derivative is analytic."""
        def fun(u):
            if local:
                v, d = r3(u, self.sphi)
                return c[2] * v, c[2] * d
            v3, d3 = r3(u, self.sm)
            v5, d5 = r5(u, self.sm)
            return c[3] * u + c[4] * v3 + c[5] * v5, c[3] + c[4] * d3 + c[5] * d5
        return midpoint(fun, x, tau)

    def substep(self, x, tau, c, part):
        m = x.mean(self.ax, keepdim=True)
        if part == 'A':
            y, ld = self.scalar(m, tau, c, False)
            return x + y - m, ld.reshape(-1)
        if part == 'B':
            y, ld = self.scalar(x, tau, c, True)
            return y, ld.sum(self.ax)
        h = m.square() / (self.sm ** 2 + m.square())
        rates = c[0] - c[1] * self.k2 + (c[6] + c[7] * h) * self.mask
        zero = torch.zeros_like(self.mask)
        zero[self.origin] = 1
        rates = rates * (1 - zero)
        ft = self.fft(x - m)
        y = m + self.ifft(ft * torch.exp(tau * rates)).real
        return y, tau * rates.sum(self.ax)

    def transform(self, x, steps, inverse=False):
        """Strang splitting A/2 B/2 C B/2 A/2 per step; returns (field, log|det J|) with analytic log-det."""
        if self.bound() / (4 * steps) > .250000000001:
            raise ValueError('time step violates derivative bound')
        dt = (-1 if inverse else 1) / steps
        ld = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        order = reversed(range(steps)) if inverse else range(steps)
        for j in order:
            c = self.at((j + .5) / steps)
            for part, frac in (('A', .5), ('B', .5), ('C', 1), ('B', .5), ('A', .5)):
                x, change = self.substep(x, dt * frac, c, part)
                ld = ld + change
        if not bool(torch.isfinite(x).all() and torch.isfinite(ld).all()):
            raise RuntimeError('nonfinite map; no samples filtered')
        return x, ld

    def logp(self, x):
        return -.5 * (x.square().sum(self.ax) + self.V * math.log(2 * math.pi))


def midpoint(fun, x, tau):
    """run_cnf.OperatorCNF.scalar / e11.ExtCNF.scalar body: Newton solve of mid = x + tau/2 v(mid)."""
    mid = x
    for _ in range(20):
        v, d = fun(mid)
        residual = mid - x - .5 * tau * v
        mid = mid - residual / (1 - .5 * tau * d)
        if float((residual.detach().abs() / x.detach().abs().clamp_min(1)).max()) < 5e-13:
            break
    v, d = fun(mid)
    residual = mid - x - .5 * tau * v
    if not bool(torch.isfinite(mid).all()) or float((residual.detach().abs() / x.detach().abs().clamp_min(1)).max()) >= 5e-13:
        raise RuntimeError('implicit midpoint root did not converge')
    u = .5 * tau * d
    if float(u.detach().abs().max()) >= 1:
        raise RuntimeError('noninvertible midpoint step')
    return 2 * mid - x, torch.log1p(u) - torch.log1p(-u)


class TimeCNF(OperatorCNF):
    """fit_time_constraint.TimeCNF: 17 time nodes, Bernstein derivative bound."""

    def __init__(self, L, coeff, sphi, sm, d=2, device='cpu'):
        super().__init__(L, np.zeros((9, 8)), sphi, sm, d=d, device=device)
        self.coeff = as_coeff(coeff, self.device)
        if self.coeff.shape != (17, 8):
            raise ValueError('expected seventeen time nodes')
        self.derivative_constraints = derivative_matrix(self.sm, self.sphi)

    def at(self, t):
        pos = float(t) * 16
        lo = min(15, int(pos))
        w = pos - lo
        return (1 - w) * self.coeff[lo] + w * self.coeff[lo + 1]

    def bound(self):
        return float(np.abs(self.coeff.cpu().numpy() @ self.derivative_constraints.T).max()) + 1e-9


class RadialCNF(TimeCNF):
    """fit_radial_response.RadialCNF: four rational Fourier rates sigma/(sigma + khat^2) (the 192 family)."""

    def __init__(self, L, coeff, sphi, sm, d=2, device='cpu'):
        c = as_coeff(coeff, device)
        if c.shape != (17, 12):
            raise ValueError('expected seventeen by twelve coefficients')
        super().__init__(L, c[:, :8], sphi, sm, d=d, device=device)
        self.coeff = c
        self.radial = torch.stack([s / (s + self.k2) for s in SCALES])
        self.radial[(slice(None),) + self.origin] = 0

    def bound(self):
        return float(np.abs(self.coeff[:, :8].cpu().numpy() @ self.derivative_constraints.T).max()) + 1e-9

    def bcast(self, c):
        return c.reshape((-1,) + (1,) * self.d)

    def extra_basis(self, x):
        ft = self.fft(x - x.mean(self.ax, keepdim=True))
        return self.ifft(ft[:, None] * self.radial).real.movedim(1, -1)

    def velocity(self, x, t):
        c = self.at(t)
        return (self.basis(x) * c[:8]).sum(-1) + (self.extra_basis(x) * c[8:]).sum(-1)

    def divergence(self, x, t):
        return super().divergence(x, t) + (self.at(t)[8:] * self.radial.sum(self.ax)).sum()

    def rates(self, c, h):
        rate = c[0] - c[1] * self.k2 + (c[6] + c[7] * h) * self.mask
        rate = rate + (self.bcast(c[8:]) * self.radial).sum(0)
        nonzero = torch.ones_like(self.mask)
        nonzero[self.origin] = 0
        return rate * nonzero

    def substep(self, x, tau, c, part):
        if part != 'C':
            return super().substep(x, tau, c, part)
        m = x.mean(self.ax, keepdim=True)
        rate = self.rates(c, m.square() / (self.sm ** 2 + m.square()))
        y = m + self.ifft(self.fft(x - m) * torch.exp(tau * rate)).real
        return y, tau * rate.sum(self.ax)

    def rate_matrix(self):
        matrices = []
        for h in (0., 1.):
            cols = []
            for j in LINEAR:
                c = torch.zeros(12, dtype=DT, device=self.device)
                c[j] = 1
                cols.append(self.rates(c, h).reshape(-1).cpu().numpy())
            matrices.append(np.stack(cols, axis=-1))
        return np.concatenate(matrices)


class ExtCNF(RadialCNF):
    """e11_local_tail.ExtCNF: two extra local (kind local/local_small) or Fourier (kind fourier) terms.
    kind='local' on a RadialCNF parent is the 224 family."""

    def __init__(self, L, coeff, sphi, sm, kind, d=2, device='cpu'):
        c = as_coeff(coeff, device)
        if c.shape != (17, 14) or kind not in ('local', 'local_small', 'fourier'):
            raise ValueError('expected (17, 14) coefficients and a valid kind')
        super().__init__(L, c[:, :12], sphi, sm, d=d, device=device)
        self.coeff, self.kind = c, kind
        self.local_scales = (self.sphi / 2, self.sphi / 4) if kind == 'local_small' else (2 * self.sphi, 4 * self.sphi)
        self.is_local = kind in ('local', 'local_small')
        self.extra_radial = torch.stack([s / (s + self.k2) for s in EXTRA_SIGMA])
        self.extra_radial[(slice(None),) + self.origin] = 0

    def new_basis(self, x):
        if self.is_local:
            return torch.stack([r3(x, s)[0] for s in self.local_scales], -1)
        ft = self.fft(x - x.mean(self.ax, keepdim=True))
        return self.ifft(ft[:, None] * self.extra_radial).real.movedim(1, -1)

    def velocity(self, x, t):
        c = self.at(t)
        return ((self.basis(x) * c[:8]).sum(-1) + (self.extra_basis(x) * c[8:12]).sum(-1)
                + (self.new_basis(x) * c[12:]).sum(-1))

    def divergence(self, x, t):
        c = self.at(t)
        base = TimeCNF.divergence(self, x, t) + (c[8:12] * self.radial.sum(self.ax)).sum()
        if self.is_local:
            return base + sum(c[12 + j] * r3(x, s)[1].sum(self.ax) for j, s in enumerate(self.local_scales))
        return base + (c[12:] * self.extra_radial.sum(self.ax)).sum()

    def local_bound(self, c):
        a = [9 / 8 * self.sphi ** 2] + [9 / 8 * s ** 2 for s in self.local_scales]
        return float(abs(c[2]) * a[0] + abs(c[12]) * a[1] + abs(c[13]) * a[2])

    def bound(self):
        cn = self.coeff.cpu().numpy()
        uniform = float(np.abs(cn[:, :8] @ self.derivative_constraints[:-1].T).max())
        if self.is_local:
            local = max(self.local_bound(c) for c in cn)
        else:
            local = float(np.abs(cn[:, :8] @ self.derivative_constraints[-1:].T).max())
        return max(uniform, local) + 1e-9

    def rates(self, c, h):
        rate = c[0] - c[1] * self.k2 + (c[6] + c[7] * h) * self.mask
        rate = rate + (self.bcast(c[8:12]) * self.radial).sum(0)
        if self.kind == 'fourier':
            rate = rate + (self.bcast(c[12:]) * self.extra_radial).sum(0)
        nonzero = torch.ones_like(self.mask)
        nonzero[self.origin] = 0
        return rate * nonzero

    def scalar(self, x, tau, c, local):
        if not (local and self.is_local):
            return OperatorCNF.scalar(self, x, tau, c, local)
        scales = (self.sphi,) + self.local_scales
        coefs = (c[2], c[12], c[13])

        def fun(u):
            v = d = 0
            for a, s in zip(coefs, scales):
                rv, rd = r3(u, s)
                v, d = v + a * rv, d + a * rd
            return v, d
        return midpoint(fun, x, tau)

    def substep(self, x, tau, c, part):
        if part == 'C':
            m = x.mean(self.ax, keepdim=True)
            rate = self.rates(c, m.square() / (self.sm ** 2 + m.square()))
            y = m + self.ifft(self.fft(x - m) * torch.exp(tau * rate)).real
            return y, tau * rate.sum(self.ax)
        return OperatorCNF.substep(self, x, tau, c, part)

    def new_rate_matrix(self):
        cols = [r.reshape(-1).cpu().numpy() for r in self.extra_radial]
        block = np.stack(cols, -1)
        return np.concatenate([block, block])


def load_model(spec, d=2, device='cpu'):
    """Rebuild a model from a fit.json dictionary (coefficient shape selects the class)."""
    c = np.asarray(spec['coeff'], dtype=np.float64)
    if c.shape == (17, 14):
        return ExtCNF(spec['L'], c, spec['sphi'], spec['sm'], spec['kind'], d=d, device=device)
    cls = {(9, 8): OperatorCNF, (17, 8): TimeCNF, (17, 12): RadialCNF}[c.shape]
    return cls(spec['L'], c, spec['sphi'], spec['sm'], d=d, device=device)


# ----------------------------------------------------------------------------- physics (PKG physics/diagnostics)
def phi4_action(cfgs, lam=LAM, kappa=.27):
    """PKG physics.phi4_action: S = sum_x [(1-2 lam) phi^2 + lam phi^4 - 2 kappa sum_mu phi_x phi_{x+mu}]."""
    density = (1 - 2 * lam) * cfgs ** 2 + lam * cfgs ** 4
    for mu in range(1, len(cfgs.shape)):
        density -= 2 * kappa * cfgs * torch.roll(cfgs, shifts=1, dims=mu)
    return density.sum(dim=tuple(range(1, len(cfgs.shape))))


def phi4_force(cfgs, lam=LAM, kappa=.27):
    """PKG physics.phi4_force (d-generic)."""
    grad_S = 2 * (1 - 2 * lam) * cfgs + 4 * lam * cfgs ** 3
    neighbor_sum = torch.zeros_like(cfgs)
    for mu in range(1, len(cfgs.shape)):
        neighbor_sum += torch.roll(cfgs, shifts=1, dims=mu)
        neighbor_sum += torch.roll(cfgs, shifts=-1, dims=mu)
    grad_S -= 2 * kappa * neighbor_sum
    return -grad_S


def action(phi, lam, kappa):
    return phi4_action(phi, lam=lam, kappa=kappa)


def classical_minimum(kappa, lam, d):
    """|phi| at the minimum of the constant-field action: phi^2 = (2 d kappa - (1 - 2 lam)) / (2 lam)."""
    if lam == 0:
        return 0.
    return math.sqrt(max(0., (2 * d * kappa - (1 - 2 * lam)) / (2 * lam)))


KEYS = ("M", "absM", "M2", "M4", "phi2", "phi4", "energy", "action_density", "Smin", "Gsub2")


@torch.no_grad()
def observables(phi, lam, kappa, d=2):
    """PKG diagnostics.observables on an L^d lattice (Smin: q^2 = 1 shell power; Gsub2: correlator at |r| = 2)."""
    ax = axes(d)
    L = phi.shape[-1]
    if L < 4 or any(n != L for n in phi.shape[-d:]) or phi.dim() != d + 1 or not bool(torch.isfinite(phi).all()):
        raise ValueError("observables require finite hypercubic fields with L >= 4")
    volume = L ** d
    m = phi.mean(ax)
    power = torch.fft.fftn(phi, dim=ax, norm='ortho').abs().square()
    qmap, _, _ = lattice(L, d)
    smin = volume * power[:, qmap == 1].mean(1)
    centered_power = power.clone()
    centered_power[(slice(None),) + (0,) * d] = 0
    corr = torch.fft.ifftn(centered_power, dim=ax, norm='ortho').real / math.sqrt(volume)
    xs = torch.minimum(torch.arange(L), L - torch.arange(L)).double()
    dist2 = None
    for mu in range(d):
        shape = [1] * d
        shape[mu] = L
        term = xs.square().reshape(shape)
        dist2 = term if dist2 is None else dist2 + term
    distance = dist2.sqrt()
    gsub2 = corr[:, (distance - 2).abs() < 0.05].mean(1)
    energy = phi * phi.roll(1, -1)
    for mu in range(2, d + 1):
        energy = energy + phi * phi.roll(1, -mu)
    energy = energy.mean(ax)
    values = (m, m.abs(), m.square(), m.pow(4), phi.square().mean(ax),
              phi.pow(4).mean(ax), energy, action(phi, lam, kappa) / volume, smin, gsub2)
    out = {key: value.cpu().numpy().copy() for key, value in zip(KEYS, values)}
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("nonfinite observable; configurations are never filtered")
    return out


def estimates(means, volume):
    out = dict(zip(KEYS, map(float, means)))
    m2, m4 = out["M2"], out["M4"]
    out["U4"] = 1 - m4 / (3 * m2 * m2) if m2 > 0 else float("nan")
    out["chi"] = volume * (m2 - out["absM"] ** 2)
    out["chi_connected"] = volume * (m2 - out["M"] ** 2)
    return out


def influence_series(features, volume):
    out = dict(features)
    m2, m4 = float(features["M2"].mean()), float(features["M4"].mean())
    if m2 > 0:
        out["U4"] = (2 * m4 / (3 * m2 ** 3) * (features["M2"] - m2)
                     - (features["M4"] - m4) / (3 * m2 ** 2))
    else:
        out["U4"] = np.zeros_like(features["M"])
    out["chi"] = volume * (features["M2"] - m2
                           - 2 * features["absM"].mean() * (features["absM"] - features["absM"].mean()))
    out["chi_connected"] = volume * (features["M2"] - m2
                                     - 2 * features["M"].mean() * (features["M"] - features["M"].mean()))
    return out


def iid_stats(features, volume):
    point = estimates([features[k].mean() for k in KEYS], volume)
    influence = influence_series(features, volume)
    n = len(features['M'])
    return {k: {'mean': v, 'se': float(influence[k].std(ddof=1) / math.sqrt(n))} for k, v in point.items()}


# ----------------------------------------------------------------------------- data paths (e18_csd, cnf_pipeline)
def with_data_target(data):
    """Replace every node's target by the conditional flow-matching target u = phi_1 - z."""
    u = data['paths'][-1] - data['paths'][0]
    out = dict(data)
    out['teacher_targets'] = data['targets']
    out['targets'] = u.unsqueeze(0).expand_as(data['targets']).contiguous()
    return out


def data_paths(fields, seed):
    """e18_csd.data_paths: split, z and straight paths from the pool seed; targets u = phi_1 - z."""
    rng = np.random.default_rng(seed + 10)
    half = len(fields) // 2
    parts = [rng.permutation(np.arange(i * half, (i + 1) * half)) for i in range(2)]
    nt = 3 * half // 4
    train = np.concatenate([p[:nt] for p in parts])
    valid = np.concatenate([p[nt:] for p in parts])
    z = torch.randn(fields.shape, dtype=DT, generator=torch.Generator().manual_seed(seed + 11))
    times = np.linspace(0, 1, 17)
    paths = torch.stack([(1 - t) * z + t * fields for t in times])
    data = {'paths': paths, 'targets': torch.zeros_like(paths), 'times': times, 'train': train, 'valid': valid}
    out = with_data_target(data)
    out.pop('teacher_targets')
    return out


def dim_of(data):
    return data['paths'].dim() - 2


# ----------------------------------------------------------------------------- fitting stages (cnf_pipeline)
@torch.no_grad()
def initial_fit(data, L, out=None):
    d = dim_of(data)
    ax = axes(d)
    weights = np.ones(17)
    weights[[0, -1]] = .5
    weights /= weights.sum()
    train, valid = data['train'], data['valid']
    sp = math.sqrt(sum(w * float(x[train].square().mean()) for w, x in zip(weights, data['paths'])))
    sm = math.sqrt(sum(w * float(x[train].mean(ax).square().mean()) for w, x in zip(weights, data['paths'])))
    template = OperatorCNF(L, np.zeros((9, 8)), sp, sm, d=d)
    reduced, projected, validation = [], [], []
    for t, w, x, y in zip(data['times'], weights, data['paths'], data['targets']):
        b = template.basis(x).numpy()
        A, yy = b[train].reshape(-1, 8), y.numpy()[train].ravel()
        q, r = scipy.linalg.qr(A, mode='economic', check_finite=True)
        scale = math.sqrt(w / len(yy))
        reduced.append(np.hstack([r * h for h in timeweights(t)]) * scale)
        projected.append((q.T @ yy) * scale)
        validation.append((b[valid], y.numpy()[valid], w, timeweights(t)))
    A, y = np.vstack(reduced), np.concatenate(projected)
    col = np.sqrt(np.square(A).sum(0))
    if np.any(col <= 1e-14):
        raise RuntimeError('zero initial design column')
    B, candidates = A / col, []
    for ridge in (0., 1e-8, 1e-6, 1e-4):
        aug = np.vstack([B, math.sqrt(ridge) * np.eye(72)])
        c, _, rank, singular = scipy.linalg.lstsq(aug, np.r_[y, np.zeros(72)], lapack_driver='gelsd')
        coeff = (c / col).reshape(9, 8)
        mse = sum(w * np.square(b @ (h @ coeff) - yy).mean() for b, yy, w, h in validation)
        candidates.append({'ridge': ridge, 'validation_mse': mse, 'rank': int(rank), 'coeff': coeff})
    index = min(range(len(candidates)), key=lambda i: candidates[i]['validation_mse'])
    spec = {'L': L, 'd': d, 'sphi': sp, 'sm': sm, 'coeff': candidates[index]['coeff'],
            'selected_index': index, 'candidates': candidates}
    write(sub(out, 'initial_fit.json'), spec)
    return OperatorCNF(L, spec['coeff'], sp, sm, d=d)


@torch.no_grad()
def constrained_fit(data, old, out, extra, L, kappa, lam=LAM):
    d = dim_of(data)
    if extra:
        initial = torch.cat([old.coeff, torch.zeros(17, 4, dtype=DT)], -1)
        template = RadialCNF(L, initial, old.sphi, old.sm, d=d)
        constraint, slots = template.rate_matrix(), LINEAR
    else:
        constraint, slots = derivative_matrix(old.sm, old.sphi), list(range(8))
    rows, fitted = [], []
    for i, (t, x, target) in enumerate(zip(data['times'], data['paths'], data['targets'])):
        basis = old.basis(x)
        if extra:
            b = torch.cat([basis[..., [0, 1, 6, 7]], template.extra_basis(x)], -1).numpy()
            fixed = (basis[..., [2, 3, 4, 5]] * old.at(t)[[2, 3, 4, 5]]).sum(-1).numpy()
            c = initial[i].numpy().copy()
        else:
            b, fixed = basis.numpy(), np.zeros_like(target.numpy())
            c = np.array([-1., 0., 0., -1., 0., 0., 0., 0.]) if i == 0 else old.at(t).numpy().copy()
        A = b[data['train']].reshape(-1, 8)
        y = (target.numpy() - fixed)[data['train']].reshape(-1)
        q, r = scipy.linalg.qr(A, mode='economic')
        if extra:
            r, y = r / math.sqrt(len(y)), (q.T @ y) / math.sqrt(len(y))
        else:
            scale = 1 / math.sqrt(len(y))
            r, y = r * scale, (q.T @ y) * scale
        col = np.linalg.norm(r, axis=0)
        if np.min(col) <= 1e-14:
            raise RuntimeError('zero constrained design column')
        B = r / col
        Q, v = B.T @ B + 1e-6 * np.eye(8), B.T @ y
        if i == 0:
            solver = {'success': True, 'fixed_initial_condition': True}
        else:
            if extra:
                u = c[slots] * col
                u *= min(1., 31.8 / max(1., np.abs(constraint @ c[slots]).max()))
            else:
                bound = abs(constraint @ c)
                c[3:6] *= min(1., 31.8 / max(1., bound[:7].max()))
                c[2] *= min(1., 31.8 / max(1., bound[7]))
                u = c * col
            result = minimize(lambda z: .5 * z @ Q @ z - v @ z, u,
                              jac=lambda z: Q @ z - v, method='SLSQP',
                              constraints=LinearConstraint(constraint / col, -31.9, 31.9),
                              options={'maxiter': 300, 'ftol': 1e-12})
            c[slots] = result.x / col
            solver = {'success': bool(result.success), 'message': str(result.message),
                      'iterations': int(result.nit), 'objective': float(result.fun)}
        bound = float(abs(constraint @ c[slots]).max())
        pred = b[data['valid']] @ c[slots] + fixed[data['valid']]
        row = {'time': float(t), 'coeff': c, 'solver': solver, 'bound': bound,
               'validation': errors(pred, target[data['valid']].numpy(), d)}
        write(sub(out, f'node_{i:02d}.json'), row)
        if not solver['success'] or bound > 31.90001:
            raise RuntimeError(f'node {i} failed; no retry')
        fitted.append(c)
        rows.append(row)
    cls = RadialCNF if extra else TimeCNF
    model = cls(L, fitted, old.sphi, old.sm, d=d)
    write(sub(out, 'fit.json'), {'L': L, 'd': d, 'kappa': kappa, 'lambda': lam, 'coeff': fitted,
          'sphi': old.sphi, 'sm': old.sm, 'n_fitted_coefficients': 192 if extra else 128,
          'n_train': len(data['train']), 'n_validation': len(data['valid']), 'rows': rows})
    return model


def mkdir(out, name):
    if out is None:
        return None
    p = Path(out) / name
    p.mkdir()
    return p


def fit_stages(data, out, L, kappa, check=True, lam=LAM):
    """initial (72) -> time-constrained (128) -> radial (192); optional step/inverse/autograd checks."""
    old = initial_fit(data, L, out)
    old = constrained_fit(data, old, mkdir(out, 'time_fit'), False, L, kappa, lam)
    final = mkdir(out, 'radial_fit')
    model = constrained_fit(data, old, final, True, L, kappa, lam)
    steps = check_fitted(model, final) if check else None
    write(sub(final, 'complete.json'), {'complete': True, 'steps': steps})
    return model, steps


@torch.no_grad()
def local_stage(parent, data, out, kind='local'):
    """Third stage (E1.1): two new local coefficients per free node on (target - parent velocity) -> 224."""
    L, d = parent.L, parent.d
    pc = parent.coeff.numpy()
    zero = np.concatenate([pc, np.zeros((17, 2))], 1)
    template = ExtCNF(L, zero, parent.sphi, parent.sm, kind, d=d)
    coeff, rows = zero.copy(), []
    for i, (t, x, target) in enumerate(zip(data['times'], data['paths'], data['targets'])):
        base = parent.velocity(x, float(t)).numpy()
        b = template.new_basis(x).numpy()
        A = b[data['train']].reshape(-1, 2)
        y = (target.numpy() - base)[data['train']].reshape(-1)
        c = np.zeros(2)
        if i > 0:
            q, r = scipy.linalg.qr(A, mode='economic')
            r, yy = r / math.sqrt(len(y)), (q.T @ y) / math.sqrt(len(y))
            col = np.linalg.norm(r, axis=0)
            if np.min(col) <= 1e-14:
                raise RuntimeError('zero design column')
            B = r / col
            Q, v = B.T @ B + 1e-6 * np.eye(2), B.T @ yy
            a = np.array([9 / 8 * s ** 2 for s in template.local_scales])
            M = np.array([[s1 * a[0], s2 * a[1]] for s1 in (1, -1) for s2 in (1, -1)])
            room = BOUND - abs(pc[i, 2]) * 9 / 8 * template.sphi ** 2
            res = minimize(lambda z: .5 * z @ Q @ z - v @ z, np.zeros(2), jac=lambda z: Q @ z - v, method='SLSQP',
                           constraints=LinearConstraint(M / col, -np.inf, room), options={'maxiter': 300, 'ftol': 1e-12})
            if not res.success:
                raise RuntimeError(f'{kind} node {i}: SLSQP failed; no retry')
            c = res.x / col
        coeff[i, 12:] = c
        tv = data.get('teacher_targets', data['targets'])[i].numpy()[data['valid']]
        rows.append({'time': float(t), 'new_coeff': c.tolist(),
                     'validation_vs_teacher_parent': errors(base[data['valid']], tv, d),
                     'validation_vs_teacher_new': errors(base[data['valid']] + b[data['valid']] @ c, tv, d)})
    model = ExtCNF(L, coeff, parent.sphi, parent.sm, kind, d=d)
    if max(model.local_bound(cc) for cc in coeff) > BOUND + 1e-6:
        raise RuntimeError('local bound violated')
    write(sub(out, 'fit.json'), {'kind': kind, 'L': L, 'd': d, 'coeff': coeff, 'sphi': parent.sphi, 'sm': parent.sm,
                                 'rows': rows, 'bound': model.bound()})
    return model


# ----------------------------------------------------------------------------- post-fit checks
def _autograd_checks(tiny, xx, steps, checks):
    d = tiny.d
    shape = (1,) + (4,) * d
    with torch.enable_grad():
        j = torch.autograd.functional.jacobian(lambda w: tiny.transform(w.reshape(shape), steps)[0].reshape(-1),
                                               xx.reshape(-1))
        sign, det = torch.linalg.slogdet(j)
        _, expected = tiny.transform(xx, steps)
        checks['fitted_L4_logdet_autograd'] = float(abs(det - expected[0]))
        checks['L4_jacobian_sign_positive'] = bool(sign > 0)
        jv = torch.autograd.functional.jacobian(lambda w: tiny.velocity(w.reshape(shape), .37).reshape(-1),
                                                xx.reshape(-1))
        checks['divergence_vs_trace'] = float(abs(jv.trace() - tiny.divergence(xx, .37)[0]))


def volume_factor(model):
    """Log-det and log-q are sums over sites: their absolute guards scale with V / 64 (1 for V <= 64, i.e. the
    original 2D L = 8 thresholds are unchanged)."""
    return max(1., model.L ** model.d / 64)


@torch.no_grad()
def check_fitted(model, out=None):
    """fit_radial_response.check_fitted: step doubling, inverse, L=4 autograd log-det."""
    shape = (128,) + (model.L,) * model.d
    vf = volume_factor(model)
    x = torch.randn(shape, dtype=DT, generator=torch.Generator().manual_seed(31021003))
    records, accepted = [], None
    for n in (32, 64, 128, 256):
        y, ld = model.transform(x, n)
        yy, lld = model.transform(x, 2 * n)
        err = float(((y - yy).square().mean() / yy.square().mean()).sqrt())
        ld_err = float((ld - lld).abs().max())
        records.append({'steps': n, 'refined_steps': 2 * n, 'relative_rms_phi': err, 'max_logdet_difference': ld_err})
        if err < 1e-3 and ld_err < .005 * vf:
            accepted = 2 * n
            break
    write(sub(out, 'step_convergence.json'), {'trials': records, 'steps': accepted})
    if accepted is None:
        raise RuntimeError('step refinement failed within 512 steps; no retry')
    probes = torch.cat([x[:8], 10 * x[:8], x[:8] + 3])
    y, ld = model.transform(probes, accepted)
    z, li = model.transform(y, accepted, True)
    checks = {'inverse_relative': float(((z - probes).abs() / probes.abs().clamp_min(1)).max()),
              'logq_inverse': float((model.logp(probes) - ld - model.logp(z) - li).abs().max())}
    tiny = RadialCNF(4, model.coeff, model.sphi, model.sm, d=model.d)
    _autograd_checks(tiny, corner(x, model.d).clone(), accepted, checks)
    checks.pop('divergence_vs_trace')
    write(sub(out, 'fitted_checks.json'), checks)
    assert checks['L4_jacobian_sign_positive']
    assert max(v for k, v in checks.items() if k not in ('L4_jacobian_sign_positive', 'logq_inverse')) < 1e-7, checks
    assert checks['logq_inverse'] < 1e-7 * vf, checks
    return accepted


def check_extended(model, parent, parent_steps, out=None):
    """cnf_pipeline.check_extended: zero extension, refinement, inverse, L=4 log-det and divergence."""
    L, d = model.L, model.d
    vf = volume_factor(model)
    nx = 128 if L ** d <= 256 else 64
    x = torch.randn((nx,) + (L,) * d, dtype=DT, generator=torch.Generator().manual_seed(31021003))
    with torch.no_grad():
        zero = ExtCNF(L, np.concatenate([parent.coeff.numpy(), np.zeros((17, 2))], 1), parent.sphi, parent.sm,
                      model.kind, d=d)
        y0, l0 = zero.transform(x[:nx // 8], parent_steps)
        y1, l1 = parent.transform(x[:nx // 8], parent_steps)
        checks = {'zero_extension_field': float((y0 - y1).abs().max()),
                  'zero_extension_logdet': float((l0 - l1).abs().max())}
        records, accepted = [], None
        for n in (64, 128, 256):
            y, ld = model.transform(x, n)
            yy, lld = model.transform(x, 2 * n)
            err = float(((y - yy).square().mean() / yy.square().mean()).sqrt())
            ld_err = float((ld - lld).abs().max())
            records.append({'steps': n, 'refined_steps': 2 * n, 'relative_rms_phi': err,
                            'max_logdet_difference': ld_err})
            if err < 1e-3 and ld_err < .005 * vf:
                accepted = 2 * n
                break
        if accepted is None:
            raise RuntimeError('step refinement failed within 512 steps; no retry')
        accepted = max(accepted, parent_steps)
        k = nx // 16
        probes = torch.cat([x[:k], 10 * x[:k], x[:k] + 3])
        y, ld = model.transform(probes, accepted)
        z, li = model.transform(y, accepted, True)
        checks['inverse_relative'] = float(((z - probes).abs() / probes.abs().clamp_min(1)).max())
        checks['logq_inverse'] = float((model.logp(probes) - ld - model.logp(z) - li).abs().max())
    tiny = ExtCNF(4, model.coeff, model.sphi, model.sm, model.kind, d=d)
    _autograd_checks(tiny, corner(x, d).clone(), accepted, checks)
    write(sub(out, 'step_convergence.json'), {'trials': records, 'steps': accepted})
    write(sub(out, 'fitted_checks.json'), checks)
    numeric = [checks[k] for k in ('zero_extension_field', 'zero_extension_logdet', 'inverse_relative',
                                   'fitted_L4_logdet_autograd', 'divergence_vs_trace')]
    if max(numeric) >= 1e-7 or checks['logq_inverse'] >= 1e-7 * vf or not checks['L4_jacobian_sign_positive']:
        raise RuntimeError(f'fitted checks failed {checks}')
    return accepted


# ----------------------------------------------------------------------------- sampling (cnf_pipeline)
FEATURES = ('M', 'phi2', 'phi4', 'energy', 'Q2', 'max_phi', 'z2', 'logw', 'logq', 'logdet', 'S')


@torch.no_grad()
def generate(model, steps, n, seed, kappa, out=None, name='proposals', lam=LAM, with_obs=True, batch=512):
    """Independent proposals from torch.Generator(seed) (CPU stream), batches of 512 (same stream as
    cnf_pipeline.generate for d = 2).  Returns (features, observables)."""
    L, d, ax = model.L, model.d, model.ax
    gen = torch.Generator().manual_seed(seed)
    keep = {k: [] for k in FEATURES}
    obs_rows = []
    for off in range(0, n, batch):
        z = torch.randn((min(batch, n - off),) + (L,) * d, dtype=DT, generator=gen).to(model.device)
        phi, ld = model.transform(z, steps)
        logq = model.logp(z) - ld
        S = action(phi, lam, kappa)
        lw = -S - logq
        if not bool(torch.isfinite(lw).all()):
            raise RuntimeError('nonfinite weights; no filtering')
        m = phi.mean(ax)
        energy = phi * phi.roll(1, -1)
        for mu in range(2, d + 1):
            energy = energy + phi * phi.roll(1, -mu)
        vals = {'M': m, 'phi2': phi.square().mean(ax), 'phi4': phi.pow(4).mean(ax),
                'energy': energy.mean(ax),
                'Q2': (phi - m.reshape((-1,) + (1,) * d)).square().mean(ax), 'max_phi': phi.abs().amax(ax),
                'z2': z.square().sum(ax), 'logw': lw, 'logq': logq, 'logdet': ld, 'S': S}
        for k, v in vals.items():
            keep[k].append(v.cpu().numpy())
        if with_obs:
            obs_rows.append(observables(phi.cpu(), lam, kappa, d))
    f = {k: np.concatenate(v) for k, v in keep.items()}
    obs = {k: np.concatenate([r[k] for r in obs_rows]) for k in KEYS} if with_obs else None
    if out is not None:
        with (Path(out) / f'{name}.npz').open('xb') as fh:
            np.savez_compressed(fh, **f, **({f'obs_{k}': v for k, v in obs.items()} if with_obs else {}))
    return f, obs


def ess(lw):
    """Normalized importance-sampling ESS (sum w)^2 / (N sum w^2)."""
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (len(w) * np.square(w).sum()))


def ratios(f, kappa, V, lam=LAM):
    """-log q regressed on (phi^2, phi^4, nn), divided by the exact coefficients of S (=1 for q = p)."""
    y = -f['logq']
    X = np.column_stack([f['phi2'], f['phi4'], f['energy']])
    X = X - X.mean(0)
    beta = np.linalg.lstsq(X, y - y.mean(), rcond=None)[0]
    return (beta / np.array([(1 - 2 * lam) * V, lam * V, -2 * kappa * V])).tolist()


def paired(lw, ref, seed=31029014, nboot=400):
    rng = np.random.default_rng(seed)
    n = len(lw)
    d = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        d.append(ess(lw[idx]) - ess(ref[idx]))
    return {'diff': ess(lw) - ess(ref), 'CI95': np.quantile(d, [.025, .975]).tolist(), 'se': float(np.std(d, ddof=1))}


def imh(logw, steps, warmup, seed):
    """Four independent Metropolis-Hastings chains on interleaved proposals (cnf_pipeline.imh)."""
    uniforms = np.random.default_rng(seed).random((steps + warmup, 4))
    indices = np.empty((steps + warmup, 4), dtype=np.int64)
    accepted = np.empty_like(indices, dtype=bool)
    current = np.arange(4)
    for i in range(steps + warmup):
        candidate = 4 * (i + 1) + np.arange(4)
        take = np.log(uniforms[i]) < logw[candidate] - logw[current]
        current = np.where(take, candidate, current)
        indices[i], accepted[i] = current, take
    return indices, accepted


# ----------------------------------------------------------------------------- pipeline (e18_csd.flow)
def fit_d224(fields, seed, kappa, out=None, lam=LAM, check=True):
    """D192 + D224 fit on one pool of configurations.  Returns (m192, s192, model, accepted)."""
    L = fields.shape[-1]
    data = data_paths(fields, seed)
    m192, s192 = fit_stages(data, mkdir(out, 'D192'), L, kappa, check=check, lam=lam)
    model = local_stage(m192, data, mkdir(out, 'D224'), 'local')
    accepted = check_extended(model, m192, s192, sub(out, 'D224')) if check else None
    return m192, s192, model, accepted


def flow(fields, seed, kappa, out, lam=LAM, n_is=65536, n_chain=65536, warm=4096):
    """e18_csd.flow in d dimensions: fit, step guard, IS ESS, 4 IMH chains (observable traces saved)."""
    d = fields.dim() - 1
    L = fields.shape[-1]
    V = L ** d
    start = time.perf_counter()
    m192, s192, model, accepted = fit_d224(fields, seed, kappa, out, lam)
    fit_seconds = time.perf_counter() - start
    steps = max(32, 16 * math.ceil(model.bound() / 16))
    g32, _ = generate(model, steps, 8192, seed + 55, kappa, lam=lam, with_obs=False)
    gacc, _ = generate(model, accepted, 8192, seed + 55, kappa, lam=lam, with_obs=False)
    step_guard = {'steps': steps, 'accepted': accepted, 'ESS_steps': ess(g32['logw']), 'ESS_accepted': ess(gacc['logw'])}
    t = time.perf_counter()
    f_is, _ = generate(model, steps, n_is, seed + 50, kappa, out, 'proposals', lam=lam)
    sec = (time.perf_counter() - t) / n_is
    n = 4 * (n_chain + warm + 1)
    f_ch, obs = generate(model, steps, n, seed + 60, kappa, lam=lam)
    idx, acc = imh(f_ch['logw'], n_chain, warm, seed + 61)
    idx, acc = idx[warm:], acc[warm:]
    m = f_ch['M'][idx]
    with (Path(out) / 'chains.npz').open('xb') as fh:
        np.savez_compressed(fh, M=m, absM=np.abs(m), M2=m ** 2, M4=m ** 4, accepted=acc,
                            energy=f_ch['energy'][idx], phi2=f_ch['phi2'][idx])
    rec = {'d': d, 'L': L, 'kappa': kappa, 'lambda': lam, 'steps': steps, 'step_guard': step_guard,
           'note_cost': 'seconds_per_proposal includes observables on every proposal',
           'accepted_steps_continuous': accepted, 'bound': model.bound(),
           'fit_seconds': fit_seconds, 'seconds_per_proposal': sec, 'ESS': ess(f_is['logw']),
           'ESS_D192_not_computed': True,
           'Var_logw_per_site': float(f_is['logw'].var() / V), 'acceptance': float(acc.mean()),
           'ratios': ratios(f_is, kappa, V, lam), 'positive_fraction_proposals': float((f_is['M'] > 0).mean()),
           'sign_changes_per_chain': (np.diff(np.sign(m), axis=0) != 0).sum(0).tolist(),
           'seeds': {'pool': seed, 'is': seed + 50, 'chain': seed + 60}}
    write(Path(out) / 'record.json', rec)
    return rec


def time_flow(model, steps, kappa, lam=LAM, batch=512, n_batch=4, repeats=3):
    """e25_retime pattern: one warm-up batch, then `repeats` x `n_batch` timed batches; seconds per proposal."""
    import statistics
    gen = torch.Generator().manual_seed(1)
    shape = (batch,) + (model.L,) * model.d

    @torch.no_grad()
    def one():
        z = torch.randn(shape, dtype=DT, generator=gen).to(model.device)
        phi, ld = model.transform(z, steps)
        return -action(phi, lam, kappa) - (model.logp(z) - ld)
    one()
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(n_batch):
            one()
        out.append((time.perf_counter() - t) / n_batch / batch)
    return {'median': statistics.median(out), 'repeats': out, 'n_batch': n_batch, 'steps': steps}


def main():
    import argparse
    import hashlib
    p = argparse.ArgumentParser(description='d-dim D224 flow fitted on one pool of configurations')
    p.add_argument('--pool', type=Path, required=True, help='folder with fit_pool.pt and fit_pool_setup.json')
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--lam', type=float, default=LAM)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-is', type=int, default=65536)
    p.add_argument('--n-chain', type=int, default=65536)
    p.add_argument('--warm', type=int, default=4096)
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('FEWTERM_THREADS', '1')))
    seed = json.loads((a.pool / 'fit_pool_setup.json').read_text(encoding='utf-8'))['seed']
    fields = torch.load(a.pool / 'fit_pool.pt', map_location='cpu', weights_only=True).double()
    a.out.mkdir(parents=True)
    files = [Path(__file__), a.pool / 'fit_pool.pt']
    write(a.out / 'sources.json', {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    rec = flow(fields, seed, a.kappa, a.out, a.lam, a.n_is, a.n_chain, a.warm)
    print(json.dumps(rec, default=encode), flush=True)
    (a.out / 'complete.json').write_text(json.dumps({'complete': True}), encoding='utf-8')


if __name__ == '__main__':
    main()
