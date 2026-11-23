"""Landau-type base distribution for the zero mode.

The base field is z = z_perp + (y / sqrt V) 1 with z_perp a standard Gaussian with its zero mode removed and
y = sum_x z_x / sqrt V drawn from the 1D density p(y) prop. to exp(-a (y/s)^2 - (y/s)^4).  The shape a and the
scale s are fixed by the two pool moments <y^2>, <y^4> of y = sqrt V mean(phi) (kurtosis ratio -> a by
bisection, then <y^2> -> s); ratio >= 3 (Gaussian or heavier) falls back to a Gaussian of variance <y^2>.

Sampling is inverse CDF on a uniform grid with linear interpolation, i.e. exactly from the piecewise-constant
density of the grid cells; logp evaluates that same piecewise-constant density, so q is exact for IS / IMH.
"""
import math

import numpy as np
import torch

DT = torch.float64


def _moments(a, lim, n=200001):
    u = np.linspace(-lim, lim, n)
    lp = -a * u * u - u ** 4
    p = np.exp(lp - lp.max())
    return (p * u ** 2).sum() / p.sum(), (p * u ** 4).sum() / p.sum()


def _ratio(a, lim):
    m2, m4 = _moments(a, lim)
    return m4 / m2 ** 2


def _lim(a):
    return math.sqrt(max(-a, 0.) / 2) + 5.


class LandauBase:
    def __init__(self, V, y2, y4, grid=400001):
        self.V = V
        r = y4 / y2 ** 2
        self.ratio = float(r)
        if r >= 2.99:                   # symmetric phase: Gaussian zero mode of variance <y^2>
            self.a, self.gauss = None, True
            self.s = math.sqrt(y2)
            lim = 9.
            u = np.linspace(-lim, lim, grid)
            lp = -.5 * u * u
        else:
            self.gauss = False
            lo, hi = -400., 400.            # ratio(a) decreases monotonically from 3 (a -> +inf) to 1 (a -> -inf)
            for _ in range(80):
                mid = .5 * (lo + hi)
                if _ratio(mid, _lim(mid)) > r:
                    hi = mid
                else:
                    lo = mid
            self.a = .5 * (lo + hi)
            lim = _lim(self.a)
            self.s = math.sqrt(y2 / _moments(self.a, lim)[0])
            u = np.linspace(-lim, lim, grid)
            lp = -self.a * u * u - u ** 4
        p = np.exp(lp - lp.max())
        cdf = np.concatenate([[0.], np.cumsum(.5 * (p[1:] + p[:-1]))])
        cdf /= cdf[-1]
        self.y = u * self.s                                        # grid in y
        self.cdf = cdf
        dy = self.y[1] - self.y[0]
        with np.errstate(divide='ignore'):
            self.cell_logp = np.log(np.diff(cdf) / dy)            # exact density of the sampler on each cell
        self.y0, self.dy = float(self.y[0]), float(dy)

    def info(self):
        return {'a': self.a, 's': self.s, 'gauss': self.gauss, 'pool_ratio': self.ratio}

    def sample_y(self, n, generator):
        r = torch.rand(n, dtype=DT, generator=generator).numpy()
        return torch.as_tensor(np.interp(r, self.cdf, self.y), dtype=DT)

    def logp_y(self, y):
        i = np.floor((y.numpy() - self.y0) / self.dy).astype(np.int64)
        ok = (i >= 0) & (i < len(self.cell_logp))
        out = np.full(y.shape, -np.inf)
        out[ok] = self.cell_logp[i[ok]]
        return torch.as_tensor(out, dtype=DT)

    def sample(self, shape, generator):
        d = len(shape) - 1
        ax = tuple(range(1, d + 1))
        z = torch.randn(shape, dtype=DT, generator=generator)
        rt = math.sqrt(self.V)
        y_old = z.sum(ax) / rt
        y_new = self.sample_y(shape[0], generator)
        return z + ((y_new - y_old) / rt).reshape((-1,) + (1,) * d)

    def logp(self, x):
        d = x.dim() - 1
        ax = tuple(range(1, d + 1))
        y = x.sum(ax) / math.sqrt(self.V)
        perp = x.square().sum(ax) - y * y
        return -.5 * (perp + (self.V - 1) * math.log(2 * math.pi)) + self.logp_y(y)


def from_pool(fields):
    d = fields.dim() - 1
    ax = tuple(range(1, d + 1))
    V = fields[0].numel()
    y = fields.sum(ax) / math.sqrt(V)
    return LandauBase(V, float((y ** 2).mean()), float((y ** 4).mean()))


def attach(model, base):
    model.base = base
    model.logp = base.logp
    return model


class EmpiricalBase(LandauBase):
    """Zero-mode marginal from the Z2-symmetrized pool quantiles (an alternative base, not used in the paper):
    nq equal-probability cells between the quantiles of {y, -y}; the two outer cells extend to +-2 max|y|.
    Piecewise-constant density on the cells, sampled by inverse CDF with linear interpolation (exact)."""

    def __init__(self, V, y, nq=64):
        self.V, self.gauss, self.a = V, False, None
        ys = np.sort(np.concatenate([y, -y]))
        knots = np.quantile(ys, np.linspace(0, 1, nq + 1))
        knots[0], knots[-1] = -2 * np.abs(ys).max(), 2 * np.abs(ys).max()
        self.knots = knots
        self.s = float(np.sqrt((ys ** 2).mean()))
        self.ratio = float((ys ** 4).mean() / (ys ** 2).mean() ** 2)
        self.cdf = np.linspace(0, 1, nq + 1)
        self.y = knots
        self.cell_logp = np.log((1 / nq) / np.diff(knots))

    def logp_y(self, y):
        i = np.searchsorted(self.knots, y.numpy(), side='right') - 1
        ok = (i >= 0) & (i < len(self.cell_logp))
        out = np.full(y.shape, -np.inf)
        out[ok] = self.cell_logp[i[ok]]
        return torch.as_tensor(out, dtype=DT)


def empirical_from_pool(fields, nq=64):
    d = fields.dim() - 1
    V = fields[0].numel()
    return EmpiricalBase(V, (fields.sum(tuple(range(1, d + 1))) / math.sqrt(V)).numpy(), nq)
