"""Shared helpers of the figure scripts: output folder, observable estimates with bootstrap errors, and the
combination of independent seeds.  Figures go to paper/figures or $FEWTERM_FIG_OUT."""
import os
import numpy as np

OUT = os.environ.get('FEWTERM_FIG_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures'))
os.makedirs(OUT, exist_ok=True)


def boot(fn, n, nb=200, seed=0):
    r = np.random.default_rng(seed)
    return np.std([fn(r.integers(0, n, n)) for _ in range(nb)])


QN = ('M', 'absM', 'chiV', 'U4')


def est(y, V, lw=None, nb=200):
    """<M>, <|M|>, chi/V=<M^2>, U4 (M = y/sqrt V), bootstrap errors; self-normalized IS if lw given."""
    w = np.ones_like(y) if lw is None else np.exp(lw - lw.max())
    m = y / np.sqrt(V)

    def f(idx):
        ww, mm = w[idx], m[idx]; Z = ww.sum()
        a1 = (ww * mm).sum() / Z; aa = (ww * np.abs(mm)).sum() / Z
        a2 = (ww * mm ** 2).sum() / Z; a4 = (ww * mm ** 4).sum() / Z
        return np.array([a1, aa, a2, 1 - a4 / (3 * a2 ** 2)])
    v = f(np.arange(len(y)))
    r = np.random.default_rng(1)
    bs = np.array([f(r.integers(0, len(y), len(y))) for _ in range(nb)])
    return v, bs.std(0)


def comb(vals, errs):
    """Independent seeds -> inverse-variance weighted mean; the error is scaled by the PDG factor sqrt(chi2/dof) when
    the seeds scatter more than their errors allow (chi2/dof > 1)."""
    v = np.array(vals, float); e = np.array(errs, float); n = len(v)
    if n == 1:
        return v[0], e[0]
    w = 1 / e ** 2
    mu = (w * v).sum(0) / w.sum(0)
    err = 1 / np.sqrt(w.sum(0))
    chi2 = (w * (v - mu) ** 2).sum(0) / (n - 1)
    return mu, err * np.sqrt(np.maximum(1., chi2))
