"""Free energy density F/V = -log Z / V: flow importance sampling vs HMC+Wolff thermodynamic integration.

Flow: log Z = log mean exp(log w) from the saved proposals (seed 1: obs/*/out/obs2d, seeds 2,3: seeds/95024),
bootstrap error per seed, between-seed scatter where three seeds exist.  Cost = training pool + fit + proposals.
TI: cubic-spline integral of the d log Z / d kappa nodes (seeds/95022); quadrature check = spline - Simpson on the
even nodes; cost = cumulative wall time up to kappa.

    python scripts/free_energy.py            -> free_energy.json, fig_free_energy.pdf
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

R = Path(__file__).resolve().parents[1]
LS = (8, 12, 16, 20, 24, 32)


def lme(x):
    m = x.max()
    return m + np.log(np.mean(np.exp(x - m)))


def ti(L):
    """Thermodynamic integration from single pure-HMC chains (eps = 0.1, 2x10^5 trajectories per node, ti1/*):
    cubic-spline integral of d log Z / d kappa over 121 nodes; error from the node errors with trapezoid weights;
    quadrature check = max |spline - Simpson| on the even nodes."""
    nodes = sorted((json.load(open(f)) for f in glob.glob(str(R / f'ti1/*/out/ti1/L{L}/k*.json'))), key=lambda r: r['kappa'])
    k = np.array([n['kappa'] for n in nodes]); g = np.array([n['dlogZ'] for n in nodes]); e = np.array([n['err'] for n in nodes])
    sec = np.cumsum([n['seconds'] for n in nodes])
    lam = .022
    u = np.linspace(-12, 12, 400001); fu = np.exp(-(1 - 2 * lam) * u * u - lam * u ** 4)
    lz0 = L * L * np.log(((fu[1:] + fu[:-1]) * .5 * (u[1] - u[0])).sum())
    cs = CubicSpline(k, g); V = L * L; dk = k[1] - k[0]

    def at(kap):
        F = -(lz0 + cs.integrate(0, kap)) / V
        i = np.searchsorted(k, kap)
        w = np.full(i + 1, dk); w[0] *= .5
        err = np.sqrt((w ** 2 * e[:i + 1] ** 2).sum()) / V
        return F, err, float(np.interp(kap, k, sec))
    chk = []
    for i in range(2, len(k), 2):
        simp = dk / 3 * (g[0] + g[i] + 4 * g[1:i:2].sum() + 2 * g[2:i - 1:2].sum())
        chk.append(abs(cs.integrate(0, k[i]) - simp) / V)
    return at, max(chk)


def flow(L, kap):
    tag = f'{kap:.3f}' if abs(kap * 1000 - round(kap * 1000)) < 1e-9 else f'{kap:.4f}'
    files = sorted(glob.glob(str(R / f'obs/*/out/obs2d/d2_L{L}_k{tag}.npz')))[:1]
    files += sorted(glob.glob(str(R / f'seeds/95024/out/obs2d_s/d2_L{L}_k{tag}_s*.npz')))
    out = []
    rng = np.random.default_rng(0)
    for f in files:
        lw = np.load(f)['landau_ot_logw'].astype(float)
        lw = lw[np.isfinite(lw)]
        j = json.load(open(f[:-4] + '.json'))
        lo = j['landau_ot']
        sec = j['pool_seconds'] + lo['fit_seconds'] + lo['seconds_per_proposal'] * len(lw)
        bs = [lme(lw[rng.integers(0, len(lw), len(lw))]) for _ in range(400)]
        out.append({'F': -lme(lw) / L ** 2, 'err': np.std(bs) / L ** 2, 'sec': sec, 'ess': lo['ESS'],
                    'sample_sec': lo['seconds_per_proposal'] * len(lw)})
    return out


def main():
    res = {}
    for L in LS:
        at, qchk = ti(L)
        for kap in (.26, .2705, .28, .30):
            fl = flow(L, kap)
            if not fl:
                continue
            Ft, et, st = at(kap)
            Fs = np.array([x['F'] for x in fl])
            mean = Fs.mean()
            eb = np.mean([x['err'] for x in fl]) / np.sqrt(len(Fs))
            err = max(Fs.std(ddof=1) / np.sqrt(len(Fs)), eb) if len(Fs) > 1 else fl[0]['err']
            res[f'L{L}_k{kap}'] = {'L': L, 'kappa': kap, 'flow_F': mean, 'flow_err_between': err,
                                   'flow_err_boot_seed': [x['err'] for x in fl], 'flow_seconds': np.mean([x['sec'] for x in fl]),
                                   'flow_sample_seconds': np.mean([x['sample_sec'] for x in fl]), 'nseeds': len(fl),
                                   'ti_F': Ft, 'ti_err': et, 'ti_seconds': st, 'ti_quad_check': qchk,
                                   'diff': mean - Ft, 'z': (mean - Ft) / np.hypot(err, et)}
            r = res[f'L{L}_k{kap}']
            print(f"L{L:2d} k{kap:.4f} flow {mean:.6f}+-{err:.1e} ({r['flow_seconds']:.0f}s)  TI {Ft:.6f}+-{et:.1e} "
                  f"({st:.0f}s)  z={r['z']:+.2f}  quad {qchk:.1e}")
    (R / 'free_energy.json').write_text(json.dumps(res, indent=1))
    figure(res)


def figure(res):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from style import plt, LCOL, W1
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ks = (.26, .2705, .28, .30)
    for i, L in enumerate(LS):
        r = [res[f'L{L}_k{k}'] for k in ks if f'L{L}_k{k}' in res]
        x = np.array([q['kappa'] for q in r]) + (i - 2.5) * .0011
        ax.errorbar(x, [1e4 * q['diff'] for q in r], [1e4 * np.hypot(q['flow_err_between'], q['ti_err']) for q in r],
                    fmt='o', ms=3, color=LCOL[L], label=f'{L}')
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(ks); ax.set_xticklabels(['0.26', '0.2705', '0.28', '0.30'])
    ax.set_xlabel(r'$\kappa$'); ax.set_ylabel(r'$f_{\rm IS}-f_{\rm TI}$')
    ax.legend(title='$L$', ncol=3, frameon=False, fontsize=7, title_fontsize=7, loc='upper left', columnspacing=.8,
              handletextpad=.2)
    from paper_figs import OUT
    fig.savefig(Path(OUT) / 'fig_free_energy.pdf')


if __name__ == '__main__':
    main()
