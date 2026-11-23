"""Fig. 2 (main): -ln ESS per site versus kappa for each L (few-parameter flow), seed scatter as error where >1 seed.
    python scripts/fig_essvol.py"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from style import plt, W1, LCOL  # noqa: E402
from paper_figs import OUT  # noqa: E402

E = {}
for f in (glob.glob('obs/*/out/obs2d/*.json') + glob.glob('seeds/95024/out/obs2d_s/*.json') + glob.glob('lscan/*/out/lscan/d2_*.json')
          + glob.glob('x2/*/out/*/d2_*.json') + glob.glob('fill/*/out/x2/d2_*.json')):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    if j.get('d') != 2 or 'landau_ot' not in j:
        continue
    E.setdefault((j['L'], round(j['kappa'], 4)), {})[f] = j['landau_ot']['ESS']
T = json.load(open('table_main.json'))
fig, ax = plt.subplots(figsize=(W1, 2.4))
out = {}
for L in (8, 12, 16, 20, 24, 32):
    ks = sorted(float(k.split('_k')[1]) for k in T if k.startswith(f'L{L}_'))
    y, e = [], []
    for k in ks:
        v = -np.log(T[f'L{L}_k{k}']['ess']) / L ** 2
        s = list(E.get((L, k), {}).values())
        err = np.std(-np.log(s), ddof=1) / np.sqrt(len(s)) / L ** 2 if len(s) > 1 else 0.
        y.append(v * 1e3); e.append(err * 1e3)
    ax.errorbar(ks, y, e, fmt='o-', color=LCOL[L], ms=2.5, lw=.8, capsize=1, label=f'$L={L}$')
    out[L] = dict(kappa=ks, y=y, err=e)
ax.axvline(.2705, color='gray', lw=.6, ls=':')
ax.set_xlabel(r'$\kappa$'); ax.set_ylabel(r'$-\log\mathrm{ESS}/V\ \ (10^{-3})$')
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc='upper right')
fig.savefig(f'{OUT}/fig_essvol.pdf', bbox_inches='tight')
json.dump(out, open('essvol.json', 'w'), indent=1)
