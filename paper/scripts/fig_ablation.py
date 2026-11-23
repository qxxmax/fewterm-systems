"""Role of the components at L = 16 (seeds 1-3; abl16/*): (a) ESS of the full flow (double-well zero-mode base + zero-mode
pairing), without the nearest-neighbor terms, without pairing, with a Gaussian base (with / without pairing), and of
the base distribution alone (Gaussian with the training power spectrum + double-well zero mode, no flow);
(b) largest zero-mode coefficient of the fitted velocity, a measure of the stiffness of the flow.  Runs in which the
integration failed or exceeded the time limit are marked at the top of (b).
    python scripts/fig_ablation.py
"""
import glob, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from style import plt, C, W2, panel  # noqa: E402
from paper_figs import OUT  # noqa: E402

VAR = [('landau_ot', 'full', dict(color=C['ours'], marker='s', ls='-')),
       ('landau_ot_nonn', 'no nearest-neighbor terms', dict(color=C['ours'], marker='s', ls='--', mfc='white')),
       ('landau', 'no zero-mode pairing', dict(color='#7570b3', marker='o', ls='-')),
       ('gauss_ot', 'Gaussian base, pairing', dict(color='#1b9e77', marker='^', ls='-')),
       ('gauss', 'Gaussian base, no pairing', dict(color='0.5', marker='v', ls='-')),
       ('base', 'no flow (base only)', dict(color='k', marker='x', ls=':'))]
ESS, COEF, FAIL = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(3))
lines = open(sorted(glob.glob('abl16/*/lines.txt'))[0]).read().split('\n') if glob.glob('abl16/*/lines.txt') else []
for f in glob.glob('abl16/*/out/abl16/d2_L16_k*_s*.json'):
    d = json.load(open(f)); k = round(d['kappa'], 4)
    for v, _, _ in VAR[:-1]:
        if v not in d:
            continue
        if 'ESS' in d[v]:
            ESS[k][v].append(d[v]['ESS']); COEF[k][v].append(d[v]['max_abs_zero_mode_coeff'])
        else:
            FAIL[k][v].append(1)
for f in glob.glob('abl16/*/out/abl16/base_L16_k*_s*.json'):
    d = json.load(open(f)); ESS[round(d['kappa'], 4)]['base'].append(d['spec_landau']['ESS'])
# Gaussian-base runs that hit the 20000 s limit (whole line killed): kappa 0.28 and 0.30, seeds 1 and 3
for k in (0.28, 0.30):
    FAIL[k]['gauss'] += [1] * (3 - len(ESS[k]['gauss']) - len(FAIL[k]['gauss']))

ks = sorted(ESS)
from style import W1
fig, ax = plt.subplots(figsize=(W1, 3.1))
for v, lab, st in VAR:
    kk = [k for k in ks if ESS[k][v]]
    m = [np.mean(ESS[k][v]) for k in kk]
    e = [np.std(ESS[k][v], ddof=1) / np.sqrt(len(ESS[k][v])) if len(ESS[k][v]) > 1 else 0 for k in kk]
    ax.errorbar(kk, m, e, label=lab, ms=3.5, lw=1, **st)
fk = [k for k in ks if FAIL[k]['gauss']]
ax.plot(fk, [7e-3] * len(fk), ls='none', marker='X', color='0.5', ms=6)
ax.set_yscale('log'); ax.set_ylim(5e-3, 1.3)
ax.set_xlabel(r'$\kappa$'); ax.set_ylabel('ESS')
ax.legend(frameon=False, fontsize=7, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, columnspacing=.8, handletextpad=.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_ablation.pdf')
json.dump({str(k): {v: dict(ess=ESS[k][v], coef=COEF[k][v], fail=len(FAIL[k][v])) for v, _, _ in VAR} for k in ks},
          open('ablation.json', 'w'), indent=1)
