"""tau_int vs L measured exactly as in Singha et al. (arXiv:2604.10209):
  pure HMC: one chain of 2x10^6 trajectories per (L, kappa) after 10^4 warm-up, eps = 0.1, 10 leapfrog steps, trajectory
            length 1 (hmc1/*);  ours IMH: one chain of 5x10^4 steps after 2x10^3 burn-in on fresh proposals (imh1/*).
  Gamma method (pyerrors, S = 2) for both; |M| (solid) and signed M (dotted).  HMC tau(M) is shown only where the chain
  changes the sign of M at least 100 times; otherwise an upward arrow marks a value beyond the plotted range.
    python scripts/fig_tau_single.py
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from style import plt, C, W2, panel  # noqa: E402
from paper_figs import OUT  # noqa: E402

H, I = {}, {}
for f in glob.glob('hmc1/*/out/hmc1/*.json'):
    if not os.path.exists(f[:-5] + '.npz'):       # reference-only runs (no stored trace) are not used here
        continue
    r = json.load(open(f)); key = (r['L'], round(r['kappa'], 4))
    M = np.load(f[:-5] + '.npz')['hmc'][:, 0]
    H[key] = dict(absM=r['hmc']['obs']['absM'], M=r['hmc']['obs']['M'], flips=int((np.diff(np.sign(M)) != 0).sum()))
for f in glob.glob('imh1/*/out/imh1/imh_*.json'):
    r = json.load(open(f)); k = round(float(f.split('_k')[1][:-5]), 4)
    I[(r['L'], k)] = r

ks = (0.26, 0.2705, 0.28, 0.30)
OT = json.load(open('tau_ours_imh.json'))   # few-parameter flow IMH: one chain per independent flow (4), median and min-max
from style import W1
# (1) main text: tau_int vs L, rows |M| (top) and signed M (bottom), columns kappa; full width
fig, axs = plt.subplots(2, 4, figsize=(W2, 3.3), sharex=True, sharey='row')
out = {}
for j, k in enumerate(ks):
    Ls = sorted(L for (L, kk) in H if kk == k)
    ok = np.array([H[(L, k)]['flips'] >= 100 for L in Ls])
    for r, q in enumerate(('absM', 'M')):
        a = axs[r, j]
        th = np.array([H[(L, k)][q]['tau_int'] for L in Ls]); eh = np.array([H[(L, k)][q]['tau_int_err'] for L in Ls])
        sel = ok if q == 'M' else np.ones(len(Ls), bool)
        a.errorbar(np.array(Ls)[sel], th[sel], eh[sel], fmt='v-', color=C['hmc'], ms=3.5, label='HMC')
        if q == 'M':
            for L in np.array(Ls)[~ok]:                    # HMC: too few sign changes, tau(M) beyond the plotted range
                a.annotate('', xy=(L, 1.6e4), xytext=(L, 4e3), arrowprops=dict(arrowstyle='->', color=C['hmc'], lw=.8))
        ti = np.array([OT[f'{L}_{k}'][q]['median'] for L in Ls])   # present flow: median over independent flows
        ei = np.array([[ti[i] - min(OT[f'{L}_{k}'][q]['all']), max(OT[f'{L}_{k}'][q]['all']) - ti[i]] for i, L in enumerate(Ls)]).T
        a.errorbar(Ls, ti, ei, fmt='s-', color=C['ours'], mfc='white', ms=3.5, capsize=1.5, label='few-parameter flow, IMH')
        a.text(.96, .05, '(%s)' % 'abcdefgh'[4 * r + j], transform=a.transAxes, ha='right', va='bottom', fontsize=8)
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(7, 37)
        a.set_xticks([8, 16, 32]); a.set_xticklabels(['8', '16', '32']); a.xaxis.set_minor_formatter(plt.NullFormatter())
    thm = np.array([H[(L, k)]['M']['tau_int'] for L in Ls])
    out[k] = dict(L=Ls, tau_hmc_absM=[H[(L, k)]['absM']['tau_int'] for L in Ls],
                  tau_hmc_M=[float(x) if o else None for x, o in zip(thm, ok)], hmc_flips=[H[(L, k)]['flips'] for L in Ls],
                  tau_imh_absM=[OT[f'{L}_{k}']['absM']['median'] for L in Ls], tau_imh_M=[OT[f'{L}_{k}']['M']['median'] for L in Ls],
                  imh_acceptance=[round(I[(L, k)]['acceptance'], 3) for L in Ls])
    axs[1, j].set_xlabel('$L$')
axs[0, 0].set_ylim(.4, 1e3); axs[1, 0].set_ylim(.4, 2e4)
axs[0, 0].set_ylabel(r'$\tau_{\rm int}(|M|)$'); axs[1, 0].set_ylabel(r'$\tau_{\rm int}(M)$')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', bbox_to_anchor=(.5, .97), ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(h_pad=.4, w_pad=.4)
fig.savefig(f'{OUT}/fig_tau_main.pdf', bbox_inches='tight')
json.dump(out, open('tau_single.json', 'w'), indent=1)
