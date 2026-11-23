"""Running integrated autocorrelation time tau_int(W) = 1/2 + sum_{l=1}^W rho(l) of |M| vs the summation window W at L=16
in three phases, single chains (see series()); dots mark the window chosen by the Gamma method (pyerrors, S = 2) and its tau_int.
    python scripts/fig_acf.py"""
import glob, json, os, sys
import numpy as np
import pyerrors as pe
sys.path.insert(0, os.path.dirname(__file__))
from style import plt, C, W1, W2  # noqa: E402
from paper_figs import OUT  # noqa: E402
from fig_main import imh  # noqa: E402

L, V, TMAX = 16, 256, 2000
KS = (0.26, 0.2705, 0.30)


def acf(chains, tmax=TMAX):
    """chains: list of 1D series; pooled mean, chain-averaged autocovariance by FFT, normalized at t=0."""
    mu = np.mean(np.concatenate(chains))
    num = np.zeros(tmax + 1); den = np.zeros(tmax + 1)
    for x in chains:
        x = np.asarray(x, float) - mu; n = len(x)
        f = np.fft.rfft(x, 2 * n)
        c = np.fft.irfft(f * np.conj(f))[:tmax + 1]
        num += c; den += n - np.arange(tmax + 1)
    g = num / den
    return g / g[0]


def series(k):
    """Single chains, as in Singha et al.: HMC (one 2x10^6 chain, hmc1), ours IMH (the 5x10^4-step chain after 2x10^3
    burn-in of imh1, rebuilt exactly as in imh_single.py), NF IMH (first stored chain, seed 0)."""
    out = {}
    M = np.load(glob.glob(f'hmc1/*/out/hmc1/d2_L16_k{k:.3f}.npz' if round(k, 3) == k else f'hmc1/*/out/hmc1/d2_L16_k{k}.npz')[0])['hmc'][:, 0]
    out['HMC'] = [M]
    tag = f'{k:.3f}' if round(k, 3) == k else f'{k}'
    z = np.load(glob.glob(f'imh1/*/out/imh1/prop_L16_k{tag}.npz')[0]); y, lw = z['landau_ot_y'], z['landau_ot_logw']
    burn, n = 2000, 50000
    u = np.log(np.random.default_rng(1).random(burn + n)); idx = np.empty(burn + n, int); cur = 0
    for i in range(burn + n):
        if i > 0 and u[i] < lw[i] - lw[cur]:
            cur = i
        idx[i] = cur
    out['ours (IMH)'] = [(y[idx] / np.sqrt(V))[burn:]]
    d = sorted(glob.glob(f'nf/*/out/nfbase/phi4_csd/runs_k{k:.3f}/L16/rkl_mix_mix_s0'))
    out['NF (IMH)'] = [np.load(d[0] + '/chains.npz')['M'][:50000, 0]] if d else []
    return out


STY = {'HMC': (C['hmc'], '-'), 'ours (IMH)': (C['ours'], '-'), 'NF (IMH)': (C['nf'], '--')}


def main():
    single = '--single' in sys.argv                     # single-column version: three panels stacked
    if single:
        fig, ax = plt.subplots(3, 1, figsize=(W1, 4.0), sharex=True, sharey=True)
        ax = ax[None, :]
    else:
        fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharex=True, sharey=True)
        ax = ax[None, :]
    t = np.arange(TMAX + 1)
    for j, k in enumerate(KS):
        S = series(k)
        for name, ch in S.items():
            if not ch:
                continue
            col, ls = STY[name]
            x = np.abs(ch[0]); r = 0
            rho = acf([x], min(TMAX, len(x) // 4))
            run = .5 + np.cumsum(rho[1:])                         # tau_int(W), W = 1, 2, ...
            o = pe.Obs([x], ['c']); o.gamma_method(S=2.0)
            tau, w = float(o.e_tauint['c']), int(o.e_windowsize['c'])
            ax[r, j].plot(np.arange(1, len(run) + 1), run, ls, color=col, lw=1.2,
                          label={'HMC': 'HMC', 'ours (IMH)': 'few-parameter flow, IMH', 'NF (IMH)': 'convolutional flow, IMH'}[name] if (r, j) == ((0, 0) if single else (0, 2)) else None)
            ax[r, j].plot([max(w, 1)], [tau], 'o', color=col, ms=3.5)
        if single:
            ax[0, j].text(.03, .92, '(%s)' % 'abc'[j], transform=ax[0, j].transAxes, ha='left', va='top', fontsize=8)
            ax[0, j].set_ylabel(r'$\tau_{\rm int}(W)$')
        else:
            ax[0, j].text(.03, .92, '(%s)' % 'abc'[j], transform=ax[0, j].transAxes, ha='left', va='top', fontsize=8)
            ax[0, j].set_xlabel(r'summation window $W$')
    for a in ax.ravel():
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(1, TMAX); a.set_ylim(.4, 1e3)
    ax[0, 0].set_ylabel(r'$\tau_{\rm int}(W)$')
    if single:
        ax[0, 2].set_xlabel(r'summation window $W$')
        h, l = ax[0, 0].get_legend_handles_labels()
        ax[0, 2].legend(h, l, frameon=False, loc='upper right', fontsize=6.5)
        fig.tight_layout(h_pad=.3)
        fig.savefig(f'{OUT}/fig_acf_col.pdf')
    else:
        ax[0, 2].legend(frameon=False, loc='upper right', fontsize=6.5)
        fig.tight_layout(w_pad=.6)
        fig.savefig(f'{OUT}/fig_acf.pdf')


if __name__ == '__main__':
    main()
