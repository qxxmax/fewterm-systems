"""Main physics figure: columns L = 8, 16, 32; rows <|M|>, chi/V, U4 (each with a residual strip: method - long HMC chain)
and ESS.  Methods: HMC-Wolff pool (reference), ours raw / IS / IMH, RKL NF IS / IMH.
IMH for ours is built from the stored independent proposals (16384 per seed, Metropolis on log w, batch-means error);
NF IMH from its 4 stored IMH chains.  Seeds combined with paper_figs.comb.  Also writes table_main.json for the
appendix table.
    python scripts/fig_main.py"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from paper_figs import est, comb, OUT  # noqa: E402

LS = (8, 16, 32)
Q = ('absM', 'chiV', 'U4')


def stats(m):
    a2, a4 = (m ** 2).mean(), (m ** 4).mean()
    return np.array([np.abs(m).mean(), a2, 1 - a4 / (3 * a2 ** 2)])


def batch_err(m, nb=None):
    """Errors of stats() = (<|M|>, <M^2>, U4) on an MCMC series of M, by the Gamma method (pyerrors, S=2) as for all
    other chains in the paper.  m: [T] (one chain) or [T, C] (C replica chains)."""
    import pyerrors as pe
    m = np.asarray(m, float).reshape(len(m), -1)
    names = [f'c|r{c}' for c in range(m.shape[1])]
    a1 = pe.Obs([np.abs(m[:, c]) for c in range(m.shape[1])], names)
    a2 = pe.Obs([m[:, c] ** 2 for c in range(m.shape[1])], names)
    a4 = pe.Obs([m[:, c] ** 4 for c in range(m.shape[1])], names)
    u4 = 1 - a4 / (3 * a2 * a2)
    out = []
    for o in (a1, a2, u4):
        o.gamma_method(S=2.0)
        out.append(o.dvalue)
    return np.array(out)


def imh(lw, rng):
    idx = np.empty(len(lw), int); cur = 0
    u = np.log(rng.random(len(lw)))
    for i in range(len(lw)):
        if u[i] < lw[i] - lw[cur]:
            cur = i
        idx[i] = cur
    return idx


def load_ours():
    S = {}
    rng = np.random.default_rng(5)
    for f in glob.glob('obs/*/out/obs2d/*.npz') + glob.glob('seeds/95024/out/obs2d_s/*.npz'):
        j = json.load(open(f[:-4] + '.json')); L = j['L']
        V = L * L; z = np.load(f); y, lw = z['landau_ot_y'], z['landau_ot_logw']
        m = y[imh(lw, rng)] / np.sqrt(V)
        S.setdefault((L, round(j['kappa'], 4)), []).append(dict(
            hmc=est(z['pool_y'], V), raw=est(y, V), IS=est(y, V, lw), IMH=(stats(m), batch_err(m)), ess=j['landau_ot']['ESS']))
    return S


def load_nf():
    S = {}
    for rec in glob.glob('nf*/*/out/nfbase/phi4_csd/*/L*/rkl_mix_mix_s*/record.json'):
        r = json.load(open(rec)); L = r['L']
        if L not in LS:
            continue
        d = os.path.dirname(rec)
        p = np.load(os.path.join(d, 'proposals.npz')); c = np.load(os.path.join(d, 'chains.npz'))
        S.setdefault((L, round(r['kappa'], 4)), []).append(dict(
            IS=est(p['M'] * L, L * L, p['logw']), IMH=(stats(c['M']), batch_err(c['M'])), ess=r['ESS']))
    return S


def load_hmc_ref():
    """Long single-chain pure HMC (2x10^6 trajectories, eps = 0.1; hmc1/*): <|M|>, <M^2>, U4 with Gamma-method errors.
    These observables are invariant under M -> -M, so they are unaffected by the absence of sign flips in the broken phase."""
    R = {}
    for f in glob.glob('hmc1/*/out/hmc1/*.json'):
        r = json.load(open(f)); o = r['hmc']['obs']
        R[(r['L'], round(r['kappa'], 4))] = (np.array([o['absM']['mean'], o['M2']['mean'], o['U4']['mean']]),
                                             np.array([o['absM']['err'], o['M2']['err'], o['U4']['err']]))
    return R


def combine(S, keys):
    out = {}
    for k, lst in S.items():
        o = {}
        for m in keys:
            v, e = comb([x[m][0][-3:] if len(x[m][0]) == 4 else x[m][0] for x in lst],
                        [x[m][1][-3:] if len(x[m][1]) == 4 else x[m][1] for x in lst])
            o[m] = (np.atleast_1d(v), np.atleast_1d(e))
        o['ess'] = float(np.mean([x['ess'] for x in lst])); o['n'] = len(lst)
        out[k] = o
    return out


def plot(O, N):
    """Two figures. fig_main: <|M|>, chi/V, U4 side by side, each with a residual strip below (colour = L, marker = method).
    fig_ess: ESS of the few-parameter flow.  fig_conv (appendix): ESS of the few-parameter flow and of the convolutional
    flow in the standard setting (3 seeds), trained 8x longer and with a wider network (1 seed each)."""
    from style import LCOL, W1, W2
    from matplotlib.lines import Line2D
    N = {k: v for k, v in N.items() if k[0] <= 16}                 # convolutional flow shown for L <= 16 only
    M = [('IS', O, 's', False, 'reweighting (IS)'), ('IMH', O, 'o', True, 'IMH'),
         ('raw', O, 'x', True, 'uncorrected')]
    off = (-.0002, 0., .0002)
    fig = plt.figure(figsize=(W2, 3.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.3], hspace=.08, wspace=.36)
    lab = [r'$\langle|M|\rangle$', r'$\chi/V$', '$U_4$']
    lim = 6.
    for q in range(3):
        a = fig.add_subplot(gs[0, q]); ar = fig.add_subplot(gs[1, q], sharex=a)
        ar.axhline(0, color='k', lw=.6)
        for L in LS:
            col = LCOL[L]
            ks = np.array(sorted(k for (l, k) in O if l == L))
            ref = np.array([O[(L, k)]['hmc'][0][q] for k in ks]); rer = np.array([O[(L, k)]['hmc'][1][q] for k in ks])
            a.fill_between(ks, ref - rer, ref + rer, color=col, alpha=.18, lw=0)
            a.plot(ks, ref, '-', color=col, lw=.9)
            for i, (key, D, mk, filled, name) in enumerate(M):
                kk = np.array([k for k in ks if (L, k) in D])
                if not len(kk):
                    continue
                v = np.array([D[(L, k)][key][0][q] for k in kk]); e = np.array([D[(L, k)][key][1][q] for k in kk])
                hr = np.array([O[(L, k)]['hmc'][1][q] for k in kk])
                d = (v - np.array([O[(L, k)]['hmc'][0][q] for k in kk])) / np.sqrt(e ** 2 + hr ** 2)   # pull
                kw = dict(fmt=mk, color=col, ms=3 if mk != 'x' else 3.5, lw=.6, mfc=col if filled else 'white', mew=.8)
                if key != 'raw':
                    a.errorbar(kk + off[i], v, e, **kw)
                ar.errorbar(kk + off[i], d, fmt=mk, color=col, ms=2.6 if mk != 'x' else 3, mfc=col if filled else 'white', mew=.7)
        for yy in (-2, 2):
            ar.axhline(yy, color='gray', lw=.4, ls=':')
        ar.set_ylim(-lim, lim)
        for L in LS:                                         # points outside the strip: arrow at the edge
            for i, (key, D, mk, filled, name) in enumerate(M):
                for k in [k for (l, k) in D if l == L]:
                    d = (D[(L, k)][key][0][q] - O[(L, k)]['hmc'][0][q]) / math.hypot(D[(L, k)][key][1][q], O[(L, k)]['hmc'][1][q])
                    if abs(d) > lim:
                        ar.annotate('', xy=(k + off[i], np.sign(d) * .97 * lim), xytext=(k + off[i], np.sign(d) * .5 * lim),
                                    arrowprops=dict(arrowstyle='->', color=LCOL[L], lw=.8))
        a.set_ylabel(lab[q]); a.tick_params(labelbottom=False)
        ar.set_xlabel(r'$\kappa$'); ar.set_xticks([.25, .26, .27, .28, .29, .30])
        if q == 0:
            ar.set_ylabel(r'$\Delta/\sigma$')
        a.set_xlim(.247, .303)
    hL = [Line2D([], [], color=LCOL[L], lw=4, alpha=.6) for L in LS]
    hM = [Line2D([], [], color='k', lw=.9)] + [Line2D([], [], color='k', marker=mk, ls='', ms=3.5, mfc='k' if f else 'white')
                                              for (_, _, mk, f, _) in M]
    fig.legend(hL + hM, [f'$L={L}$' for L in LS] + ['HMC'] + [m[4] for m in M], loc='lower center',
               bbox_to_anchor=(.5, .91), ncol=7, frameon=False, fontsize=7, handlelength=1.2, handletextpad=.3, columnspacing=1.1)
    fig.savefig(f'{OUT}/fig_main.pdf')

    fig, ae = plt.subplots(figsize=(W1, 2.2))
    for L in LS:
        kk = np.array(sorted(k for (l, k) in O if l == L))
        ae.semilogy(kk, [O[(L, k)]['ess'] for k in kk], 's-', color=LCOL[L], ms=3, lw=.9, mfc='white', mew=.8, label=f'$L={L}$')
    ae.set_ylim(5e-3, 1.5); ae.set_xlim(.247, .303); ae.set_ylabel('ESS'); ae.set_xlabel(r'$\kappa$')
    ae.legend(loc='lower center', bbox_to_anchor=(.5, 1.), ncol=3, frameon=False, fontsize=7, handlelength=1.6,
              handletextpad=.4, columnspacing=1.)
    fig.savefig(f'{OUT}/fig_ess.pdf')

    X = {}                                               # 8x longer training and wider network, one seed each
    for f in glob.glob('nflong/95188/nflong/*/phi4_csd/*/L*/rkl_mix_mix_s0/record.json'):
        r = json.load(open(f)); X[(f.split('/')[3], r['L'], round(r['kappa'], 4))] = r['ESS']
    fig, ae = plt.subplots(figsize=(W1, 2.4))
    for L in (8, 16):
        for D, mk, ls in ((O, 's', '-'), (N, 'D', '--')):
            kk = np.array(sorted(k for (l, k) in D if l == L))
            ae.semilogy(kk, [D[(L, k)]['ess'] for k in kk], mk + ls, color=LCOL[L], ms=3, lw=.9, mfc='white', mew=.8)
        for v, mk in (('long', '^'), ('wide', '*')):
            kk = sorted(k for (w, l, k) in X if w == v and l == L)
            ae.semilogy(kk, [X[(v, L, k)] for k in kk], mk, color=LCOL[L], ms=4.5 if mk == '*' else 3.5)
    ae.set_ylim(5e-4, 1.5); ae.set_xlim(.247, .303); ae.set_ylabel('ESS'); ae.set_xlabel(r'$\kappa$')
    h = [Line2D([], [], color=LCOL[L], lw=4, alpha=.6) for L in (8, 16)] + \
        [Line2D([], [], color='k', marker='s', ms=3, mfc='white', lw=.9),
         Line2D([], [], color='k', marker='D', ms=3, mfc='white', lw=.9, ls='--'),
         Line2D([], [], color='k', marker='^', ms=3.5, ls=''), Line2D([], [], color='k', marker='*', ms=4.5, ls='')]
    lb = ['$L=8$', '$L=16$', 'few-parameter flow', 'convolutional flow', '$8\\times$ longer training', 'wider network']
    o = (0, 2, 4, 1, 3, 5)
    ae.legend([h[i] for i in o], [lb[i] for i in o], loc='lower center', bbox_to_anchor=(.5, 1.), ncol=2, frameon=False,
              fontsize=7, handlelength=1.6, handletextpad=.4, columnspacing=1.)
    fig.savefig(f'{OUT}/fig_conv.pdf')


def main():
    import pickle
    cache = 'cache_main.pkl'
    if os.path.exists(cache) and '--reload' not in sys.argv:
        O, N = pickle.load(open(cache, 'rb'))
    else:
        O = combine(load_ours(), ('hmc', 'raw', 'IS', 'IMH'))
        ref = load_hmc_ref()
        for key in O:                                   # reference = long HMC chain where available (else the pool)
            O[key]['pool'] = O[key]['hmc']
            if key in ref:
                O[key]['hmc'] = ref[key]
            O[key]['ref'] = 'hmc1' if key in ref else 'pool'
        N = combine(load_nf(), ('IS', 'IMH'))
        pickle.dump((O, N), open(cache, 'wb'))
    plot(O, N)
    tab = {f'L{L}_k{k}': {'n_seeds': O[(L, k)]['n'], **{m: [O[(L, k)][m][0].tolist(), O[(L, k)][m][1].tolist()] for m in ('hmc', 'raw', 'IS', 'IMH')},
                         'ess': O[(L, k)]['ess'],
                         **({f'nf_{m}': [N[(L, k)][m][0].tolist(), N[(L, k)][m][1].tolist()] for m in ('IS', 'IMH')} if (L, k) in N else {}),
                         **({'nf_ess': N[(L, k)]['ess']} if (L, k) in N else {})}
           for (L, k) in O}
    json.dump(tab, open('table_main.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
