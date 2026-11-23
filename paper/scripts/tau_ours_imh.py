"""IMH tau_int of the present flow as median and 16-84% range over independent flows (seeds), as for the NF:
pools = obs2d (seed 1), obs2d_s (seeds 2, 3), imh1 (fresh 52000 candidates); one IMH chain per pool, Gamma method (S=2).
    python scripts/tau_ours_imh.py   -> tau_ours_imh.json"""
import glob, json, os
import numpy as np
import pyerrors as pe

KS = (0.26, 0.2705, 0.28, 0.30)


def imh(lw, rng):
    idx = np.empty(len(lw), int); cur = 0
    u = np.log(rng.random(len(lw)))
    for i in range(len(lw)):
        if u[i] < lw[i] - lw[cur]:
            cur = i
        idx[i] = cur
    return idx


def tau(x):
    o = pe.Obs([np.asarray(x, float)], ['c']); o.gamma_method(S=2.0)
    return float(o.e_tauint['c'])


pools = {}
for f in glob.glob('obs/*/out/obs2d/*.npz') + glob.glob('seeds/95024/out/obs2d_s/*.npz'):
    j = json.load(open(f[:-4] + '.json')); pools.setdefault((j['L'], round(j['kappa'], 4)), []).append(f)
for f in glob.glob('imh1/*/out/imh1/prop_*.npz'):
    b = os.path.basename(f)[5:-4]; L = int(b.split('_')[0][1:]); k = round(float(b.split('_k')[1]), 4)
    pools.setdefault((L, k), []).append(f)
rng = np.random.default_rng(11)
out = {}
for (L, k), fs in sorted(pools.items()):
    if k not in KS:
        continue
    ta, tm = [], []
    for f in fs:
        z = np.load(f); y, lw = z['landau_ot_y'], z['landau_ot_logw']
        m = y[imh(lw, rng)] / np.sqrt(L * L)
        ta.append(tau(np.abs(m))); tm.append(tau(m))
    out[f'{L}_{k}'] = {q: dict(median=float(np.median(t)), p16=float(np.percentile(t, 16)), p84=float(np.percentile(t, 84)), all=t)
                       for q, t in (('absM', ta), ('M', tm))}
    out[f'{L}_{k}']['n'] = len(fs)
    print(L, k, len(fs), np.round(ta, 2), np.round(tm, 2))
json.dump(out, open('tau_ours_imh.json', 'w'), indent=1)
