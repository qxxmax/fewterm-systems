"""Evaluation of a trained convolutional flow: reweighting, IMH chains and ESS with an exact density.

Checkpoints come from train_nf_phi4.py or train_nf_phi4_mixture.py (16 affine checkerboard couplings,
hidden [8, 8], kernel 3) and are evaluated in float64 on one CPU thread.
  single : q_psi
  mix    : q_mix(phi) = [q_psi(phi) + q_psi(-phi)] / 2   (exact Z2 mixture, reverse pass for q_psi(-phi))
Output: <FEWTERM_BASELINE_OUT>/phi4_nf/L{L}/... with record.json, proposals.npz and chains.npz."""
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from baselines import paths as _paths  # noqa: E402  (repo-relative paths, env overrides)
import train_nf_phi4 as nfsrc  # noqa: E402

torch.set_num_threads(1)
CKPT = _paths.nf_grid()
ROOT = _paths.out() / 'phi4_nf'
N_IS, N_CHAIN, WARM = int(os.environ.get("NF_N_IS", 65536)), int(os.environ.get("NF_N_CHAIN", 65536)), 4096


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def ess(lw):
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (len(w) * np.square(w).sum()))


def imh(logw, steps, warmup, seed):
    """Four independent IMH chains on the proposal stream logw (proposal i of chain c is 4 (i+1) + c)."""
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


def ckpt_path(name):
    """A bare name is looked up in FEWTERM_NF_GRID; otherwise give the full path."""
    q = Path(name)
    return q if q.is_absolute() else CKPT / name


def load(name):
    base = ckpt_path(name)
    meta = json.loads(base.with_name(base.name + '.meta.json').read_text(encoding='utf-8'))
    L = meta['L']
    layers = nfsrc.make_phi4_affine_layers(n_layers=meta['n_layers'], lattice_shape=(L, L),
                                           hidden_sizes=meta['hidden_sizes'], kernel_size=meta['kernel_size'])
    layers.load_state_dict(torch.load(base.with_name(base.name + '.pt'), map_location='cpu', weights_only=True))
    return layers.double().eval(), meta


def forward(layers, z):
    x, logq = z, -.5 * (z ** 2).sum((1, 2)) - .5 * z[0].numel() * math.log(2 * math.pi)
    for layer in layers:
        x, logJ = layer(x)
        logq = logq - logJ
    return x, logq


def log_q_psi(layers, phi):
    z, logJ = phi, torch.zeros(len(phi), dtype=phi.dtype)
    for layer in reversed(list(layers)):
        z, lj = layer.reverse(z)
        logJ = logJ + lj
    return -.5 * (z ** 2).sum((1, 2)) - .5 * z[0].numel() * math.log(2 * math.pi) + logJ


@torch.no_grad()
def generate(layers, L, kappa, lam, n, seed, mode):
    gen = torch.Generator().manual_seed(seed)
    action = nfsrc.ScalarPhi4Action(kappa, lam)
    M, S, lw = np.empty(n), np.empty(n), np.empty(n)
    for start in range(0, n, 512):
        size = min(512, n - start)
        z = torch.randn(size, L, L, dtype=torch.float64, generator=gen)
        phi, logq = forward(layers, z)
        if mode == 'mix':
            sign = torch.where(torch.rand(size, dtype=torch.float64, generator=gen) < .5, -1., 1.)
            phi = phi * sign[:, None, None]
            lq_plus, lq_minus = log_q_psi(layers, phi), log_q_psi(layers, -phi)
            logq = torch.logaddexp(lq_plus, lq_minus) - math.log(2)
        s = action(phi)
        M[start:start + size] = phi.mean((1, 2)).numpy()
        S[start:start + size] = s.numpy()
        lw[start:start + size] = (-s - logq).numpy()
    if not np.isfinite(lw).all():
        raise RuntimeError('nonfinite weights')
    return M, S, lw


@torch.no_grad()
def selftest(layers, L):
    z = torch.randn(4, L, L, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    phi, logq = forward(layers, z)
    return float((log_q_psi(layers, phi) - logq).abs().max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', choices=('single', 'mix'), required=True)
    a = p.parse_args()
    layers, meta = load(a.ckpt)
    L, kappa, lam = meta['L'], meta['kappa'], meta['lam']
    assert abs(lam - .022) < 1e-12
    err = selftest(layers, L)
    if err > 1e-9:
        raise RuntimeError(f'inverse/logdet selftest failed {err}')
    stem = ckpt_path(a.ckpt).name
    variant = stem.split('_')[-1] if not Path(a.ckpt).is_absolute() else stem.split('_')[-2]
    runs = ROOT / ('runs' if abs(kappa - .27) < 1e-12 else f'runs_k{kappa:.3f}')
    tag = f'rkl_{variant}_{a.mode}' + ('' if not Path(a.ckpt).is_absolute() else '_' + stem.split('_')[-1])
    out = runs / f'L{L}' / tag
    out.mkdir(parents=True)
    files = [Path(__file__), Path(nfsrc.__file__), ckpt_path(a.ckpt).with_name(stem + '.pt')]
    if variant == 'mix':
        files.append(Path(nfsrc.__file__).with_name('train_nf_phi4_mixture.py'))
    write(out / 'sources.json', {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    seed = (31040000 + 100 * ['single', 'z2', 'mix'].index(variant) + (10 if a.mode == 'mix' else 0)
            + 1000 * L + int(round(kappa * 1e4)) * 100000 + 7 * int(meta.get('seed', 0))
            + (50 if Path(a.ckpt).is_absolute() else 0))  # grid checkpoints: stream distinct from the originals
    t = time.perf_counter()
    M_is, S_is, lw_is = generate(layers, L, kappa, lam, N_IS, seed, a.mode)
    sec = (time.perf_counter() - t) / N_IS
    n = 4 * (N_CHAIN + WARM + 1)
    M, S, lw = generate(layers, L, kappa, lam, n, seed + 1, a.mode)
    idx, acc = imh(lw, N_CHAIN, WARM, seed + 2)
    idx, acc = idx[WARM:], acc[WARM:]
    m = M[idx]
    with (out / 'chains.npz').open('xb') as fh:
        np.savez_compressed(fh, M=m, absM=np.abs(m), M2=m ** 2, M4=m ** 4, accepted=acc, S=S[idx])
    with (out / 'proposals.npz').open('xb') as fh:
        np.savez_compressed(fh, M=M_is, S=S_is, logw=lw_is)
    rec = {'L': L, 'kappa': kappa, 'ckpt': a.ckpt, 'variant': variant, 'mode': a.mode, 'eval_seed': seed, 'selftest_logq_inverse': err,
           'seconds_per_proposal': sec, 'ESS': ess(lw_is), 'Var_logw_per_site': float(lw_is.var() / L ** 2),
           'acceptance': float(acc.mean()), 'positive_fraction_proposals': float((M_is > 0).mean()),
           'sign_changes_per_chain': (np.diff(np.sign(m), axis=0) != 0).sum(0).tolist(),
           'training': {k: meta[k] for k in ('epochs', 'batch_size', 'n_params', 'final_ess') if k in meta},
           'checkpoint': str(ckpt_path(a.ckpt))}
    for k in ('wall_min', 'wall_seconds', 'train_seconds', 'device'):
        if k in meta:
            rec['training'][k] = meta[k]
    write(out / 'record.json', rec)
    (out / 'complete.json').write_text(json.dumps({'complete': True}), encoding='utf-8')
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
