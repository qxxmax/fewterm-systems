#!/usr/bin/env python3
"""Fixed-step HMC / Wolff+HMC chains for phi4 autocorrelation studies.

The measurement spacing is one raw Markov step for all kappas and all sizes.
This script is separate from TI production scripts, whose adaptive thinning is
useful for free energies but unsuitable for fixed-spacing autocorrelation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit
from scipy.special import kv as bessel_kv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action import phi4_action, phi4_force

try:
    import pyerrors as pe

    HAS_PE = True
except ImportError:
    pe = None
    HAS_PE = False


LAM_DEFAULT = 0.022


def f0_per_site(lam: float) -> float:
    a = (1 - 2 * lam) ** 2 / (8 * lam)
    return float(-np.log(np.sqrt((1 - 2 * lam) / (4 * lam)) * np.exp(a) * bessel_kv(0.25, a)))


@torch.no_grad()
def hmc_proposal(phi: torch.Tensor, epsilon: float, tau: float, lam: float, kappa: float):
    n_leap = max(1, int(round(tau / epsilon)))
    eps = tau / n_leap
    pi = torch.randn_like(phi)
    sdims = tuple(range(1, phi.ndim))

    action_old = phi4_action(phi, lam, kappa)
    h_old = 0.5 * pi.pow(2).sum(dim=sdims) + action_old

    prop = phi.clone()
    mom = pi.clone()
    mom = mom + 0.5 * eps * phi4_force(prop, lam, kappa)
    for _ in range(n_leap - 1):
        prop = prop + eps * mom
        mom = mom + eps * phi4_force(prop, lam, kappa)
    prop = prop + eps * mom
    mom = mom + 0.5 * eps * phi4_force(prop, lam, kappa)

    action_prop = phi4_action(prop, lam, kappa)
    h_prop = 0.5 * mom.pow(2).sum(dim=sdims) + action_prop
    delta_h = h_prop - h_old
    log_accept = -delta_h
    accept = torch.log(torch.rand(phi.shape[0], device=phi.device).clamp_min(1e-30)) <= log_accept
    mask = accept.view(-1, *([1] * (phi.ndim - 1))).expand_as(phi)
    phi_after = torch.where(mask, prop, phi)
    return phi_after, prop, accept, action_old, action_prop, h_old, h_prop, delta_h, log_accept, n_leap


def measure_dsdk(phi: torch.Tensor) -> torch.Tensor:
    nn = torch.zeros_like(phi)
    for mu in range(1, phi.ndim):
        nn += torch.roll(phi, 1, dims=mu) + torch.roll(phi, -1, dims=mu)
    return -(phi * nn).sum(dim=tuple(range(1, phi.ndim)))


@njit(cache=True)
def _build_nb_rect(Lx: int, Lt: int):
    V = Lx * Lt
    nb = np.empty((V, 4), dtype=np.int32)
    for ix in range(Lx):
        for it in range(Lt):
            i = ix * Lt + it
            nb[i, 0] = ix * Lt + (it + 1) % Lt
            nb[i, 1] = ix * Lt + (it - 1 + Lt) % Lt
            nb[i, 2] = ((ix + 1) % Lx) * Lt + it
            nb[i, 3] = ((ix - 1 + Lx) % Lx) * Lt + it
    return nb


@njit(cache=True)
def _wolff_one(phi_flat, nb, kappa: float, V: int):
    seed = np.random.randint(0, V)
    included = np.zeros(V, dtype=np.int32)
    stack = np.empty(V, dtype=np.int32)
    included[seed] = 1
    stack[0] = seed
    sp = 1
    cluster_size = 1
    while sp > 0:
        sp -= 1
        i = stack[sp]
        for jpos in range(4):
            j = nb[i, jpos]
            if included[j] == 0:
                bond = phi_flat[i] * phi_flat[j]
                if bond > 0.0 and np.random.random() < 1.0 - np.exp(-4.0 * kappa * bond):
                    included[j] = 1
                    stack[sp] = j
                    sp += 1
                    cluster_size += 1
    for i in range(V):
        if included[i]:
            phi_flat[i] = -phi_flat[i]
    return cluster_size


@njit(cache=True)
def _wolff_batch(phi_batch, nb, kappa: float, B: int, V: int):
    sizes = np.empty(B, dtype=np.int32)
    for b in range(B):
        sizes[b] = _wolff_one(phi_batch[b], nb, kappa, V)
    return sizes


def wolff_batch_update(phi_gpu: torch.Tensor, nb, kappa: float):
    B, Lx, Lt = phi_gpu.shape
    V = Lx * Lt
    phi_np = phi_gpu.detach().cpu().to(torch.float64).numpy().reshape(B, V).copy()
    sizes = _wolff_batch(phi_np, nb, float(kappa), B, V)
    result = torch.from_numpy(phi_np.reshape(B, Lx, Lt)).to(device=phi_gpu.device, dtype=phi_gpu.dtype)
    return result, sizes


def as_np(t: torch.Tensor, dtype) -> np.ndarray:
    return t.detach().cpu().numpy().astype(dtype, copy=False)


def autocorr_fft(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    lag = min(max_lag, n - 1)
    x = x - x.mean()
    ac0 = np.dot(x, x) / n
    if lag <= 0 or ac0 <= 0:
        out = np.zeros(lag + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n]
    acf = acf / np.arange(n, 0, -1)
    return acf[: lag + 1] / acf[0]


def sokal_tau(values: np.ndarray, c: float = 5.0) -> tuple[float, float]:
    rho = autocorr_fft(values, min(len(values) // 4, 4096))
    tau = 0.5
    window = 0
    for lag in range(1, len(rho)):
        tau += rho[lag]
        window = lag
        if lag > c * max(2.0 * tau, 1.0):
            break
    err = tau * np.sqrt(2.0 * (2 * window + 1) / max(len(values), 1))
    return float(max(tau, 0.5)), float(err)


def gamma_stats(values: np.ndarray, name: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    out = {
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "err": float("nan"),
        "tau_int": float("nan"),
        "dtau_int": float("nan"),
        "method": "none",
        "n": int(values.size),
    }
    if values.size < 2:
        return out
    if HAS_PE:
        try:
            obs = pe.Obs([values.tolist()], [name])
            obs.gamma_method()
            out.update(
                {
                    "mean": float(obs.value),
                    "err": float(obs.dvalue),
                    "tau_int": float(obs.e_tauint[name]),
                    "dtau_int": float(obs.e_dtauint[name]),
                    "method": "pyerrors.gamma_method_fixed_step",
                }
            )
            return out
        except Exception as exc:
            out["pyerrors_error"] = repr(exc)
    tau, dtau = sokal_tau(values)
    out.update(
        {
            "err": float(np.std(values, ddof=1) / np.sqrt(values.size)),
            "tau_int": tau,
            "dtau_int": dtau,
            "method": "sokal_fallback_fixed_step",
        }
    )
    return out


def block_jackknife(values: np.ndarray, fn, n_blocks: int) -> tuple[float, float, int]:
    values = np.asarray(values)
    n = values.shape[0]
    if n == 0:
        return float("nan"), float("nan"), 0
    nb = max(1, min(int(n_blocks), n))
    full = float(fn(values))
    if nb <= 1:
        return full, float("nan"), nb
    blocks = np.array_split(np.arange(n), nb)
    vals = []
    for block in blocks:
        mask = np.ones(n, dtype=bool)
        mask[block] = False
        vals.append(float(fn(values[mask])))
    vals = np.asarray(vals, dtype=np.float64)
    err = np.sqrt((nb - 1) / nb * np.sum((vals - vals.mean()) ** 2))
    return full, float(err), nb


def empty_buf() -> dict:
    keys = [
        "step",
        "chain_id",
        "hmc_accept",
        "wolff_applied",
        "wolff_cluster_size",
        "log_accept",
        "delta_h",
        "phi_chain",
        "phi_hmc_proposal",
        "phi_hmc_accepted",
        "phi_hmc_rejected",
        "accepted_step",
        "accepted_chain",
        "rejected_step",
        "rejected_chain",
        "S_current",
        "S_hmc_proposal",
        "dS_dk_current",
        "mag",
        "abs_mag",
    ]
    return {key: [] for key in keys}


def flush_chunk(out_dir: Path, chunk_id: int, buf: dict, phi_dtype, compress: bool) -> str:
    if not buf["step"]:
        return ""
    phi_shape = np.stack(buf["phi_chain"]).shape[-2:]
    save = {
        "step": np.asarray(buf["step"], dtype=np.int64),
        "chain_id": np.asarray(buf["chain_id"], dtype=np.int64),
        "hmc_accept": np.stack(buf["hmc_accept"]).astype(bool),
        "wolff_applied": np.stack(buf["wolff_applied"]).astype(bool),
        "wolff_cluster_size": np.stack(buf["wolff_cluster_size"]).astype(np.int32),
        "log_accept": np.stack(buf["log_accept"]).astype(np.float64),
        "delta_h": np.stack(buf["delta_h"]).astype(np.float64),
        "phi_chain": np.stack(buf["phi_chain"]).astype(phi_dtype, copy=False),
        "phi_hmc_proposal": np.stack(buf["phi_hmc_proposal"]).astype(phi_dtype, copy=False),
        "S_current": np.stack(buf["S_current"]).astype(np.float64),
        "S_hmc_proposal": np.stack(buf["S_hmc_proposal"]).astype(np.float64),
        "dS_dk_current": np.stack(buf["dS_dk_current"]).astype(np.float64),
        "mag": np.stack(buf["mag"]).astype(np.float64),
        "abs_mag": np.stack(buf["abs_mag"]).astype(np.float64),
        "accepted_step": np.asarray(buf["accepted_step"], dtype=np.int64),
        "accepted_chain": np.asarray(buf["accepted_chain"], dtype=np.int64),
        "rejected_step": np.asarray(buf["rejected_step"], dtype=np.int64),
        "rejected_chain": np.asarray(buf["rejected_chain"], dtype=np.int64),
    }
    if buf["phi_hmc_accepted"]:
        save["phi_hmc_accepted"] = np.concatenate(buf["phi_hmc_accepted"], axis=0).astype(phi_dtype, copy=False)
    else:
        save["phi_hmc_accepted"] = np.empty((0,) + tuple(phi_shape), dtype=phi_dtype)
    if buf["phi_hmc_rejected"]:
        save["phi_hmc_rejected"] = np.concatenate(buf["phi_hmc_rejected"], axis=0).astype(phi_dtype, copy=False)
    else:
        save["phi_hmc_rejected"] = np.empty((0,) + tuple(phi_shape), dtype=phi_dtype)

    path = out_dir / "chunks" / f"chunk_{chunk_id:05d}.npz"
    if compress:
        np.savez_compressed(path, **save)
    else:
        np.savez(path, **save)
    for value in buf.values():
        value.clear()
    return str(path)


@torch.no_grad()
def run(args):
    if args.algorithm not in {"hmc", "wolff_hmc"}:
        raise ValueError("--algorithm must be hmc or wolff_hmc")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Lx, Lt = args.Lx, args.Lt
    V = Lx * Lt
    lam = args.lam
    epsilon = args.epsilon
    if epsilon <= 0:
        epsilon = max(0.005, min(0.05, 0.05 * (512 / V) ** 0.25))
    n_leap = max(1, int(round(args.tau / epsilon)))
    epsilon = args.tau / n_leap
    phi_dtype = np.float16 if args.phi_dtype == "float16" else np.float32

    out_dir = Path(args.out_root).resolve()
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis").mkdir(parents=True, exist_ok=True)

    nb = _build_nb_rect(Lx, Lt) if args.algorithm == "wolff_hmc" else None

    if args.init_config and os.path.exists(args.init_config):
        saved = torch.load(args.init_config, map_location="cpu", weights_only=True)
        if saved.ndim == 2:
            saved = saved.unsqueeze(0)
        reps = (args.n_chains + saved.shape[0] - 1) // saved.shape[0]
        phi = saved.repeat(reps, *([1] * (saved.ndim - 1)))[: args.n_chains].to(device=device, dtype=torch.float64)
        init_source = args.init_config
    else:
        phi = torch.randn(args.n_chains, Lx, Lt, dtype=torch.float64, device=device)
        init_source = "random"

    metadata = {
        "script": os.path.basename(__file__),
        "algorithm": args.algorithm,
        "Lx": Lx,
        "Lt": Lt,
        "V": V,
        "kappa": args.kappa,
        "lam": lam,
        "n_chains": args.n_chains,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "measurement_spacing_raw_steps": 1,
        "fixed_step_autocorr": True,
        "tau": args.tau,
        "epsilon": epsilon,
        "n_leapfrog": n_leap,
        "wolff_every": args.wolff_every if args.algorithm == "wolff_hmc" else 0,
        "chunk_size": args.chunk_size,
        "phi_dtype": args.phi_dtype,
        "device": str(device),
        "init_source": init_source,
        "pyerrors": HAS_PE,
        "pyerrors_version": getattr(pe, "__version__", None) if HAS_PE else None,
        "f0_per_site": f0_per_site(lam),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 80, flush=True)
    print(f"{args.algorithm} fixed-step chain: {Lx}x{Lt} V={V} kappa={args.kappa:.4f}", flush=True)
    print(f"steps={args.n_steps}, burn={args.burn_in}, chains={args.n_chains}, spacing=1 raw step", flush=True)
    print(f"epsilon={epsilon:.6f}, n_leapfrog={n_leap}, pyerrors={'yes' if HAS_PE else 'no'}", flush=True)
    print(f"out={out_dir}", flush=True)
    print("=" * 80, flush=True)

    n = args.n_steps
    B = args.n_chains
    acc_ts = np.empty((n, B), dtype=bool)
    wolff_ts = np.zeros((n, B), dtype=bool)
    cluster_ts = np.zeros((n, B), dtype=np.int32)
    mag_ts = np.empty((n, B), dtype=np.float64)
    absmag_ts = np.empty((n, B), dtype=np.float64)
    S_ts = np.empty((n, B), dtype=np.float64)
    dsdk_ts = np.empty((n, B), dtype=np.float64)
    delta_h_ts = np.empty((n, B), dtype=np.float64)
    log_accept_ts = np.empty((n, B), dtype=np.float64)
    step_times = []
    chunk_files = []
    buf = empty_buf()
    chunk_id = 0
    t0 = time.time()

    for step in range(n):
        t_step = time.time()
        phi_after_hmc, phi_prop, accept, _S_old, S_prop, _H_old, _H_prop, delta_h, log_accept, _ = hmc_proposal(
            phi, epsilon, args.tau, lam, args.kappa
        )
        phi = phi_after_hmc

        wolff_applied = np.zeros(B, dtype=bool)
        wolff_sizes = np.zeros(B, dtype=np.int32)
        if args.algorithm == "wolff_hmc" and args.wolff_every > 0 and (step + 1) % args.wolff_every == 0:
            phi, wolff_sizes = wolff_batch_update(phi, nb, args.kappa)
            wolff_applied[:] = True

        if device.type == "cuda":
            torch.cuda.synchronize()

        S_cur = phi4_action(phi, lam, args.kappa)
        dsdk = measure_dsdk(phi)
        mag = phi.sum(dim=(1, 2)) / V
        absmag = mag.abs()

        accept_np = as_np(accept, np.bool_)
        phi_chain_np = as_np(phi, phi_dtype)
        phi_prop_np = as_np(phi_prop, phi_dtype)
        S_cur_np = as_np(S_cur, np.float64)
        S_prop_np = as_np(S_prop, np.float64)
        dsdk_np = as_np(dsdk, np.float64)
        mag_np = as_np(mag, np.float64)
        absmag_np = as_np(absmag, np.float64)
        delta_h_np = as_np(delta_h, np.float64)
        log_accept_np = as_np(log_accept, np.float64)

        acc_ts[step] = accept_np
        wolff_ts[step] = wolff_applied
        cluster_ts[step] = wolff_sizes
        mag_ts[step] = mag_np
        absmag_ts[step] = absmag_np
        S_ts[step] = S_cur_np
        dsdk_ts[step] = dsdk_np
        delta_h_ts[step] = delta_h_np
        log_accept_ts[step] = log_accept_np

        chain_ids = np.arange(B, dtype=np.int64)
        accepted = np.flatnonzero(accept_np)
        rejected = np.flatnonzero(~accept_np)
        buf["step"].append(step)
        buf["chain_id"].append(chain_ids)
        buf["hmc_accept"].append(accept_np)
        buf["wolff_applied"].append(wolff_applied)
        buf["wolff_cluster_size"].append(wolff_sizes)
        buf["log_accept"].append(log_accept_np)
        buf["delta_h"].append(delta_h_np)
        buf["phi_chain"].append(phi_chain_np)
        buf["phi_hmc_proposal"].append(phi_prop_np)
        buf["S_current"].append(S_cur_np)
        buf["S_hmc_proposal"].append(S_prop_np)
        buf["dS_dk_current"].append(dsdk_np)
        buf["mag"].append(mag_np)
        buf["abs_mag"].append(absmag_np)
        if accepted.size:
            buf["accepted_step"].extend([step] * accepted.size)
            buf["accepted_chain"].extend(accepted.tolist())
            buf["phi_hmc_accepted"].append(phi_prop_np[accepted])
        if rejected.size:
            buf["rejected_step"].extend([step] * rejected.size)
            buf["rejected_chain"].extend(rejected.tolist())
            buf["phi_hmc_rejected"].append(phi_prop_np[rejected])

        step_times.append(time.time() - t_step)
        if (step + 1) % args.chunk_size == 0:
            p = flush_chunk(out_dir, chunk_id, buf, phi_dtype, not args.no_compress)
            if p:
                chunk_files.append(p)
                chunk_id += 1

        if (step + 1) % args.print_every == 0 or step + 1 == n:
            print(
                f"step {step+1}/{n} acc={acc_ts[:step+1].mean():.4f} "
                f"|M|={absmag_np.mean():.6f} S/V={S_cur_np.mean()/V:.6f} "
                f"dS/dk/V={dsdk_np.mean()/V:.6f} wall={time.time()-t0:.1f}s",
                flush=True,
            )

    p = flush_chunk(out_dir, chunk_id, buf, phi_dtype, not args.no_compress)
    if p:
        chunk_files.append(p)

    burn = min(max(0, args.burn_in), n - 1)
    flat = lambda arr: np.asarray(arr[burn:]).reshape(-1)
    post = {
        "mag": flat(mag_ts),
        "abs_mag": flat(absmag_ts),
        "S": flat(S_ts),
        "S_per_V": flat(S_ts) / V,
        "dS_dk": flat(dsdk_ts),
        "dS_dk_per_V": flat(dsdk_ts) / V,
        "hmc_accept": flat(acc_ts).astype(np.float64),
        "delta_h": flat(delta_h_ts),
        "log_accept": flat(log_accept_ts),
        "wolff_cluster_fraction": flat(cluster_ts).astype(np.float64) / V,
    }
    stats = {name: gamma_stats(vals, name) for name, vals in post.items()}
    for name in ("S_per_V", "dS_dk_per_V", "abs_mag", "hmc_accept"):
        mean, err, nb = block_jackknife(post[name], np.mean, args.n_jackknife)
        stats[name]["block_jackknife_mean"] = mean
        stats[name]["block_jackknife_err"] = err
        stats[name]["block_jackknife_blocks"] = nb

    lag_len = min(args.max_autocorr_lag, n - burn - 1) + 1
    rho_data = {"lag": np.arange(lag_len, dtype=np.int64)}
    for name in ("abs_mag", "mag", "S_per_V", "dS_dk_per_V", "hmc_accept"):
        rho_data[f"rho_{name}"] = autocorr_fft(post[name], args.max_autocorr_lag)
    np.savez_compressed(out_dir / "analysis" / "autocorr_functions.npz", **rho_data)
    np.savez_compressed(
        out_dir / "analysis" / "scalar_traces.npz",
        hmc_accept=acc_ts,
        wolff_applied=wolff_ts,
        wolff_cluster_size=cluster_ts,
        mag=mag_ts,
        abs_mag=absmag_ts,
        S=S_ts,
        dS_dk=dsdk_ts,
        delta_h=delta_h_ts,
        log_accept=log_accept_ts,
    )

    result = {
        **metadata,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_sec": time.time() - t0,
        "median_step_ms": float(np.median(step_times[max(1, min(20, len(step_times) - 1)) :]) * 1000.0),
        "acceptance_rate": float(flat(acc_ts).mean()),
        "n_chunks": len(chunk_files),
        "chunk_files": chunk_files,
        "stats": stats,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "mean", "err", "tau_int", "dtau_int", "method", "n"])
        for name, item in stats.items():
            writer.writerow(
                [
                    name,
                    item.get("mean", ""),
                    item.get("err", ""),
                    item.get("tau_int", ""),
                    item.get("dtau_int", ""),
                    item.get("method", ""),
                    item.get("n", ""),
                ]
            )

    print("=" * 80, flush=True)
    print(f"DONE {args.algorithm} {Lx}x{Lt} kappa={args.kappa:.4f}", flush=True)
    print(f"tau_int(|M|)={stats['abs_mag']['tau_int']:.4f} +/- {stats['abs_mag']['dtau_int']:.4f}", flush=True)
    print(f"dS/dk/V={stats['dS_dk_per_V']['mean']:.8f} +/- {stats['dS_dk_per_V']['err']:.8f}", flush=True)
    print(f"saved={out_dir}", flush=True)
    print("=" * 80, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("hmc", "wolff_hmc"), required=True)
    parser.add_argument("--Lx", type=int, required=True)
    parser.add_argument("--Lt", type=int, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--lam", type=float, default=LAM_DEFAULT)
    parser.add_argument("--n-chains", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--wolff-every", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-autocorr-lag", type=int, default=2500)
    parser.add_argument("--n-jackknife", type=int, default=20)
    parser.add_argument("--phi-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-config", default="")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
