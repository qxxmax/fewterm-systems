from __future__ import annotations

import math

import numpy as np
import torch

from .kernels import action, fft, ifft

KEYS = ("M", "absM", "M2", "M4", "phi2", "phi4", "energy", "action_density", "Smin", "Gsub2")
EST_KEYS = KEYS + ("U4", "chi", "chi_connected")


@torch.no_grad()
def observables(phi, lam, kappa, cost=None):
    L = phi.shape[-1]
    if L < 4 or phi.shape[-2] != L or not bool(torch.isfinite(phi).all()):
        raise ValueError("observables require finite square fields with L >= 4")
    volume = L * L
    m = phi.mean((-2, -1))
    power = fft(phi, cost).abs().square()
    freq = torch.fft.fftfreq(L, d=1.0 / L, dtype=torch.float64)
    qmap = freq.square()[:, None] + freq.square()[None, :]
    smin = volume * power[:, qmap == 1].mean(1)
    centered_power = power.clone()
    centered_power[:, 0, 0] = 0
    corr = ifft(centered_power, cost).real / math.sqrt(volume)
    xs = torch.minimum(torch.arange(L), L - torch.arange(L)).double()
    distance = (xs.square()[:, None] + xs.square()[None, :]).sqrt()
    gsub2 = corr[:, (distance - 2).abs() < 0.05].mean(1)
    energy = (phi * phi.roll(1, -1) + phi * phi.roll(1, -2)).mean((-2, -1))
    values = (m, m.abs(), m.square(), m.pow(4), phi.square().mean((-2, -1)),
              phi.pow(4).mean((-2, -1)), energy, action(phi, lam, kappa, cost) / volume, smin, gsub2)
    out = {key: value.numpy().copy() for key, value in zip(KEYS, values)}
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("nonfinite observable; configurations are never filtered")
    return out


def tau_details(series, c=6.0):
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or x.size == 0:
        raise ValueError("tau input must be a finite nonempty single-chain series")
    n = x.size
    constant = bool(np.all(x == x[0]))
    x = x - x.mean()
    variance = float(np.mean(x * x))
    if constant or variance <= 0 or n < 4:
        return {"tau_int": 0.5, "window": 0, "window_closed": False,
                "constant": constant or variance <= 0, "ess_chain": 0.0 if constant or variance <= 0 else float(n), "reliable": False}
    nfft = 1 << (2 * n - 1).bit_length()
    ft = np.fft.rfft(x, n=nfft)
    covariance = np.fft.irfft(ft * ft.conj(), n=nfft)[:n] / np.arange(n, 0, -1)
    tau, lag, closed = 0.5, 0, False
    for lag in range(1, n // 2):
        tau += float(covariance[lag] / variance)
        if lag >= c * tau:
            closed = True
            break
    tau = max(tau, 0.5)
    return {"tau_int": float(tau), "window": lag, "window_closed": closed,
            "constant": False, "ess_chain": float(n / (2 * tau)),
            "reliable": bool(closed and n >= 50 * tau)}


def tau_int(series, c=6.0):
    return tau_details(series, c)["tau_int"]


def estimates(means, volume):
    out = dict(zip(KEYS, map(float, means)))
    m2, m4 = out["M2"], out["M4"]
    out["U4"] = 1 - m4 / (3 * m2 * m2) if m2 > 0 else float("nan")
    out["chi"] = volume * (m2 - out["absM"] ** 2)
    out["chi_connected"] = volume * (m2 - out["M"] ** 2)
    return out


def influence_series(features, volume):
    out = dict(features)
    m2, m4 = float(features["M2"].mean()), float(features["M4"].mean())
    if m2 > 0:
        out["U4"] = (2 * m4 / (3 * m2 ** 3) * (features["M2"] - m2)
                     - (features["M4"] - m4) / (3 * m2 ** 2))
    else:
        out["U4"] = np.zeros_like(features["M"])
    out["chi"] = volume * (features["M2"] - m2
                            - 2 * features["absM"].mean() * (features["absM"] - features["absM"].mean()))
    out["chi_connected"] = volume * (features["M2"] - m2
                                     - 2 * features["M"].mean() * (features["M"] - features["M"].mean()))
    return out


def split_rhat(series):
    n, chains = series.shape
    half = n // 2
    if chains < 2 or half < 4:
        return None
    x = np.concatenate((series[:half], series[-half:]), axis=1)
    within = float(x.var(axis=0, ddof=1).mean())
    between = half * float(x.mean(axis=0).var(ddof=1))
    if within <= 0:
        return float("inf") if between > 0 else None
    return math.sqrt(((half - 1) / half * within + between / half) / within)


def circular_indices(n, block, rng):
    starts = rng.integers(0, n, size=math.ceil(n / block))
    return ((starts[:, None] + np.arange(block)[None, :]) % n).reshape(-1)[:n]


def bootstrap_stats(features, volume, n_boot, seed, block_sizes):
    data = np.stack([features[key] for key in KEYS], axis=-1)
    if data.ndim != 3 or not np.isfinite(data).all() or n_boot < 2:
        raise ValueError("bootstrap needs finite [step, chain] features and at least two replicates")
    n, chains, _ = data.shape
    if len(block_sizes) != chains or any(b < 1 or b > n for b in block_sizes):
        raise ValueError("one block size in [1, steps] per chain required")
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, len(EST_KEYS)))
    chain_reps = np.empty((n_boot, chains, len(EST_KEYS)))
    for rep in range(n_boot):
        means = np.stack([data[circular_indices(n, int(block_sizes[ch]), rng), ch].mean(0)
                          for ch in range(chains)])
        reps[rep] = list(estimates(means.mean(0), volume).values())
        chain_reps[rep] = [list(estimates(row, volume).values()) for row in means]
    return {
        "estimate": estimates(data.mean((0, 1)), volume),
        "bootstrap_se": dict(zip(EST_KEYS, reps.std(axis=0, ddof=1).tolist())),
        "per_chain_estimate": [estimates(row, volume) for row in data.mean(0)],
        "per_chain_bootstrap_se": [dict(zip(EST_KEYS, row)) for row in chain_reps.std(axis=0, ddof=1).tolist()],
        "replicates": reps, "chain_replicates": chain_reps,
        "undefined_replicates": {key: int((~np.isfinite(reps[:, i])).sum()) for i, key in enumerate(EST_KEYS)},
    }


def summarize(features, volume, n_boot, seed, block_size=None):
    shape = features["M"].shape
    if len(shape) != 2 or not all(features[k].shape == shape and np.isfinite(features[k]).all() for k in KEYS):
        raise ValueError("finite [step, chain] arrays required for every observable")
    n, chains = shape
    influence = influence_series(features, volume)
    autocorrelation = {key: [tau_details(series[:, ch]) for ch in range(chains)]
                       for key, series in influence.items()}
    requested = ([max(1, math.ceil(2 * max(autocorrelation[key][ch]["tau_int"] for key in EST_KEYS)))
                  for ch in range(chains)] if block_size is None else [int(block_size)] * chains)
    if any(b < 1 for b in requested):
        raise ValueError("bootstrap block length must be positive")
    blocks = [min(n, b) for b in requested]
    boot = bootstrap_stats(features, volume, n_boot, seed, blocks)
    metrics = {}
    for key, details in autocorrelation.items():
        metrics[key] = {"per_chain": details, "ess_total": sum(d["ess_chain"] for d in details),
                        "split_rhat": split_rhat(influence[key]),
                        "reliable": all(d["reliable"] for d in details)}
    return {
        "n_steps": n, "n_chains": chains, "n_configurations": n * chains,
        "estimates": {key: {"mean": boot["estimate"][key], "bootstrap_se": boot["bootstrap_se"][key]}
                      for key in EST_KEYS},
        "per_chain_estimate": boot["per_chain_estimate"],
        "per_chain_bootstrap_se": boot["per_chain_bootstrap_se"],
        "autocorrelation": metrics,
        "bootstrap": {"seed": seed, "replicates": n_boot, "block_sizes": blocks,
                      "requested_block_sizes": requested, "blocks_per_chain": [n / b for b in blocks],
                      "method": "circular moving-block resampling within each independent chain; joint moments; no chain concatenation",
                      "undefined_replicates": boot["undefined_replicates"]},
        "warnings": ["finite-chain tau and block bootstrap do not establish equilibration or cross-well mixing"]
                    + (["at least one bootstrap chain has fewer than eight blocks"] if any(n / b < 8 for b in blocks) else [])
                    + (["at least one autocorrelation estimate is constant, window-truncated, or shorter than 50 tau"]
                       if not all(m["reliable"] for m in metrics.values()) else []),
    }, {"bootstrap_estimates": boot["replicates"], "bootstrap_chain_estimates": boot["chain_replicates"]}
