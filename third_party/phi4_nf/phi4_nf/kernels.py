from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .physics import phi4_action, phi4_force


@dataclass
class Cost:
    entries: dict = field(default_factory=dict)

    def call(self, kind, batch, fn):
        start = time.perf_counter()
        result = fn()
        row = self.entries.setdefault(kind, {"calls": 0, "configurations": 0, "wall_s": 0.0})
        row["calls"] += 1
        row["configurations"] += batch
        row["wall_s"] += time.perf_counter() - start
        return result

    def report(self):
        out = {k: dict(self.entries.get(k, {"calls": 0, "configurations": 0, "wall_s": 0.0}))
               for k in dict.fromkeys(("action", "force", "fft", "ifft", *self.entries))}
        out["fft_total"] = {k: (None if out["fft"][k] is None or out["ifft"][k] is None
                                else out["fft"][k] + out["ifft"][k]) for k in out["fft"]}
        return out

    def structural_fft(self, forward, inverse, batch):
        for kind, calls in (("fft", forward), ("ifft", inverse)):
            row = self.entries.setdefault(kind, {"calls": 0, "configurations": 0, "wall_s": None})
            row["calls"] += calls
            row["configurations"] += calls * batch


def counted(cost, kind, batch, fn):
    return fn() if cost is None else cost.call(kind, batch, fn)


def fft(x, cost=None):
    return counted(cost, "fft", x.shape[0], lambda: torch.fft.fft2(x, norm="ortho"))


def ifft(x, cost=None):
    return counted(cost, "ifft", x.shape[0], lambda: torch.fft.ifft2(x, norm="ortho"))


class IdentityMass:
    def sample(self, noise, cost=None):
        return noise

    def velocity(self, momentum, cost=None):
        return momentum

    def kinetic(self, momentum, cost=None):
        return 0.5 * momentum.square().sum((-2, -1))


class FourierMass:
    def __init__(self, spectrum):
        spectrum = torch.as_tensor(spectrum, dtype=torch.float64, device="cpu").detach().clone()
        if spectrum.ndim != 2 or spectrum.shape[0] != spectrum.shape[1]:
            raise ValueError("mass spectrum must be a square two-dimensional array")
        if not bool(torch.isfinite(spectrum).all() and (spectrum > 0).all()):
            raise ValueError("mass spectrum must be finite and strictly positive")
        indices = (-torch.arange(spectrum.shape[0])) % spectrum.shape[0]
        reflected = spectrum[indices][:, indices]
        if not torch.allclose(spectrum, reflected, rtol=1e-13, atol=1e-15):
            raise ValueError("mass spectrum must satisfy M(k)=M(-k)")
        self._spectrum = spectrum

    @property
    def spectrum(self):
        return self._spectrum.clone()

    def sample(self, noise, cost=None):
        return ifft(fft(noise, cost) * self._spectrum.sqrt(), cost).real

    def velocity(self, momentum, cost=None):
        return ifft(fft(momentum, cost) / self._spectrum, cost).real

    def kinetic(self, momentum, cost=None):
        return 0.5 * (fft(momentum, cost).abs().square() / self._spectrum).sum((-2, -1))


def regulator_mass(L, kappa, regulator):
    if L < 2 or kappa < 0 or not math.isfinite(kappa) or regulator <= 0 or not math.isfinite(regulator):
        raise ValueError("positive regulator and nonnegative finite kappa required")
    frequencies = torch.fft.fftfreq(L, dtype=torch.float64)
    laplacian = 8.0 * kappa * (torch.sin(math.pi * frequencies).square()[:, None]
                                + torch.sin(math.pi * frequencies).square()[None, :])
    mass = regulator + laplacian
    return mass / mass.mean()


def covariance_mass(configs, floor=1e-6, cost=None):
    if floor <= 0 or not math.isfinite(floor):
        raise ValueError("covariance floor must be positive")
    flat = torch.as_tensor(configs, dtype=torch.float64, device="cpu").flatten(0, 1)
    if flat.shape[0] < 4 or not bool(torch.isfinite(flat).all()):
        raise ValueError("pilot covariance needs at least four finite configurations")
    covariance = fft(flat, cost).abs().square().mean(0)
    indices = (-torch.arange(flat.shape[-1])) % flat.shape[-1]
    covariance = 0.5 * (covariance + covariance[indices][:, indices])
    if not bool(torch.isfinite(covariance).all()) or float(covariance.mean()) <= 0:
        raise ValueError("pilot covariance is degenerate")
    cutoff = max(floor * float(covariance.mean()), np.finfo(np.float64).tiny)
    precision = covariance.clamp_min(cutoff).reciprocal()
    return precision / precision.mean(), covariance, cutoff


def action(phi, lam, kappa, cost=None):
    return counted(cost, "action", phi.shape[0], lambda: phi4_action(phi, lam=lam, kappa=kappa))


def force(phi, lam, kappa, cost=None):
    return counted(cost, "force", phi.shape[0], lambda: phi4_force(phi, lam=lam, kappa=kappa))


def leapfrog(phi, momentum, mass, epsilon, n_leap, lam, kappa, cost=None):
    if epsilon <= 0 or not math.isfinite(epsilon) or n_leap < 1 or int(n_leap) != n_leap:
        raise ValueError("positive finite step size and positive integer leapfrog count required")
    q = phi + (0.5 * epsilon) * mass.velocity(momentum, cost)
    p = momentum.clone()
    for _ in range(n_leap - 1):
        p = p + epsilon * force(q, lam, kappa, cost)
        q = q + epsilon * mass.velocity(p, cost)
    p = p + epsilon * force(q, lam, kappa, cost)
    q = q + (0.5 * epsilon) * mass.velocity(p, cost)
    return q, p


def chain_generators(seeds):
    return [torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seeds]


def random_inputs(phi, generators=None):
    if generators is None:
        return torch.randn_like(phi), torch.rand(phi.shape[0], dtype=phi.dtype, device=phi.device)
    if len(generators) != phi.shape[0]:
        raise ValueError("one CPU generator per chain required")
    noise = torch.stack([torch.randn(phi.shape[1:], dtype=phi.dtype, generator=g) for g in generators])
    uniform = torch.stack([torch.rand((), dtype=phi.dtype, generator=g) for g in generators])
    return noise, uniform


@torch.no_grad()
def hmc_step(phi, mass, epsilon, n_leap, lam=0.022, kappa=0.27, cost=None,
             generators=None, noise=None, uniform=None):
    if phi.device.type != "cpu" or phi.dtype != torch.float64:
        raise ValueError("this implementation requires float64 CPU fields")
    if noise is None and uniform is None:
        noise, uniform = random_inputs(phi, generators)
    if noise is None or uniform is None or noise.shape != phi.shape or uniform.shape != phi.shape[:1]:
        raise ValueError("momentum noise and uniforms must both be supplied with matching shapes")
    if not bool(torch.isfinite(noise).all() and torch.isfinite(uniform).all()
                and (uniform >= 0).all() and (uniform < 1).all()):
        raise ValueError("invalid random inputs")
    momentum = mass.sample(noise, cost)
    h_old = action(phi, lam, kappa, cost) + mass.kinetic(momentum, cost)
    if not bool(torch.isfinite(h_old).all()):
        raise ValueError("initial Hamiltonian is not finite")
    q, p = leapfrog(phi, momentum, mass, epsilon, n_leap, lam, kappa, cost)
    h_new = action(q, lam, kappa, cost) + mass.kinetic(p, cost)
    delta_h = h_new - h_old
    finite = (torch.isfinite(delta_h) & torch.isfinite(q).all((-2, -1))
              & torch.isfinite(p).all((-2, -1)))
    probability = torch.where(finite, torch.exp(-delta_h.clamp_min(0)), torch.zeros_like(delta_h))
    accepted = finite & (uniform.log() < -delta_h)
    return torch.where(accepted[:, None, None], q, phi), {
        "accepted": accepted, "accept_probability": probability, "delta_h": delta_h,
        "finite_proposal": finite, "h_old": h_old, "h_new": h_new,
    }
