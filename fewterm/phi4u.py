"""Few-parameter flow of the paper, without coefficient bounds.

UnboundedCNF has the 14 zero-mode, local and Fourier terms, CoupledCNF adds the two nearest-neighbour terms
(16 terms).  Substeps of the Strang splitting, in the names used below:
  A   zero-mode substep, an ODE for y = sum_x phi_x / sqrt V
  B   local substep, a 1D ODE dphi_x/dt = g(phi_x) at every site
  C   Fourier step, exact exponential of the rates on the nonzero modes
  D0, D1   nearest-neighbour substeps on the two checkerboard colours (other colour fixed)
A, B and D are integrated with the implicit midpoint rule in microsteps chosen such that |theta| =
|h_micro g'| / 2 <= micro_u = 1/4, using a bound on g' that depends only on the coefficients and the fixed sites.
Each microstep is then a monotone 1D map with log-derivative log((1+theta)/(1-theta)), so the map is a bijection
and the log-Jacobian is exact.  The inverse runs the steps in reverse order with -h.
"""
import math

import numpy as np
import torch

from .phi4 import DT, ExtCNF, OperatorCNF, axes, action

NODES = 17


class UnboundedCNF(ExtCNF):
    def __init__(self, L, coeff, sphi, sm, d=2, device='cpu', micro_u=.25):
        super().__init__(L, coeff, sphi, sm, 'local', d=d, device=device)
        self.micro_u = micro_u

    def part_bound(self, c, part):
        if part == 'A':
            return float(abs(c[3]) + abs(c[4]) * 9 / 8 * self.sm ** 2 + abs(c[5]) * 25 / 16 * self.sm ** 4)
        return self.local_bound(c)

    def substep(self, x, tau, c, part):
        if part == 'C':
            return ExtCNF.substep(self, x, tau, c, part)
        n = max(1, math.ceil(abs(tau) * self.part_bound(c, part) / (2 * self.micro_u)))
        ld = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        for _ in range(n):
            x, l = ExtCNF.substep(self, x, tau / n, c, part)
            ld = ld + l
        return x, ld

    def transform(self, x, steps, inverse=False):
        dt = (-1 if inverse else 1) / steps
        ld = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        for j in (reversed(range(steps)) if inverse else range(steps)):
            c = self.at((j + .5) / steps)
            for part, frac in (('A', .5), ('B', .5), ('C', 1), ('B', .5), ('A', .5)):
                x, ch = self.substep(x, dt * frac, c, part)
                ld = ld + ch
        if not bool(torch.isfinite(x).all() and torch.isfinite(ld).all()):
            raise RuntimeError('nonfinite map')
        return x, ld

    def micro_counts(self, steps):
        """Micro steps per outer step for A and B (max over nodes), for cost reporting."""
        out = []
        for j in range(steps):
            c = self.at((j + .5) / steps)
            out.append([max(1, math.ceil(.5 / steps * self.part_bound(c, p) / (2 * self.micro_u))) for p in 'AB'])
        return np.array(out).max(0).tolist()

    def features(self, x):
        """[B, *lat, 14] in coefficient order (basis[:8], radial 4, local 2)."""
        return torch.cat([self.basis(x), self.extra_basis(x), self.new_basis(x)], -1)


@torch.no_grad()
def base_noise(shape, g, base=None):
    """Base-field draw: standard Gaussian, or base.sample (e.g. fewterm.landau.LandauBase)."""
    return torch.randn(shape, dtype=DT, generator=g) if base is None else base.sample(shape, g)


def zm_ot_pair(z, fields):
    """Monotone (1D optimal-transport) pairing of zero modes: the sorted base zero modes are reassigned to the
    configurations in the rank order of the target zero modes.  The multiset of base zero modes, hence the base
    marginal, is unchanged; nonzero modes stay independent."""
    d = fields.dim() - 1
    ax = tuple(range(1, d + 1))
    rt = math.sqrt(fields[0].numel())
    yz, yx = z.sum(ax) / rt, fields.sum(ax) / rt
    ynew = torch.empty_like(yz)
    ynew[torch.argsort(yx)] = torch.sort(yz).values
    return z + ((ynew - yz) / rt).reshape((-1,) + (1,) * d)


def fit(fields, seed, d=None, ridges=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1), chunk=64, base=None, zm_ot=False):
    """Per-node ridge LSQ to the conditional target u = phi_1 - z on x_t = (1-t) z + t phi_1 (3/4 train,
    1/4 validation, ridge by validation MSE); nodes 0 and 16 fixed to the exact values (psi and m coefficients
    -1 resp. +1).  Scales s_phi, s_m as in phi4.initial_fit (path rms of phi and of the zero mode)."""
    d = fields.dim() - 1 if d is None else d
    L = fields.shape[-1]
    ax = axes(d)
    g = torch.Generator().manual_seed(seed)
    n = len(fields)
    perm = torch.randperm(n, generator=g)
    tr, va = perm[:3 * n // 4], perm[3 * n // 4:]
    z = base_noise(fields.shape, g, base)
    if zm_ot:
        z = zm_ot_pair(z, fields)
    ts = np.linspace(0, 1, NODES)
    w = np.ones(NODES); w[[0, -1]] = .5; w /= w.sum()
    sphi = math.sqrt(sum(wt * float(((1 - t) * z + t * fields).square().mean()) for t, wt in zip(ts, w)))
    sm = math.sqrt(sum(wt * float(((1 - t) * z + t * fields).mean(ax).square().mean()) for t, wt in zip(ts, w)))
    K = 14
    model = UnboundedCNF(L, np.zeros((NODES, K)), sphi, sm, d)
    coeff, rows = np.zeros((NODES, K)), []
    for j, t in enumerate(ts):
        if j in (0, NODES - 1):
            coeff[j, [0, 3]] = -1. if j == 0 else 1.
            rows.append({'t': float(t), 'fixed': True})
            continue
        AtA, Aty = np.zeros((K, K)), np.zeros(K)
        for a in range(0, len(tr), chunk):
            idx = tr[a:a + chunk]
            xt = (1 - t) * z[idx] + t * fields[idx]
            F = model.features(xt).reshape(-1, K).numpy()
            AtA += F.T @ F
            Aty += F.T @ (fields[idx] - z[idx]).reshape(-1).numpy()
        xv = (1 - t) * z[va] + t * fields[va]
        Fv = model.features(xv).reshape(-1, K).numpy()
        yv = (fields[va] - z[va]).reshape(-1).numpy()
        col = np.sqrt(np.diag(AtA)).clip(1e-300)
        Q, b = AtA / np.outer(col, col), Aty / col
        best = None
        for lam in ridges:
            cc = np.linalg.solve(Q + lam * np.eye(K), b) / col
            mse = float(np.mean((Fv @ cc - yv) ** 2))
            if best is None or mse < best[1]:
                best = (cc, mse, lam)
        coeff[j] = best[0]
        rows.append({'t': float(t), 'ridge': best[2], 'val_rel_mse': best[1] / float(np.mean(yv ** 2))})
    model.coeff = torch.as_tensor(coeff, dtype=DT)
    return model, {'sphi': sphi, 'sm': sm, 'rows': rows}


def exact_site_samples(n, L, d, lam, seed, grid=200001, lim=8.):
    """kappa = 0: independent sites with density prop. to exp(-(1-2 lam) phi^2 - lam phi^4), inverse-CDF on a grid."""
    u = np.linspace(-lim, lim, grid)
    p = np.exp(-(1 - 2 * lam) * u ** 2 - lam * u ** 4)
    cdf = np.cumsum(p); cdf /= cdf[-1]
    r = np.random.default_rng(seed).random((n,) + (L,) * d)
    return torch.as_tensor(np.interp(r, cdf, u), dtype=DT)


@torch.no_grad()
def ess_of(model, steps, n, seed, lam, kappa, batch=512):
    L, d, ax = model.L, model.d, model.ax
    g = torch.Generator().manual_seed(seed)
    lw = []
    for off in range(0, n, batch):
        z = base_noise((min(batch, n - off),) + (L,) * d, g, getattr(model, 'base', None))
        x, ld = model.transform(z, steps)
        lw.append((-action(x, lam, kappa) - (model.logp(z) - ld)).numpy())
    lw = np.concatenate(lw)
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (len(w) * (w * w).sum())), lw


# ----------------------------------------------------------------------------- two-site nonlinear couplings
def neighbor_sums(x, d):
    """n1 = sum_nn phi_y, n2 = sum_nn phi_y^2 over the 2d nearest neighbours (last d axes)."""
    n1 = n2 = 0
    for ax in range(-d, 0):
        for s in (1, -1):
            y = x.roll(s, ax)
            n1, n2 = n1 + y, n2 + y * y
    return n1, n2


def gsat(u, s):
    """g(u) = u^2 / (1 + u^2/s^2) (even, bounded by s^2) and g'(u)."""
    w = (u / s).square()
    return u * u / (1 + w), 2 * u / (1 + w).square()


class CoupledCNF(UnboundedCNF):
    """UnboundedCNF + two gradient-form nearest-neighbour couplings, updated on the checkerboard (neighbours frozen):
        c14: d/dphi_x sum_<xy> phi_x^2 phi_y^2 = 2 phi_x n2_x
        c15: d/dphi_x sum_<xy> G(phi_x) phi_y  = 3 g(phi_x) n1_x,  g = phi^2/(1+phi^2/s^2), s = s_phi
    Substep D_a (colour a): per site 1D ODE dphi/dt = 2 c14 n2 phi + 3 c15 n1 g(phi), sub-cycled implicit midpoint
    with a per-sample micro count from that sample's frozen neighbours (block-triangular bijection, exact log-det).
    Strang step: A/2 B/2 De/2 Do/2 C Do/2 De/2 B/2 A/2."""
    K = 16

    def __init__(self, L, coeff, sphi, sm, d=2, device='cpu', micro_u=.25):
        c = np.asarray(coeff, dtype=np.float64)
        if c.shape != (NODES, self.K) or L % 2:
            raise ValueError('expected (17, 16) coefficients and even L')
        super().__init__(L, c[:, :14], sphi, sm, d, device, micro_u)
        self.full = torch.as_tensor(c, dtype=DT)
        idx = torch.stack(torch.meshgrid(*[torch.arange(L)] * d, indexing='ij')).sum(0) % 2
        self.colour = [idx == 0, idx == 1]
        self.gmax = .65 * self.sphi            # sup |g'| = (3 sqrt 3 / 8) s ~ 0.6495 s

    @property
    def coeff_full(self):
        return self.full

    def at(self, t):
        pos = float(t) * (NODES - 1)
        lo = min(NODES - 2, int(pos))
        w = pos - lo
        return (1 - w) * self.full[lo] + w * self.full[lo + 1]

    def features(self, x):
        n1, n2 = neighbor_sums(x, self.d)
        g, _ = gsat(x, self.sphi)
        return torch.cat([self.basis(x), self.extra_basis(x), self.new_basis(x),
                          torch.stack([2 * x * n2, 3 * g * n1], -1)], -1)

    def velocity(self, x, t):
        return (self.features(x) * self.at(t)).sum(-1)

    def coupling_substep(self, x, tau, c, mask):
        n1, n2 = neighbor_sums(x, self.d)          # frozen for the sites of this colour
        a = 2 * c[14] * n2
        b = 3 * c[15] * n1
        lat = tuple(range(1, x.dim()))
        bound = ((a.abs() + b.abs() * self.gmax) * mask).amax(lat)            # per sample
        nmic = torch.clamp(torch.ceil(abs(tau) * bound / (2 * self.micro_u)), min=1).long()
        ld = torch.zeros(len(x), dtype=x.dtype)
        y = x.clone()
        m = mask.to(x.dtype)
        for k in range(int(nmic.max())):
            active = (k < nmic).to(x.dtype).reshape((-1,) + (1,) * self.d)
            h = (tau / nmic.to(x.dtype)).reshape((-1,) + (1,) * self.d)

            def fun(u):
                gv, gd = gsat(u, self.sphi)
                return a * u + b * gv, a + b * gd
            # implicit midpoint per site: mid = y + h/2 f(mid)
            mid = y.clone()
            for _ in range(50):
                v, dv = fun(mid)
                res = mid - y - .5 * h * v
                mid = mid - res / (1 - .5 * h * dv)
                if float(torch.where(m * active > 0, res.abs(), torch.zeros_like(res)).max()) < 1e-13:
                    break
            v, dv = fun(mid)
            u = .5 * h * dv
            sel = m * active
            if float(torch.where(sel > 0, u.abs(), torch.zeros_like(u)).max()) >= .999:
                raise RuntimeError('micro step not invertible')
            ynew = 2 * mid - y
            us = torch.where(sel > 0, u, torch.zeros_like(u))          # unselected sites may have |u| > 1: never log them
            ld = ld + (torch.log1p(us) - torch.log1p(-us)).sum(lat)
            y = torch.where(sel > 0, ynew, y)
        return y, ld

    def transform(self, x, steps, inverse=False):
        dt = (-1 if inverse else 1) / steps
        ld = torch.zeros(len(x), dtype=x.dtype)
        seq = (('A', .5), ('B', .5), ('D0', .5), ('D1', .5), ('C', 1), ('D1', .5), ('D0', .5), ('B', .5), ('A', .5))
        for j in (reversed(range(steps)) if inverse else range(steps)):
            c = self.at((j + .5) / steps)
            for part, frac in seq:
                if part.startswith('D'):
                    x, ch = self.coupling_substep(x, dt * frac, c, self.colour[int(part[1])])
                else:
                    x, ch = self.substep(x, dt * frac, c, part)
                ld = ld + ch
        if not bool(torch.isfinite(x).all() and torch.isfinite(ld).all()):
            raise RuntimeError('nonfinite map')
        return x, ld


@torch.no_grad()
def fit_coupled(fields, seed, ridges=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1), chunk=64, base=None, zm_ot=False):
    """Fit of the 16-term flow used in the paper: ridge least squares to the flow-matching target u = phi_1 - z on
    x_t = (1-t) z + t phi_1 at the 15 interior nodes.  base: zero-mode base distribution
    (fewterm.landau.LandauBase); zm_ot: pair the zero modes by sorting (zm_ot_pair)."""
    # rows[j]['val_res_zero_frac']: share of the validation residual power in the zero mode (1/V if uniform).
    _, info = fit(fields, seed, ridges=ridges, chunk=chunk, base=base, zm_ot=zm_ot)       # only for the scales s_phi, s_m; the 16 terms are fitted below
    d = fields.dim() - 1
    L = fields.shape[-1]
    g = torch.Generator().manual_seed(seed)
    n = len(fields)
    perm = torch.randperm(n, generator=g)
    tr, va = perm[:3 * n // 4], perm[3 * n // 4:]
    z = base_noise(fields.shape, g, base)
    if zm_ot:
        z = zm_ot_pair(z, fields)
    K = CoupledCNF.K
    model = CoupledCNF(L, np.zeros((NODES, K)), info['sphi'], info['sm'], d)
    coeff, rows = np.zeros((NODES, K)), []
    for j, t in enumerate(np.linspace(0, 1, NODES)):
        if j in (0, NODES - 1):
            coeff[j, [0, 3]] = -1. if j == 0 else 1.
            rows.append({'t': float(t), 'fixed': True})
            continue
        AtA, Aty = np.zeros((K, K)), np.zeros(K)
        for a in range(0, len(tr), chunk):
            idx = tr[a:a + chunk]
            F = model.features((1 - t) * z[idx] + t * fields[idx]).reshape(-1, K).numpy()
            AtA += F.T @ F
            Aty += F.T @ (fields[idx] - z[idx]).reshape(-1).numpy()
        Fv = model.features((1 - t) * z[va] + t * fields[va]).reshape(-1, K).numpy()
        yv = (fields[va] - z[va]).reshape(-1).numpy()
        col = np.sqrt(np.diag(AtA)).clip(1e-300)
        Q, b = AtA / np.outer(col, col), Aty / col
        best = None
        for lam in ridges:
            cc = np.linalg.solve(Q + lam * np.eye(K), b) / col
            mse = float(np.mean((Fv @ cc - yv) ** 2))
            if best is None or mse < best[1]:
                best = (cc, mse, lam)
        coeff[j] = best[0]
        res = (Fv @ best[0] - yv).reshape(len(va), -1)
        rows.append({'t': float(t), 'ridge': best[2], 'val_rel_mse': best[1] / float(np.mean(yv ** 2)),
                     'val_res_zero_frac': float(res.shape[1] * np.mean(res.mean(1) ** 2) / np.mean(res ** 2))})
    model.full = torch.as_tensor(coeff, dtype=DT)
    model.coeff = model.full[:, :14].clone()
    if base is not None:
        model.base, model.logp = base, base.logp
    return model, {'sphi': info['sphi'], 'sm': info['sm'], 'rows': rows}


class CoupledLRCNF(CoupledCNF):
    """CoupledCNF + masked long-range nonlinear couplings (checkerboard substep, other colour frozen):
        c16: phi_x (K * phi^2 1_other)_x,   c17: g(phi_x) (K * phi 1_other)_x,   K^ = s/(s + khat^2), s = lr_scale
    The site ODE stays dphi/dt = a_x phi + b_x g(phi) with a_x, b_x from frozen sites only (exact sub-cycled midpoint)."""
    K = 18

    def __init__(self, L, coeff, sphi, sm, d=2, device='cpu', micro_u=.25, lr_scale=.25):
        c = np.asarray(coeff, dtype=np.float64)
        if c.shape != (NODES, self.K) or L % 2:
            raise ValueError('expected (17, 18) coefficients and even L')
        UnboundedCNF.__init__(self, L, c[:, :14], sphi, sm, d, device, micro_u)
        self.full = torch.as_tensor(c, dtype=DT)
        idx = torch.stack(torch.meshgrid(*[torch.arange(L)] * d, indexing='ij')).sum(0) % 2
        self.colour = [idx == 0, idx == 1]
        self.gmax = .65 * self.sphi
        self.lr_scale = lr_scale
        self.Khat = lr_scale / (lr_scale + self.k2)

    def _conv(self, f):
        return torch.fft.ifftn(torch.fft.fftn(f, dim=self.ax) * self.Khat, dim=self.ax).real

    def lr_fields(self, x, colour_mask):
        other = (~colour_mask).to(x.dtype)
        return self._conv(x * x * other), self._conv(x * other)

    def features(self, x):
        base = CoupledCNF.features(self, x)
        f16 = torch.zeros_like(x)
        f17 = torch.zeros_like(x)
        g, _ = gsat(x, self.sphi)
        for cm in self.colour:
            q2, q1 = self.lr_fields(x, cm)
            f16 = torch.where(cm, x * q2, f16)
            f17 = torch.where(cm, g * q1, f17)
        return torch.cat([base, torch.stack([f16, f17], -1)], -1)

    def coupling_substep(self, x, tau, c, mask):
        n1, n2 = neighbor_sums(x, self.d)
        q2, q1 = self.lr_fields(x, mask)
        a = 2 * c[14] * n2 + c[16] * q2
        b = 3 * c[15] * n1 + c[17] * q1
        lat = tuple(range(1, x.dim()))
        bound = ((a.abs() + b.abs() * self.gmax) * mask).amax(lat)
        nmic = torch.clamp(torch.ceil(abs(tau) * bound / (2 * self.micro_u)), min=1).long()
        ld = torch.zeros(len(x), dtype=x.dtype)
        y = x.clone()
        m = mask.to(x.dtype)
        for k in range(int(nmic.max())):
            active = (k < nmic).to(x.dtype).reshape((-1,) + (1,) * self.d)
            h = (tau / nmic.to(x.dtype)).reshape((-1,) + (1,) * self.d)

            def fun(u):
                gv, gd = gsat(u, self.sphi)
                return a * u + b * gv, a + b * gd
            mid = y.clone()
            for _ in range(50):
                v, dv = fun(mid)
                res = mid - y - .5 * h * v
                mid = mid - res / (1 - .5 * h * dv)
                if float(torch.where(m * active > 0, res.abs(), torch.zeros_like(res)).max()) < 1e-13:
                    break
            v, dv = fun(mid)
            u = .5 * h * dv
            sel = m * active
            if float(torch.where(sel > 0, u.abs(), torch.zeros_like(u)).max()) >= .999:
                raise RuntimeError('micro step not invertible')
            us = torch.where(sel > 0, u, torch.zeros_like(u))
            ld = ld + (torch.log1p(us) - torch.log1p(-us)).sum(lat)
            y = torch.where(sel > 0, 2 * mid - y, y)
        return y, ld


@torch.no_grad()
def fit_generic(fields, seed, cls, ridges=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1), chunk=64, base=None, zm_ot=False,
                **kw):
    """Per-node ridge LSQ for any phi4u model class with .features (K = cls.K); nodes 0, 16 exact."""
    _, info = fit(fields, seed, ridges=ridges, chunk=chunk, base=base, zm_ot=zm_ot)
    d = fields.dim() - 1
    L = fields.shape[-1]
    g = torch.Generator().manual_seed(seed)
    n = len(fields)
    perm = torch.randperm(n, generator=g)
    tr, va = perm[:3 * n // 4], perm[3 * n // 4:]
    z = base_noise(fields.shape, g, base)
    if zm_ot:
        z = zm_ot_pair(z, fields)
    K = cls.K
    model = cls(L, np.zeros((NODES, K)), info['sphi'], info['sm'], d, **kw)
    coeff, rows = np.zeros((NODES, K)), []
    for j, t in enumerate(np.linspace(0, 1, NODES)):
        if j in (0, NODES - 1):
            coeff[j, [0, 3]] = -1. if j == 0 else 1.
            rows.append({'t': float(t), 'fixed': True})
            continue
        AtA, Aty = np.zeros((K, K)), np.zeros(K)
        for a in range(0, len(tr), chunk):
            idx = tr[a:a + chunk]
            F = model.features((1 - t) * z[idx] + t * fields[idx]).reshape(-1, K).numpy()
            AtA += F.T @ F
            Aty += F.T @ (fields[idx] - z[idx]).reshape(-1).numpy()
        Fv = model.features((1 - t) * z[va] + t * fields[va]).reshape(-1, K).numpy()
        yv = (fields[va] - z[va]).reshape(-1).numpy()
        col = np.sqrt(np.diag(AtA)).clip(1e-300)
        Q, b = AtA / np.outer(col, col), Aty / col
        best = None
        for lam in ridges:
            cc = np.linalg.solve(Q + lam * np.eye(K), b) / col
            mse = float(np.mean((Fv @ cc - yv) ** 2))
            if best is None or mse < best[1]:
                best = (cc, mse, lam)
        coeff[j] = best[0]
        rows.append({'t': float(t), 'ridge': best[2], 'val_rel_mse': best[1] / float(np.mean(yv ** 2))})
    model.full = torch.as_tensor(coeff, dtype=DT)
    model.coeff = model.full[:, :14].clone()
    if base is not None:
        model.base, model.logp = base, base.logp
    return model, {'sphi': info['sphi'], 'sm': info['sm'], 'rows': rows}
