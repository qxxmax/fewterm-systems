"""The analytic Strang-split map (A/2 B/2 C B/2 A/2 per step, coefficients frozen at the step midpoint)
integrates exactly the real-space velocity v(x, t) = model.velocity(x, t) that the fit regresses on.

Reference: classical RK4 on dx/dt = model.velocity(x, t) with 256 steps.  The coefficients are piecewise linear
on 9 or 17 time nodes; every step count used is a multiple of 16, so each step stays inside one linear piece and
both integrators keep their nominal order.  RK4 error ~ h^4 (512 vs 1024 RK4 steps differ by < 6e-11 here, so
256 steps are good to ~1e-9; checked when the test was written), while the splitting error is O(dt^2) with measured constants
err(N) * N^2 in [0.009, 2.5] for these random models.  The test therefore requires
  * err(256) < 1e-4 (max abs field difference), and
  * err(64) / err(128) in [3.5, 4.5]  (second-order convergence to the RK4 solution; observed 3.99-4.00),
which fails if the split map integrated a different velocity (the error would then not decrease)."""
import pytest
import torch

from fewterm import phi4 as P
from test_logdet import make, KINDS


def rk4(model, x, n):
    h = 1 / n
    for j in range(n):
        t = j * h
        k1 = model.velocity(x, t)
        k2 = model.velocity(x + .5 * h * k1, t + .5 * h)
        k3 = model.velocity(x + .5 * h * k2, t + .5 * h)
        k4 = model.velocity(x + h * k3, t + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@pytest.mark.parametrize('d,L', [(2, 4), (3, 3)])
@pytest.mark.parametrize('kind', KINDS)
def test_split_map_converges_to_rk4(kind, d, L):
    model, g = make(kind, d, L)
    x = torch.randn((4,) + (L,) * d, dtype=P.DT, generator=g)
    ref = rk4(model, x, 256)
    err = {n: float((model.transform(x, n)[0] - ref).abs().max()) for n in (64, 128, 256)}
    print(kind, d, L, err)
    assert err[256] < 1e-4, err
    assert 3.5 < err[64] / err[128] < 4.5, err
