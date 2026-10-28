"""
Test 4 — gradients of SO33Activation at the experimental solver settings.

The experiments integrate with fixed-step RK4 over T = 0.3 with step 0.03
(10 steps). At excited connections (||c|| in {0.03, 0.3, 1.0}) this checks:
  • adjoint (A) and direct autograd (B) agree on dL/dc;
  • both agree with central finite differences of the forward map. This is
    the decisive check: A and B share any bug in the forward graph, so
    A-vs-B agreement alone cannot detect one;
  • the radial derivative d/dλ L(λc) at λ = 1 equals c · ∇L. With a
    stop-gradient on the norm cap omega -> kappa omega / (kappa + ||omega||)
    the autograd value is off by exactly the factor 1 + ||omega||_F / kappa;
  • dL/dx agrees with finite differences;
  • autograd of ||omega_eff||_F matches its closed form;
  • gradients stay finite at c = 0.

Measured agreement (float64): FD vs autograd ~1e-9 (the FD truncation
floor), adjoint vs direct ~1e-11. A gradient that drops the derivative of
the cap (the stop-gradient bug) is off by 2e-3 .. 4e-2 in the
finite-difference test and by 1.4% .. 142% in the radial test, so the 1e-6
tolerance catches it with a margin of more than three orders of magnitude.

Run:
    python -m pytest tests/test_adjoint.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import pytest
import torch

from so33.activation import SO33Activation
from so33.basis import N_BASIS, DIM

# The integrator used by every experiment: 10 RK4 steps over [0, 0.3].
SOLVER = dict(T=0.3, method="rk4", solver_options={"step_size": 0.03})
COEFF_NORMS = (0.03, 0.3, 1.0)
TOL = 1e-6
FD_STEP = 1e-6


def _setup(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, DIM, dtype=torch.float64, generator=g)
    w = torch.randn(8, DIM, dtype=torch.float64, generator=g)
    d = torch.randn(N_BASIS, dtype=torch.float64, generator=g)
    return x, w, d / d.norm()


def _act(adjoint: bool, coeffs: torch.Tensor, **kw) -> SO33Activation:
    act = SO33Activation(adjoint=adjoint, **SOLVER, **kw)
    with torch.no_grad():
        act.coeffs.copy_(coeffs)
    return act


def _loss(act, x, w) -> torch.Tensor:
    # A generic linear functional of sigma(x); y.sum() would hide errors in
    # directions where the output components cancel.
    return (w * act(x)).sum()


def _coeff_grad(act, x, w) -> torch.Tensor:
    act.coeffs.grad = None
    _loss(act, x, w).backward()
    return act.coeffs.grad.detach().clone()


def _loss_at(act, coeffs, x, w) -> float:
    with torch.no_grad():
        act.coeffs.copy_(coeffs)
        return _loss(act, x, w).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("cnorm", COEFF_NORMS)
def test_adjoint_vs_direct(cnorm: float) -> None:
    x, w, d = _setup()
    c = cnorm * d
    g_adj = _coeff_grad(_act(True, c), x, w)
    g_dir = _coeff_grad(_act(False, c), x, w)
    rel = _rel(g_adj, g_dir)
    print(f"  ||c||={cnorm}: adjoint vs direct rel err = {rel:.2e}")
    assert rel < TOL, f"adjoint and direct gradients disagree: {rel:.2e}"


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("cnorm", COEFF_NORMS)
def test_coeff_gradient_matches_finite_differences(cnorm: float, adjoint: bool) -> None:
    x, w, d = _setup()
    c = cnorm * d
    act = _act(adjoint, c)
    g = _coeff_grad(act, x, w)
    fd = torch.zeros(N_BASIS, dtype=torch.float64)
    for k in range(N_BASIS):
        e = torch.zeros(N_BASIS, dtype=torch.float64)
        e[k] = FD_STEP
        fd[k] = (_loss_at(act, c + e, x, w) - _loss_at(act, c - e, x, w)) / (2 * FD_STEP)
    rel = _rel(g, fd)
    print(f"  ||c||={cnorm} adjoint={adjoint}: FD vs autograd rel err = {rel:.2e}")
    assert rel < TOL, f"autograd gradient disagrees with finite differences: {rel:.2e}"


@pytest.mark.parametrize("kappa", [1.0, 0.05])
@pytest.mark.parametrize("adjoint", [False, True])
def test_radial_derivative_includes_the_norm_cap(adjoint: bool, kappa: float) -> None:
    """d/dλ L(λc) = c·∇L. A stop-gradient on the cap makes the autograd
    side larger by exactly 1 + ||omega||_F / kappa (1.42 at ||c|| = 0.3)."""
    x, w, d = _setup(1)
    c = 0.3 * d
    act = _act(adjoint, c, connection_cap=kappa)
    radial_autograd = (_coeff_grad(act, x, w) * c).sum().item()
    delta = 1e-4      # relative step along the ray; the map is smooth along it
    radial_fd = (_loss_at(act, (1 + delta) * c, x, w)
                 - _loss_at(act, (1 - delta) * c, x, w)) / (2 * delta)
    rel = abs(radial_autograd - radial_fd) / abs(radial_fd)
    print(f"  kappa={kappa} adjoint={adjoint}: radial rel err = {rel:.2e}")
    assert rel < TOL, f"radial derivative mismatch: {rel:.2e}"


def test_input_gradient_matches_finite_differences() -> None:
    x, w, d = _setup(2)
    act = _act(False, 0.3 * d)
    xg = x.clone().requires_grad_(True)
    (w * act(xg)).sum().backward()
    g = xg.grad.detach()
    fd = torch.zeros_like(x)
    with torch.no_grad():
        for b in range(x.shape[0]):
            for m in range(DIM):
                e = torch.zeros_like(x)
                e[b, m] = FD_STEP
                fd[b, m] = ((w * act(x + e)).sum() - (w * act(x - e)).sum()) / (2 * FD_STEP)
    rel = _rel(g, fd)
    print(f"  dL/dx FD vs autograd rel err = {rel:.2e}")
    assert rel < TOL


@pytest.mark.parametrize("kappa", [1.0, 0.05])
def test_effective_connection_norm_gradient_closed_form(kappa: float) -> None:
    """||omega_eff|| = kappa r / (kappa + r) with r = ||omega||_F = sqrt(2)||c||,
    so its gradient is kappa^2 / (kappa + r)^2 * 2c / r. Detaching the scale
    would give kappa / (kappa + r) * 2c / r instead."""
    _, _, d = _setup(3)
    c = 0.3 * d
    act = _act(False, c, connection_cap=kappa)
    torch.linalg.vector_norm(act.effective_connection()).backward()
    r = math.sqrt(2.0) * c.norm().item()
    expected = kappa ** 2 / (kappa + r) ** 2 * 2.0 * c / r
    assert torch.allclose(act.coeffs.grad, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("adjoint", [False, True])
def test_gradient_is_finite_at_zero_connection(adjoint: bool) -> None:
    x, w, _ = _setup(4)
    act = _act(adjoint, torch.zeros(N_BASIS, dtype=torch.float64))
    g = _coeff_grad(act, x, w)
    assert torch.isfinite(g).all()
    assert g.norm().item() > 0     # the linear term still moves the coefficients


if __name__ == "__main__":
    print("\n╔" + "─"*62 + "╗")
    print("║  TEST 4 · SO33Activation gradients (adjoint, direct, FD)    ║")
    print("╚" + "─"*62 + "╝")
    for cn in COEFF_NORMS:
        test_adjoint_vs_direct(cn)
        for adj in (False, True):
            test_coeff_gradient_matches_finite_differences(cn, adj)
    for adj in (False, True):
        for kap in (1.0, 0.05):
            test_radial_derivative_includes_the_norm_cap(adj, kap)
        test_gradient_is_finite_at_zero_connection(adj)
    test_input_gradient_matches_finite_differences()
    for kap in (1.0, 0.05):
        test_effective_connection_norm_gradient_closed_form(kap)
    print("  ✓ Gradients agree with finite differences")
