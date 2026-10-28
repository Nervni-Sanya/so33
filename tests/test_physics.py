"""
Test 10 — structural properties of the geodesic flow and its stabilisers.

All checks use the experimental integrator (RK4, step 0.03, T = 0.3)
unless stated otherwise. Tolerances are set from measured values (in
parentheses) with at least two orders of magnitude of headroom.

  • <v, v>_eta is conserved at an excited connection (drift ~1e-11),
    and the drift falls by ~16x when the step is halved (fourth order).
  • sigma is the identity at c = 0.
  • Conservation is not equivariance: at ||c|| = 0.3 the activation is
    clearly not SO(3,3)-equivariant (eps_sigma ~6e-2) while the eta-norm
    readout error is at round-off (eps_m ~7e-12).
  • The eta rescaling commutes with SO(3,3) (defect ~3e-14); the
    Euclidean bound does not, with the closed-form defect
    1 - 2 / (1 + sqrt(cosh 2)) for x = e_3 boosted by rapidity 1 (the
    mechanism behind the 0.663 boost-OOD AUC of Architecture B).
  • The eta rescaling is the identity on the null cone, so it bounds
    nothing there (massless constituents).
  • RK4 converges at fourth order on the exact null-cone solution
    v(t) = x / (1 - c u0 t) of a single boost coefficient.
  • safe_connection_cap: the worst case reaches exactly the predicted
    growth 1 / (1 - margin * 3 sqrt(6) / 8) <= 2; random and adversarial
    connections of any size stay finite (forward and adjoint) and obey
    ||sigma(x)|| <= r0 / (1 - alpha r0 T); the legacy cap blows up on
    the same worst case.

Run:
    python -m pytest tests/test_physics.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import pytest
import torch

from so33.activation import SO33Activation, safe_connection_cap, BOOST_GROWTH_CONST
from so33.basis import DIM, ETA, N_BASIS
from benchmarks.equivariance import activation_equivariance, sample_group_elements
from benchmarks.synthetic import _boost_matrix

SOLVER = dict(T=0.3, method="rk4", solver_options={"step_size": 0.03})
PAIRS = [(i, j) for i in range(DIM) for j in range(i + 1, DIM)]
K03 = PAIRS.index((0, 3))          # the boost generator mixing axes 0 and 3


def _act(coeffs, bound=False, kappa=1.0, adjoint=False, **solver):
    act = SO33Activation(adjoint=adjoint, bound_input=bound,
                         connection_cap=kappa, **{**SOLVER, **solver})
    with torch.no_grad():
        act.coeffs.copy_(coeffs)
    return act


def _m2(v):
    return (v * ETA * v).sum(-1)


def _data(seed=0, n=64):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(N_BASIS, dtype=torch.float64, generator=g)
    x = torch.randn(n, DIM, dtype=torch.float64, generator=g)
    return d / d.norm(), x


def _single_boost(c):
    coeffs = torch.zeros(N_BASIS, dtype=torch.float64)
    coeffs[K03] = c
    return coeffs


def _null_input(u0):
    """u0 (-1, 0, 0, 1, 0, 0): a null vector on which the (0,3) boost acts radially."""
    x = torch.zeros(1, DIM, dtype=torch.float64)
    x[0, 0], x[0, 3] = -u0, u0
    return x


ZERO = torch.zeros(N_BASIS, dtype=torch.float64)


# ── conservation ────────────────────────────────────────────────────────────

def test_eta_norm_is_conserved_at_excited_connection():
    d, x = _data()
    y = _act(0.4 * d)(x)
    drift = ((_m2(y) - _m2(x)).abs() / (1 + x.pow(2).sum(-1))).max().item()
    assert drift < 1e-9          # measured ~1e-11 (RK4 truncation)


def test_eta_norm_drift_is_fourth_order_in_the_step():
    d, x = _data()
    e1 = (_m2(_act(0.4 * d)(x)) - _m2(x)).abs().max().item()
    e2 = (_m2(_act(0.4 * d, solver_options={"step_size": 0.015})(x))
          - _m2(x)).abs().max().item()
    assert 8.0 < e1 / e2 < 32.0  # measured 15.8 (16 for a fourth-order method)


def test_identity_at_zero_connection():
    _, x = _data()
    assert torch.equal(_act(ZERO)(x), x)


def test_conservation_is_not_equivariance():
    """At c != 0 the eta-norm readout is invariant but the activation is not."""
    d, x = _data()
    gs = sample_group_elements(8, rapidity=0.6, seed=1)
    res = activation_equivariance(_act(0.3 * d), 0.5 * x, gs)
    assert res["eps_sigma"] > 1e-3   # measured 5.9e-2
    assert res["eps_m"] < 1e-8       # measured 7.4e-12


# ── input bounds ────────────────────────────────────────────────────────────

def test_eta_rescaling_is_equivariant():
    _, x = _data(n=256)
    x = x[_m2(x).abs() >= 0.1]
    act = _act(ZERO, bound="eta")
    for g in sample_group_elements(8, rapidity=0.6, seed=1):
        lhs, rhs = act(x @ g.T), act(x) @ g.T
        defect = ((lhs - rhs).norm(dim=-1) / rhs.norm(dim=-1)).max().item()
        assert defect < 1e-12        # measured 2.7e-14


def test_euclidean_bound_is_not_equivariant_closed_form():
    """beta_euc(x) = x / (1 + ||x||); for x = e_3 and a rapidity-1 boost in
    the (0, 3) plane the relative defect is 1 - 2 / (1 + sqrt(cosh 2))."""
    boost = _boost_matrix(torch.tensor([1.0], dtype=torch.float64),
                          space_axis=0, time_axis=3)[0]
    e3 = torch.zeros(1, DIM, dtype=torch.float64)
    e3[0, 3] = 1.0
    act = _act(ZERO, bound="euclidean")
    lhs, rhs = act(e3 @ boost.T), act(e3) @ boost.T
    defect = ((lhs - rhs).norm() / rhs.norm()).item()
    assert defect == pytest.approx(1 - 2 / (1 + math.sqrt(math.cosh(2.0))), abs=1e-12)


def test_euclidean_bound_breaks_the_eta_norm_readout():
    """The readout <beta(x), beta(x)>_eta changes under a boost for beta_euc
    (the cause of the boost-OOD failure) but not for beta_eta."""
    _, x = _data(n=256)
    gs = sample_group_elements(8, rapidity=0.6, seed=1)
    euc = activation_equivariance(_act(ZERO, bound="euclidean"), x, gs)
    eta = activation_equivariance(_act(ZERO, bound="eta"), x, gs)
    assert euc["eps_m"] > 0.05
    assert eta["eps_m"] < 1e-12


def test_eta_rescaling_is_identity_on_the_null_cone():
    x = torch.tensor([[3.0, 4.0, 12.0, 13.0, 0.0, 0.0]], dtype=torch.float64)
    assert _m2(x).item() == 0.0
    y = _act(ZERO, bound="eta")(x)
    assert torch.allclose(y, x / (1 + 1e-6), rtol=1e-12, atol=0)
    big = _act(ZERO, bound="eta")(1e6 * x)
    assert big.norm().item() > 1e7   # no bound at all along the cone


# ── integrator ──────────────────────────────────────────────────────────────

def test_rk4_is_fourth_order_on_the_exact_solution():
    T, growth_rate = 0.25, 0.25             # effective coefficient c_eff
    c = growth_rate / (1 - math.sqrt(2) * growth_rate)   # raw c giving c_eff = 0.25 at kappa = 1
    u0 = 0.5 / (growth_rate * T)            # c_eff u0 T = 1/2 -> v(T) = 2 x exactly
    x = _null_input(u0)
    errors = []
    for n in (8, 16, 32, 64):               # dyadic steps land exactly on T
        y = _act(_single_boost(c), T=T, solver_options={"step_size": T / n})(x)
        errors.append(((y - 2 * x).norm() / (2 * x).norm()).item())
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(3)]
    assert all(3.8 < o < 4.3 for o in orders), orders   # measured 4.12, 4.10, 4.07


# ── stability guarantee ─────────────────────────────────────────────────────

def test_safe_cap_worst_case_reaches_the_predicted_growth():
    R, T, margin = 50.0, 0.3, 0.5
    kappa = safe_connection_cap(R, T, margin)
    x = _null_input(R / math.sqrt(2))                 # ||x|| = R
    y = _act(_single_boost(1e6 * kappa), kappa=kappa)(x)
    predicted = 1 / (1 - margin * 3 * math.sqrt(6) / 8)
    assert y.norm().item() / R == pytest.approx(predicted, rel=1e-5)
    assert predicted <= 1 / (1 - margin)


def test_legacy_cap_blows_up_on_the_same_worst_case():
    x = _null_input(40 / math.sqrt(2))
    y = _act(_single_boost(1.0))(x)
    assert not bool(torch.isfinite(y).all()) or y.norm().item() > 1e6


def test_safe_cap_keeps_every_connection_finite():
    R, T = 50.0, 0.3
    kappa = safe_connection_cap(R, T)
    g = torch.Generator().manual_seed(3)
    # Inputs up to 1.5 R: half Gaussian directions, half lifted massless
    # momenta (p, |p|, 0, 0), i.e. null vectors.
    gauss = torch.randn(128, DIM, dtype=torch.float64, generator=g)
    p = torch.randn(128, 3, dtype=torch.float64, generator=g)
    null = torch.cat([p, p.norm(dim=-1, keepdim=True),
                      torch.zeros(128, 2, dtype=torch.float64)], dim=-1)
    x = torch.cat([gauss, null])
    x = x / x.norm(dim=-1, keepdim=True) * (1.5 * R * torch.rand(256, 1, dtype=torch.float64, generator=g))
    r0 = x.norm(dim=-1)
    for trial in range(12):
        size = kappa * 10 ** (-2 + 6 * trial / 11)          # 1e-2 .. 1e4 times kappa
        if trial % 3 == 0:
            c = _single_boost(size)
        else:
            c = torch.randn(N_BASIS, dtype=torch.float64, generator=g)
            c = c / c.norm() * size
        act = _act(c, kappa=kappa, adjoint=True)
        y = act(x)
        assert torch.isfinite(y).all()
        alpha = BOOST_GROWTH_CONST * act.effective_connection_norm()
        bound = r0 / (1 - alpha * r0 * T)
        assert (y.norm(dim=-1) <= bound * (1 + 1e-6)).all()
        (y * torch.randn(y.shape, dtype=torch.float64, generator=g)).sum().backward()
        assert torch.isfinite(act.coeffs.grad).all()


def test_safe_connection_cap_formula():
    assert safe_connection_cap(1.0, 1.0, margin=0.5) == pytest.approx(
        0.5 * 3 * math.sqrt(6) / 4)
    with pytest.raises(ValueError):
        safe_connection_cap(0.0, 0.3)
    with pytest.raises(ValueError):
        safe_connection_cap(1.0, 0.3, margin=1.0)
