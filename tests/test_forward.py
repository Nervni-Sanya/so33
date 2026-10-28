"""
Test 2 — SO33Activation forward pass.

Verifies:
  • Output shape matches input shape (B, 6), dtype float64, finite values
    (adaptive dopri5 and the experimental fixed-step RK4 configuration).
  • A fixed-grid solver without a step defaults to T / 10, not one step.
  • With c = 0 the activation is the identity; with c != 0 it is not.

Run:
    python tests/test_forward.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from so33.activation import SO33Activation
from so33.basis import DIM, N_BASIS


def test_forward() -> None:
    torch.manual_seed(1)

    act = SO33Activation(T=0.5, method="dopri5", adjoint=False,
                         rtol=1e-4, atol=1e-5)
    B = 4
    x = torch.randn(B, DIM, dtype=torch.float64) * 0.3

    print(f"  Input  : shape={tuple(x.shape)}, dtype={x.dtype}")
    y = act(x)
    print(f"  Output : shape={tuple(y.shape)}, dtype={y.dtype}")
    print(f"  Input  norms: {x.norm(dim=1).detach().numpy().round(4)}")
    print(f"  Output norms: {y.norm(dim=1).detach().numpy().round(4)}")

    assert y.shape == (B, DIM),        f"Wrong shape: {y.shape}"
    assert y.dtype == torch.float64,   f"Wrong dtype: {y.dtype}"
    assert torch.isfinite(y).all(),    "Non-finite output detected!"

    print("  ✓ Forward pass correct")


def test_forward_experimental_config() -> None:
    """The configuration of every experiment: 10 RK4 steps over T = 0.3."""
    torch.manual_seed(2)
    act = SO33Activation(T=0.3, method="rk4", solver_options={"step_size": 0.03},
                         adjoint=True)
    x = torch.randn(16, DIM, dtype=torch.float64)
    y = act(x)
    assert y.shape == (16, DIM) and torch.isfinite(y).all()


def test_fixed_grid_solver_defaults_to_ten_steps() -> None:
    act = SO33Activation(T=0.3, method="rk4")
    assert act.solver_options == {"step_size": 0.03}
    act = SO33Activation(T=0.3, method="rk4", solver_options={"step_size": 0.01})
    assert act.solver_options == {"step_size": 0.01}


def test_identity_only_at_zero_connection() -> None:
    torch.manual_seed(3)
    x = torch.randn(8, DIM, dtype=torch.float64)
    act = SO33Activation(T=0.3, method="rk4", adjoint=False)
    with torch.no_grad():
        act.coeffs.zero_()
    assert torch.equal(act(x), x)
    with torch.no_grad():
        act.coeffs.copy_(torch.full((N_BASIS,), 0.1, dtype=torch.float64))
    assert (act(x) - x).norm() > 1e-3


if __name__ == "__main__":
    print("\n╔" + "─"*61 + "╗")
    print("║  TEST 2 · SO33Activation forward pass                      ║")
    print("╚" + "─"*61 + "╝")
    test_forward()
    test_forward_experimental_config()
    test_fixed_grid_solver_defaults_to_ten_steps()
    test_identity_only_at_zero_connection()
