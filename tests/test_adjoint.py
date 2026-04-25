"""
Test 4 — Adjoint (A) vs Direct autograd (B) gradient consistency.

Both backprop approaches must agree on the gradient of the connection
coefficients to within 5% relative error (threshold accounts for different
numerical ODE trajectories between adjoint and direct passes).

Run:
    python tests/test_adjoint.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from so33.activation import SO33Activation
from so33.basis import N_BASIS, DIM


def test_adjoint_vs_direct() -> None:
    torch.manual_seed(9)

    # Fixed coefficients and input so both methods see identical omega and x
    fixed_coeffs = torch.randn(N_BASIS, dtype=torch.float64) * 0.05
    x_fixed      = torch.randn(2, DIM, dtype=torch.float64) * 0.2

    def get_grad(adjoint: bool) -> torch.Tensor:
        act = SO33Activation(T=0.3, rtol=1e-6, atol=1e-7,
                              method="rk4", adjoint=adjoint)
        with torch.no_grad():
            act.coeffs.copy_(fixed_coeffs)
        y = act(x_fixed.detach().clone())
        y.sum().backward()
        return act.coeffs.grad.clone()

    g_adj    = get_grad(adjoint=True)
    g_direct = get_grad(adjoint=False)

    max_diff = (g_adj - g_direct).abs().max().item()
    rel_diff = max_diff / (g_direct.abs().max().item() + 1e-12)

    print(f"  Gradient (adjoint)   : {g_adj.numpy().round(5)}")
    print(f"  Gradient (direct AD) : {g_direct.numpy().round(5)}")
    print(f"  Max absolute diff    : {max_diff:.2e}")
    print(f"  Max relative diff    : {rel_diff:.2e}  (threshold: 5e-2)")

    assert rel_diff < 5e-2, f"Gradient mismatch too large: rel_diff = {rel_diff:.2e}"
    print("  ✓ Adjoint and direct gradients are consistent")


if __name__ == "__main__":
    print("\n╔" + "─"*62 + "╗")
    print("║  TEST 4 · Adjoint (A) vs Direct autograd (B) consistency   ║")
    print("╚" + "─"*62 + "╝")
    test_adjoint_vs_direct()
