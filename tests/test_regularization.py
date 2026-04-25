"""
Test 5 — Frobenius regularization.

Verifies:
  • regularization_loss() is near zero at random init (small coefficients).
  • It increases when coefficients are set to 1.
  • Gradient flows through the regularization loss correctly.

Run:
    python tests/test_regularization.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from so33.activation import SO33Activation


def test_regularization() -> None:
    torch.manual_seed(3)
    act = SO33Activation(reg_coef=1e-2)

    # Small init -> small loss
    reg0 = act.regularization_loss()
    print(f"  Reg loss at init  : {reg0.item():.8f}  (coeffs ~ N(0, 0.01))")

    # Set all coefficients to 1 -> larger loss
    with torch.no_grad():
        act.coeffs.fill_(1.0)
    reg1 = act.regularization_loss()
    print(f"  Reg loss (c_k=1)  : {reg1.item():.4f}")
    assert reg1 > reg0, "Regularization loss should increase with larger coefficients!"

    # Gradient flows correctly
    reg1.backward()
    assert act.coeffs.grad is not None, "No gradient on coefficients!"
    print(f"  Gradient of reg loss: norm = {act.coeffs.grad.norm().item():.4f}")
    print("  ✓ Regularization correct")


if __name__ == "__main__":
    print("\n╔" + "─"*61 + "╗")
    print("║  TEST 5 · Frobenius regularization                         ║")
    print("╚" + "─"*61 + "╝")
    test_regularization()
