"""
Test 2 — SO33Activation forward pass.

Verifies:
  • Output shape matches input shape (B, 6).
  • Output dtype is float64.
  • All output values are finite.

Run:
    python tests/test_forward.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from so33.activation import SO33Activation
from so33.basis import DIM


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


if __name__ == "__main__":
    print("\n╔" + "─"*61 + "╗")
    print("║  TEST 2 · SO33Activation forward pass                      ║")
    print("╚" + "─"*61 + "╝")
    test_forward()
