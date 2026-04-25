"""
Test 3 — Minimal training step (spec requirement).

Architecture: nn.Sequential(Linear(6→6), SO33Activation(T=0.5), Linear(6→1))
Batch size 4, one Adam step, MSE loss.

Verifies:
  • Forward pass produces finite output.
  • Gradient on connection coefficients is non-zero and finite.
  • Loss decreases after one optimizer step.

Run:
    python tests/test_training.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from so33.activation import SO33Activation
from so33.basis import N_BASIS


def test_minimal_training() -> None:
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(6, 6).double(),
        SO33Activation(T=0.5, rtol=1e-3, atol=1e-4,
                       method="dopri5", adjoint=True),
        nn.Linear(6, 1).double(),
    )

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: Linear(6→6) → SO33Activation(T=0.5) → Linear(6→1)")
    print(f"  Total trainable parameters: {n_total}")
    print(f"    ├─ Linear(6→6)   : {6*6+6} params")
    print(f"    ├─ SO33Activation: {N_BASIS} params  [connection coefficients]")
    print(f"    └─ Linear(6→1)   : {6*1+1} params")

    x_batch = torch.randn(4, 6, dtype=torch.float64)
    y_true  = torch.randn(4, 1, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Forward
    y_pred = model(x_batch)
    loss   = criterion(y_pred, y_true)
    print(f"\n  ─── Forward pass ───────────────────────────────────")
    print(f"  y_pred  : {y_pred.detach().squeeze().numpy().round(5)}")
    print(f"  y_true  : {y_true.squeeze().numpy().round(5)}")
    print(f"  MSE Loss: {loss.item():.6f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    act      = model[1]
    g_coeffs = act.coeffs.grad

    assert g_coeffs is not None,             "No gradient on connection coefficients!"
    assert torch.isfinite(g_coeffs).all(),   "Non-finite gradients!"
    assert g_coeffs.norm().item() > 0,       "Zero gradient on coefficients!"

    print(f"\n  ─── Gradient check ─────────────────────────────────")
    print(f"  ∂L/∂c (all 15): {g_coeffs.detach().numpy().round(5)}")
    print(f"  ‖∂L/∂c‖₂      : {g_coeffs.norm().item():.6f}")

    # Optimizer step
    optimizer.step()
    loss2 = criterion(model(x_batch), y_true)
    print(f"\n  ─── After one Adam step ────────────────────────────")
    print(f"  New MSE Loss  : {loss2.item():.6f}")
    print(f"  Loss decreased: {loss2.item() < loss.item()}")

    assert loss2.item() < loss.item(), "Loss did not decrease after one Adam step!"
    print("  ✓ Minimal training step completed without errors")


if __name__ == "__main__":
    print("\n╔" + "─"*61 + "╗")
    print("║  TEST 3 · Minimal training step  [SPEC REQUIREMENT]       ║")
    print("╚" + "─"*61 + "╝")
    test_minimal_training()
