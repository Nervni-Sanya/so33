"""
demo_minimal.py
---------------
End-to-end demonstration of SO33Activation and SO33Network.

Run:
    python examples/demo_minimal.py

Optional: install matplotlib for a loss curve plot.
    pip install matplotlib
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from so33 import SO33Activation, SO33Network

print("=" * 58)
print("  SO33 Activation — Minimal Demonstration")
print("=" * 58)

torch.manual_seed(0)

# ── 1. Single activation layer ────────────────────────────────────────────────
print("\n[1] SO33Activation — forward pass")
act = SO33Activation(T=0.5, adjoint=False)
x   = torch.randn(3, 6, dtype=torch.float64) * 0.4
y   = act(x)
print(f"    Input  shape : {tuple(x.shape)},  dtype : {x.dtype}")
print(f"    Output shape : {tuple(y.shape)},  dtype : {y.dtype}")
print(f"    Input  norms : {x.norm(dim=1).detach().numpy().round(4)}")
print(f"    Output norms : {y.norm(dim=1).detach().numpy().round(4)}")

# ── 2. Short training loop ────────────────────────────────────────────────────
print("\n[2] Training  Linear(6→6) → SO33Activation → Linear(6→1)")
model = nn.Sequential(
    nn.Linear(6, 6).double(),
    SO33Activation(T=0.5, adjoint=True),
    nn.Linear(6, 1).double(),
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.MSELoss()

X_data = torch.randn(16, 6, dtype=torch.float64)
y_data = torch.randn(16, 1, dtype=torch.float64)

losses = []
for step in range(30):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_data), y_data)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
    if step % 5 == 0:
        print(f"    Step {step:3d}  loss = {loss.item():.5f}")

print(f"\n    Initial loss : {losses[0]:.5f}")
print(f"    Final loss   : {losses[-1]:.5f}")
print(f"    Decreased    : {losses[-1] < losses[0]}")

# ── 3. SO33Network ────────────────────────────────────────────────────────────
print("\n[3] SO33Network  (in=6, out=2, T=0.3)")
net    = SO33Network(in_features=6, out_features=2, T=0.3)
logits = net(X_data)
print(f"    Logits shape : {tuple(logits.shape)}")

# ── 4. Optional loss curve ────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import os
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.plot(losses, color="steelblue", lw=2)
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title("SO33Network — training curve")
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=120)
    print("\n    Loss curve saved → results/loss_curve.png")
except ImportError:
    print("\n    (install matplotlib to save the loss curve)")

print("\n  Demo complete. ✓")
