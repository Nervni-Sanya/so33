"""
Test 6 — Synthetic causal classification (Lorentzian vs Euclidean).

Generates two classes of 6-D trajectories with hidden so(3,3) symmetry:
  Class 0 — Lorentz boost connection  omega_{03}  (cross-signature, timelike)
  Class 1 — Spatial rotation connection omega_{01} (same-signature, spacelike)

Trains SO33Network for 20 full-batch steps and asserts that the loop runs
end to end: every loss and gradient is finite and the training loss
decreases. It does NOT assert accuracy: at this input scale (v(0) ~
N(0, 0.3^2)) the flow barely moves the inputs, and the classes are only
weakly separable (a gradient-boosted tree reaches test AUC ~0.61 with 3,000
training samples), so 70 samples cannot be classified above chance.

Run:
    python tests/test_causal_classification.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

from so33.network import SO33Network
from so33.basis import DIM, ETA


def generate_causal_dataset(n_samples: int = 100, seed: int = 7):
    """Return (X, y) tensors for the synthetic causal classification task."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    half = n_samples // 2
    eta  = ETA

    # Lorentz boost (0, 3): cross-signature pair — eta[0]=+1, eta[3]=-1
    A_boost = torch.zeros(DIM, DIM, dtype=torch.float64)
    A_boost[0, 3] =  1.0
    A_boost[3, 0] = -(eta[0] / eta[3])          # = +1
    omega_boost = torch.zeros(DIM, DIM, DIM, dtype=torch.float64)
    omega_boost[:, :, 3] = A_boost

    # Spatial rotation (0, 1): same-signature pair — eta[0]=eta[1]=+1
    A_rot = torch.zeros(DIM, DIM, dtype=torch.float64)
    A_rot[0, 1] =  1.0
    A_rot[1, 0] = -(eta[0] / eta[1])            # = -1
    omega_rot = torch.zeros(DIM, DIM, DIM, dtype=torch.float64)
    omega_rot[:, :, 1] = A_rot

    t_span  = torch.tensor([0.0, 1.0], dtype=torch.float64)
    samples = []

    for cls_idx, omega_cls in enumerate([omega_boost, omega_rot]):
        for _ in range(half):
            v0 = torch.randn(1, DIM, dtype=torch.float64) * 0.3

            def rhs(t, v, w=omega_cls):
                return -torch.einsum("mnl, bn, bl -> bm", w, v, v)

            with torch.no_grad():
                traj = odeint(rhs, v0, t_span, method="rk4",
                              options={"step_size": 0.1})

            samples.append((traj[-1].squeeze(0), cls_idx))

    perm = torch.randperm(len(samples))
    X = torch.stack([samples[i][0] for i in perm])
    y = torch.tensor([samples[i][1] for i in perm], dtype=torch.long)
    return X, y


def test_causal_classification() -> None:
    torch.manual_seed(7)

    X, y    = generate_causal_dataset(n_samples=100, seed=7)
    n_train = 70
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    print(f"  Dataset  : {len(X)} samples  |  "
          f"Class 0 (boost): {(y==0).sum()},  Class 1 (rot): {(y==1).sum()}")
    print(f"  Train/Test split: {n_train} / {len(X) - n_train}")

    model     = SO33Network(in_features=6, out_features=2, T=0.3, adjoint=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n  {'Epoch':>6}  {'Loss':>10}  {'Tr.Acc':>8}  {'Te.Acc':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")

    losses = []
    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss   = criterion(logits, y_tr) + model.regularization_loss()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert torch.isfinite(loss) and torch.isfinite(grad_norm), f"non-finite at epoch {epoch}"
        optimizer.step()
        losses.append(loss.item())

        if epoch % 4 == 0 or epoch == 1:
            with torch.no_grad():
                tr_acc = (logits.argmax(1) == y_tr).float().mean().item()
                te_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
            print(f"  {epoch:>6}  {loss.item():>10.5f}  "
                  f"{tr_acc:>8.3f}  {te_acc:>8.3f}")

    assert losses[-1] < losses[0], f"training loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("  ✓ Training loop completed; loss decreased")


if __name__ == "__main__":
    print("\n╔" + "─"*62 + "╗")
    print("║  TEST 6 · Synthetic causal classification                   ║")
    print("╚" + "─"*62 + "╝")
    test_causal_classification()
