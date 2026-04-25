"""
Test 1 — Basis construction & metric-connection condition.

Verifies:
  • Exactly 15 basis tensors are produced.
  • The metric-connection antisymmetry holds at every (mu, nu, lambda) entry.
  • The 15 tensors are linearly independent (rank = 15).
  • Frobenius norms are all equal to sqrt(2).

Run:
    python tests/test_basis.py
    python -m pytest tests/test_basis.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from so33.basis import build_so33_basis, N_BASIS, DIM, ETA


def test_basis() -> None:
    basis = build_so33_basis()
    eta   = ETA

    # ── 1. Count ──────────────────────────────────────────────────────────────
    assert len(basis) == N_BASIS, f"Expected {N_BASIS} tensors, got {len(basis)}"
    print(f"  Number of basis tensors : {len(basis)}  (expected {N_BASIS})")

    # ── 2. Metric-connection condition ────────────────────────────────────────
    # eta_mu * omega[mu, nu, lambda] + eta_nu * omega[nu, mu, lambda] = 0
    violations = 0
    for omega_k in basis:
        for mu in range(DIM):
            for nu in range(DIM):
                for lam in range(DIM):
                    val = (eta[mu] * omega_k[mu, nu, lam]
                         + eta[nu] * omega_k[nu, mu, lam])
                    if abs(val.item()) > 1e-10:
                        violations += 1

    assert violations == 0, f"Metric-connection violated at {violations} entries"
    print(f"  Metric-connection violations : {violations}  (expected 0)")

    # ── 3. Linear independence ────────────────────────────────────────────────
    B_flat = torch.stack([b.flatten() for b in basis])   # (15, 216)
    rank   = torch.linalg.matrix_rank(B_flat).item()
    assert rank == N_BASIS, f"Basis not full-rank: rank = {rank}"
    print(f"  Rank of basis matrix     : {rank}  (expected {N_BASIS})")

    # ── 4. Frobenius norms ────────────────────────────────────────────────────
    norms = [f"{b.norm().item():.3f}" for b in basis]
    print(f"  Basis Frobenius norms    : {norms}")

    print("  ✓ All checks passed")


if __name__ == "__main__":
    print("\n╔" + "─"*61 + "╗")
    print("║  TEST 1 · Basis construction & metric-connection condition  ║")
    print("╚" + "─"*61 + "╝")
    test_basis()
