"""
basis.py
--------
Construction of the 15-dimensional so(3,3) connection basis.

Mathematical background
-----------------------
Manifold : R^{3,3},  metric  eta = diag(+1, +1, +1, -1, -1, -1)

Lie algebra:
    so(3,3) = { A in R^{6x6} : A^T eta + eta A = 0 },   dim = 15

For each ordered pair (i, j) with i < j, the generator matrix is:
    A^{(ij)}[mu, nu] = delta_{mu,i} * delta_{nu,j}
                     - (eta_i / eta_j) * delta_{mu,j} * delta_{nu,i}

Indicator lift to a (6, 6, 6) connection tensor:
    omega^{(k=(i,j))}[mu, nu, lambda] = A^{(ij)}[mu, nu] * delta_{lambda, j}

This lift satisfies the metric-connection condition:
    eta_mu * omega[mu, nu, lambda] + eta_nu * omega[nu, mu, lambda] = 0
for all mu, nu, lambda — verified analytically and in test_basis.py.

Full connection reconstructed as linear combination:
    omega^mu_{nu lambda} = sum_{k=1}^{15} c_k * omega^{(k), mu}_{nu lambda}
"""

from __future__ import annotations

import torch

# ── Constants ─────────────────────────────────────────────────────────────────
DIM     = 6                        # dimension of R^{3,3}
N_BASIS = DIM * (DIM - 1) // 2    # = 15 : dim(so(3,3))

# Metric tensor eta = diag(+1, +1, +1, -1, -1, -1)  (float64 for ODE precision)
ETA = torch.tensor([1., 1., 1., -1., -1., -1.], dtype=torch.float64)

# Process-level cache — built once, reused across calls
_BASIS_STACK: torch.Tensor | None = None


# ── Public functions ──────────────────────────────────────────────────────────

def build_so33_basis() -> list[torch.Tensor]:
    """Return the 15 canonical (6, 6, 6) basis tensors of so(3,3).

    Each tensor satisfies:
        eta_mu * omega[mu, nu, lambda] + eta_nu * omega[nu, mu, lambda] = 0
    for all indices mu, nu, lambda.

    Returns
    -------
    basis : list of 15 torch.Tensor, each shape (6, 6, 6), dtype=float64,
            ordered by index pair (i, j) with 0 <= i < j <= 5.
    """
    eta   = ETA
    basis = []

    for i in range(DIM):
        for j in range(i + 1, DIM):
            # Step 1 — 6x6 so(3,3) generator matrix
            A = torch.zeros(DIM, DIM, dtype=torch.float64)
            A[i, j] =  1.0
            A[j, i] = -(eta[i] / eta[j])   # antisymmetry condition

            # Step 2 — indicator lift: omega^{(k)}[mu, nu, lambda] = A[mu, nu] * delta_{lambda, j}
            omega_k = torch.zeros(DIM, DIM, DIM, dtype=torch.float64)
            omega_k[:, :, j] = A

            basis.append(omega_k)

    assert len(basis) == N_BASIS, f"Expected {N_BASIS} basis tensors, got {len(basis)}"
    return basis


def get_basis_stack() -> torch.Tensor:
    """Return (and lazily cache) the stacked basis of shape (15, 6, 6, 6).

    The result is cached in a module-level variable so the basis is
    constructed only once per Python process.
    """
    global _BASIS_STACK
    if _BASIS_STACK is None:
        _BASIS_STACK = torch.stack(build_so33_basis(), dim=0)  # (15, 6, 6, 6)
    return _BASIS_STACK


def get_connection_tensor(
    coeffs: torch.Tensor,
    basis_stack: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the (6, 6, 6) connection tensor omega from 15 coefficients.

    Parameters
    ----------
    coeffs      : (15,)       learnable parameter vector (nn.Parameter)
    basis_stack : (15, 6, 6, 6)  fixed basis (registered buffer in the module)

    Returns
    -------
    omega : (6, 6, 6) float64 tensor
        omega = sum_{k=1}^{15} coeffs[k] * basis_stack[k]

    This is a linear map, so gradients w.r.t. coeffs are exact.
    """
    return torch.einsum("k, kmnl -> mnl", coeffs.double(), basis_stack)
