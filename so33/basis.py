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

# Default metric tensor eta = diag(+1, +1, +1, -1, -1, -1).
# Cast to float64 for ODE precision; pass dtype= to build_so33_basis to use
# float32 for benchmarking.
ETA = torch.tensor([1., 1., 1., -1., -1., -1.], dtype=torch.float64)

# Process-level cache, keyed by (dtype, signature_only). Built lazily.
_BASIS_CACHE: dict[tuple, torch.Tensor] = {}


# ── Public functions ──────────────────────────────────────────────────────────

def build_so33_basis(
    dtype: torch.dtype = torch.float64,
    signature_only: bool = False,
) -> list[torch.Tensor]:
    """Return the canonical (6, 6, 6) basis tensors of so(3,3).

    Each tensor satisfies:
        eta_mu * omega[mu, nu, lambda] + eta_nu * omega[nu, mu, lambda] = 0
    for all indices mu, nu, lambda.

    Parameters
    ----------
    dtype : torch.dtype
        Element type of the basis tensors. Default float64 for ODE precision;
        pass torch.float32 for faster benchmarking (with reduced numerical
        stability under the indefinite metric).
    signature_only : bool
        When True, return only the 6 same-signature generators (pairs (i, j)
        with eta[i] == eta[j]). This realizes the so(3) ⊕ so(3) Euclidean
        ablation: 3 generators from the spacelike block {0,1,2} and 3 from
        the timelike block {3,4,5}, total 6. Default False (full so(3,3),
        15 generators).

    Returns
    -------
    basis : list of torch.Tensor, each shape (6, 6, 6), dtype=dtype,
            ordered by index pair (i, j) with 0 <= i < j <= 5.
    """
    eta   = ETA.to(dtype)
    basis = []

    for i in range(DIM):
        for j in range(i + 1, DIM):
            if signature_only and eta[i] != eta[j]:
                continue

            # Step 1 — 6x6 so(3,3) generator matrix
            A = torch.zeros(DIM, DIM, dtype=dtype)
            A[i, j] =  1.0
            A[j, i] = -(eta[i] / eta[j])   # antisymmetry condition

            # Step 2 — indicator lift: omega^{(k)}[mu, nu, lambda] = A[mu, nu] * delta_{lambda, j}
            omega_k = torch.zeros(DIM, DIM, DIM, dtype=dtype)
            omega_k[:, :, j] = A

            basis.append(omega_k)

    if not signature_only:
        assert len(basis) == N_BASIS, f"Expected {N_BASIS} basis tensors, got {len(basis)}"
    return basis


def get_basis_stack(
    dtype: torch.dtype = torch.float64,
    signature_only: bool = False,
) -> torch.Tensor:
    """Return (and lazily cache) the stacked basis of shape (K, 6, 6, 6).

    K = 15 by default, K = 6 when signature_only=True.

    The result is cached per (dtype, signature_only) so each variant is
    constructed only once per Python process.
    """
    key = (dtype, signature_only)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = torch.stack(
            build_so33_basis(dtype=dtype, signature_only=signature_only), dim=0
        )
    return _BASIS_CACHE[key]


def get_connection_tensor(
    coeffs: torch.Tensor,
    basis_stack: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the (6, 6, 6) connection tensor omega from coefficients.

    Parameters
    ----------
    coeffs      : (K,)            learnable parameter vector (nn.Parameter)
    basis_stack : (K, 6, 6, 6)    fixed basis (registered buffer in the module)

    Returns
    -------
    omega : (6, 6, 6) tensor with dtype matching basis_stack.

        omega = sum_{k=1}^{K} coeffs[k] * basis_stack[k]

    This is a linear map, so gradients w.r.t. coeffs are exact. Coefficients
    are cast to basis_stack.dtype for the contraction so callers may keep
    coeffs in any dtype (e.g. float32 for parameter storage).
    """
    return torch.einsum("k, kmnl -> mnl", coeffs.to(basis_stack.dtype), basis_stack)
