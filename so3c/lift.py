"""
lift.py
-------
Lorentz-covariant lift of 4-momenta into C^3: the bivector (Riemann-
Silberstein) map.

Why not a direct lift
---------------------
A single 4-momentum transforms in the (1/2, 1/2) representation of the
Lorentz group — it is NOT a C^3 object, and any componentwise placement of
(E, px, py, pz) into C^3 breaks covariance (e.g. z = (px + iE, py, pz) has
Im(z . z) = 2 px E, which is frame-dependent).

The (1, 0) representation — our C^3 with its SO(3, C) action — is the
self-dual antisymmetric 2-tensor. The natural covariant source is the
bivector of a PAIR of 4-vectors, F = p ^ q, decomposed like an
electromagnetic field:

    E_i = F^{0i} = E_p q_i - p_i E_q          ("electric" part)
    B_i = (1/2) eps_{ijk} F^{jk} = (p x q)_i  ("magnetic" part)
    z   = E + i B  in C^3

Under a Lorentz transformation Lambda = exp(omega(rho, eta)) acting on both
4-vectors, z transforms EXACTLY as z -> Q z with Q = expm_so3c(rho + i eta)
— same (rho, eta), verified to machine precision in test_so3c_lift.py.

Jet lift
--------
``jet_bivectors`` pairs every constituent with the total (masked) jet
momentum P: z_a = bivec(p_a, P). This is O(K), exactly covariant, and
permutation-equivariant.

Two honest structural notes (they shape the readout design in
benchmarks/so3c_models.py):
1. For simple bivectors (any p ^ q) the pseudoscalar invariant vanishes:
   Im(z . z) = 2 E.B = 0 identically. Moreover, for two bivectors SHARING
   a leg (here P), the cross pseudoscalar Im(z_a . z_b) ~ eps(p_a, P, p_b, P)
   also vanishes. Parity-odd information therefore does NOT survive this
   lift — which is fine for top tagging (a parity-even task); the modelling
   power comes from the geodesic flow mixing bivectors (after which states
   are no longer simple and cross-Im terms are populated) and from richer
   pooled pair statistics.
2. Re(z_a . z_b) is the symmetric bivector pairing — a polynomial in the
   Minkowski products (p_a . p_b, p_a . P, P^2), i.e. a covariantly
   complete set of the same invariants eta_invariants pools.
"""

from __future__ import annotations

import torch

from .algebra import cross_matrix, so3_generators


def bivector_lift(p4: torch.Tensor, q4: torch.Tensor) -> torch.Tensor:
    """(…, 4) x (…, 4) real (E, px, py, pz) -> (…, 3) complex bivector.

    z = (E_p * q_vec - E_q * p_vec)  +  i (p_vec x q_vec)
    """
    Ep, pv = p4[..., :1], p4[..., 1:]
    Eq, qv = q4[..., :1], q4[..., 1:]
    e_part = Ep * qv - Eq * pv
    b_part = torch.linalg.cross(pv, qv, dim=-1)
    return torch.complex(e_part, b_part)


def jet_bivectors(p4: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, K, 4) real 4-momenta + (B, K) mask -> (B, K, 3) complex.

    z_a = bivec(p_a, P) with P the masked jet-total 4-momentum. Padded
    entries produce z = 0 (they are zero 4-vectors after masking); callers
    should still apply the mask to pooled statistics.
    """
    p4 = p4 * mask.unsqueeze(-1)
    P = p4.sum(dim=1, keepdim=True)                      # (B, 1, 4)
    return bivector_lift(p4, P.expand_as(p4))


def minkowski_inner(p4: torch.Tensor, q4: torch.Tensor) -> torch.Tensor:
    """(…, 4) x (…, 4) -> (…,): <p, q> = E_p E_q - p_vec . q_vec  (+---)."""
    return p4[..., 0] * q4[..., 0] - (p4[..., 1:] * q4[..., 1:]).sum(dim=-1)


# ── 4x4 Lorentz transforms (for tests and augmentation) ──────────────────────

def lorentz_matrix(
    rho: torch.Tensor,
    beta: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """exp of the so(3,1) element with rotation rho and boost rapidity beta.

    Layout matches the harness 4-momentum order (E, px, py, pz):

        omega[0, 1+i] = omega[1+i, 0] = beta_i     (boost block)
        omega[1:, 1:] = [rho]_x                    (rotation block)

    The correspondence  Lambda(rho, beta) on 4-vectors  <->
    expm_so3c(rho + i beta) on bivectors  uses the SAME (rho, beta).
    """
    rho = rho.to(dtype)
    beta = beta.to(dtype)
    omega = torch.zeros(4, 4, dtype=dtype)
    omega[0, 1:] = beta
    omega[1:, 0] = beta
    omega[1:, 1:] = cross_matrix(rho)
    return torch.matrix_exp(omega)


def random_lorentz_pair(
    rot_scale: float = 1.0,
    boost_scale: float = 0.5,
    dtype: torch.dtype = torch.float64,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matched pair (Lambda 4x4 real, a complex 3-vector) for equivariance
    tests: Lambda = lorentz_matrix(rho, beta), a = rho + i beta so that
    bivec(Lambda p, Lambda q) = expm_so3c(a) @ bivec(p, q)."""
    rho = torch.randn(3, dtype=dtype, generator=generator) * rot_scale
    beta = torch.randn(3, dtype=dtype, generator=generator) * boost_scale
    return lorentz_matrix(rho, beta, dtype=dtype), torch.complex(rho, beta)


__all__ = [
    "bivector_lift",
    "jet_bivectors",
    "minkowski_inner",
    "lorentz_matrix",
    "random_lorentz_pair",
    "so3_generators",
]
