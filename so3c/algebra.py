"""
algebra.py
----------
Complexification of so(3): the real Lie algebra so(3, C) acting on C^3 = R^6.

Mathematical background
-----------------------
Complexifying the rotation algebra gives the Lorentz algebra:

    so(3) (+) i so(3)  =  so(3, C)  ~=  so(3, 1)  ~=  sl(2, C)_R

The generators J_i of so(3) stay rotations; the imaginary copies K_i = i J_i
obey the boost commutation relations:

    [J_i, J_j] = eps_ijk J_k
    [J_i, K_j] = eps_ijk K_k
    [K_i, K_j] = -eps_ijk J_k

Physical model: the Riemann-Silberstein vector F = E + i B lives in C^3 and
transforms under exactly this action (rotations rotate E and B together,
boosts mix them through the imaginary part).

Realification C^3 ~= R^6
------------------------
A complex vector z = x + i y is stored as the real 6-vector v = (x, y).
Multiplication by i becomes the complex-structure matrix

    Jc = [[0, -I3], [I3, 0]] ,      Jc v  <->  i z .

The complex bilinear form  z . z = sum_i z_i^2  (NO conjugation) splits as

    Re(z . z) = |x|^2 - |y|^2 = v^T eta v ,   eta = diag(+1,+1,+1,-1,-1,-1)
    Im(z . z) = 2 x . y

so the realified SO(3, C) sits inside the SO(3, 3) of the parent `so33`
package as the subgroup commuting with Jc, and Re(z . z) is precisely the
eta-invariant already used throughout `so33`. SO(3, C) preserves BOTH real
invariants; the larger SO(3, 3) preserves only the first.

Basis convention
----------------
so(3) generators on R^3:  L_i[j, k] = -eps_ijk , so that

    sum_i a_i L_i = [a]_x   (cross-product matrix:  [a]_x u = a x u)

and so(3, C) ~= C^3 via  a = rho + i beta  <->  [a]_x  (complex 3x3,
antisymmetric). On R^6 the realified generators are

    R_i = blockdiag(L_i, L_i)          (rotations)
    B_i = Jc R_i = [[0, -L_i], [L_i, 0]]   (boosts)

Closed-form exponential (complex Rodrigues formula)
---------------------------------------------------
For A = [a]_x with a in C^3 and theta^2 = a . a (complex):

    exp(A) = I + sin(theta)/theta * A + (1 - cos(theta))/theta^2 * A^2

Both coefficient functions are even entire functions of theta, so the branch
of the complex square root is irrelevant; a Taylor series handles theta ~ 0.
"""

from __future__ import annotations

import torch

# ── Constants ─────────────────────────────────────────────────────────────────
DIM_C = 3   # complex dimension
DIM_R = 6   # real dimension of the realification C^3 ~= R^6
N_GEN = 6   # dim so(3, C) as a real Lie algebra (3 rotations + 3 boosts)

# Same metric as so33.basis.ETA — Re(z . z) = v^T eta v (tested for equality).
ETA = torch.tensor([1., 1., 1., -1., -1., -1.], dtype=torch.float64)

# Taylor-series switchover for the Rodrigues coefficients.
_SERIES_THRESHOLD = 1e-8


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


# ── Basic maps between R^6 and C^3 ────────────────────────────────────────────

def real_to_complex(v: torch.Tensor) -> torch.Tensor:
    """(…, 6) real  ->  (…, 3) complex :  (x, y) -> x + i y."""
    return torch.complex(v[..., :DIM_C], v[..., DIM_C:])


def complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """(…, 3) complex  ->  (…, 6) real :  z -> (Re z, Im z)."""
    return torch.cat([z.real, z.imag], dim=-1)


def complex_structure(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The complex-structure matrix Jc (6, 6):  Jc v <-> i z,  Jc^2 = -I."""
    eye = torch.eye(DIM_C, dtype=dtype)
    top = torch.cat([torch.zeros(DIM_C, DIM_C, dtype=dtype), -eye], dim=1)
    bot = torch.cat([eye, torch.zeros(DIM_C, DIM_C, dtype=dtype)], dim=1)
    return torch.cat([top, bot], dim=0)


# ── Invariants ────────────────────────────────────────────────────────────────

def complex_bilinear(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """SO(3, C)-invariant bilinear form  z . w = sum_i z_i w_i  (no conjugate)."""
    return (z * w).sum(dim=-1)


def bilinear_invariant(v: torch.Tensor) -> torch.Tensor:
    """(…, 6) real -> (…,) complex :  the conserved quadratic invariant z . z.

    Re(z . z) = v^T eta v  (the so33 eta-invariant);  Im(z . z) = 2 x . y.
    """
    z = real_to_complex(v)
    return complex_bilinear(z, z)


def invariant_features(v: torch.Tensor) -> torch.Tensor:
    """(…, 6) real -> (…, 2) real :  arcsinh-normalised (Re, Im) of z . z.

    arcsinh compresses the heavy tails of quadratic invariants while staying
    smooth and sign-preserving — suitable as MLP input features.
    """
    q = bilinear_invariant(v)
    return torch.stack([torch.asinh(q.real), torch.asinh(q.imag)], dim=-1)


# ── Generators ────────────────────────────────────────────────────────────────

def so3_generators(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The so(3) basis on R^3, stacked (3, 3, 3):  L_i[j, k] = -eps_ijk."""
    L = torch.zeros(3, 3, 3, dtype=dtype)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        L[i, j, k] = -1.0
        L[i, k, j] = 1.0
    return L


def so3c_generator_stack(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The 6 realified so(3, C) generators on R^6, stacked (6, 6, 6).

    Order: [R_1, R_2, R_3, B_1, B_2, B_3] — rotations then boosts.
    Every generator A satisfies the so(3, 3) condition A^T eta + eta A = 0
    AND commutes with the complex structure Jc (tested in test_so3c_algebra).
    """
    L = so3_generators(dtype)
    Jc = complex_structure(dtype)
    gens = []
    for i in range(3):
        R = torch.zeros(DIM_R, DIM_R, dtype=dtype)
        R[:DIM_C, :DIM_C] = L[i]
        R[DIM_C:, DIM_C:] = L[i]
        gens.append(R)
    for i in range(3):
        gens.append(Jc @ gens[i])          # B_i = Jc R_i
    return torch.stack(gens, dim=0)


def cross_matrix(a: torch.Tensor) -> torch.Tensor:
    """(…, 3) (real or complex) -> (…, 3, 3) :  [a]_x with [a]_x u = a x u."""
    zero = torch.zeros_like(a[..., 0])
    a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2]
    rows = [
        torch.stack([zero, -a3, a2], dim=-1),
        torch.stack([a3, zero, -a1], dim=-1),
        torch.stack([-a2, a1, zero], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


# ── Exponential map ───────────────────────────────────────────────────────────

def expm_so3c(a: torch.Tensor, t: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Closed-form exp(t [a]_x) for a in C^3 — complex Rodrigues formula.

    Parameters
    ----------
    a : (…, 3) complex tensor — algebra element in the a <-> [a]_x convention.
    t : scalar — integration time (folded into a).

    Returns
    -------
    Q : (…, 3, 3) complex tensor in SO(3, C):  Q^T Q = I (complex bilinear
        orthogonality, no conjugation) and det Q = 1.

    Batched, differentiable, exact (no solver). Near theta = 0 the even-power
    Taylor series of sin(theta)/theta and (1-cos(theta))/theta^2 is used, so
    the map is smooth through the origin and the sqrt branch never matters.
    """
    if not torch.is_complex(a):
        a = a.to(_complex_dtype(a.dtype))
    b = a * t
    theta2 = complex_bilinear(b, b)                      # (…,) complex
    small = theta2.abs() < _SERIES_THRESHOLD

    # Series (used where |theta^2| is small; safe everywhere):
    c1_series = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    c2_series = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0

    # Closed form (guard theta = 0 with a dummy value; result discarded there):
    theta = torch.sqrt(torch.where(small, torch.ones_like(theta2), theta2))
    c1_exact = torch.sin(theta) / theta
    c2_exact = (1.0 - torch.cos(theta)) / (theta * theta)

    c1 = torch.where(small, c1_series, c1_exact)[..., None, None]
    c2 = torch.where(small, c2_series, c2_exact)[..., None, None]

    M = cross_matrix(b)
    eye = torch.eye(DIM_C, dtype=M.dtype, device=M.device).expand_as(M)
    return eye + c1 * M + c2 * (M @ M)


def complex_matrix_to_real(Q: torch.Tensor) -> torch.Tensor:
    """(…, 3, 3) complex -> (…, 6, 6) real rep:  [[Re Q, -Im Q], [Im Q, Re Q]]."""
    top = torch.cat([Q.real, -Q.imag], dim=-1)
    bot = torch.cat([Q.imag, Q.real], dim=-1)
    return torch.cat([top, bot], dim=-2)


def random_group_element(
    rot_scale: float = 1.0,
    boost_scale: float = 0.5,
    dtype: torch.dtype = torch.float64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random g in SO(3, C) as a real (6, 6) matrix, for equivariance tests.

    g = exp([rho + i beta]_x) with rho ~ N(0, rot_scale^2),
    beta ~ N(0, boost_scale^2). boost_scale plays the role of rapidity.
    """
    rho = torch.randn(3, dtype=dtype, generator=generator) * rot_scale
    beta = torch.randn(3, dtype=dtype, generator=generator) * boost_scale
    Q = expm_so3c(torch.complex(rho, beta))
    return complex_matrix_to_real(Q)
