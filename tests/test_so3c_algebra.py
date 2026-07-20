"""
Test — so3c algebra: generators, embedding into so(3,3), exponential map.

Verifies:
  • Lorentz commutation relations [R,R]=eps R, [R,B]=eps B, [B,B]=-eps R.
  • Every generator satisfies A^T eta + eta A = 0  (so(3,C) ⊂ so(3,3))
    and commutes with the complex structure Jc.
  • ETA matches so33.basis.ETA (shared invariant Re(z.z) = v^T eta v).
  • expm_so3c agrees with torch.matrix_exp, including the small-theta
    series branch; results lie in SO(3,C): Q^T Q = I, det Q = 1.
  • The real 6x6 representation is consistent with complex 3x3 action.
  • z.z is invariant under random group elements (rotations AND boosts).

Run:
    python tests/test_so3c_algebra.py
    python -m pytest tests/test_so3c_algebra.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from so3c.algebra import (
    ETA,
    bilinear_invariant,
    complex_matrix_to_real,
    complex_structure,
    complex_to_real,
    cross_matrix,
    expm_so3c,
    random_group_element,
    real_to_complex,
    so3c_generator_stack,
)
from so33.basis import ETA as ETA_SO33

TOL = 1e-12
EPS = torch.zeros(3, 3, 3, dtype=torch.float64)
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[_i, _j, _k] = 1.0
    EPS[_i, _k, _j] = -1.0


def test_commutation_relations() -> None:
    G = so3c_generator_stack()
    R, B = G[:3], G[3:]

    def comm(X, Y):
        return X @ Y - Y @ X

    for i in range(3):
        for j in range(3):
            rr = comm(R[i], R[j]) - sum(EPS[i, j, k] * R[k] for k in range(3))
            rb = comm(R[i], B[j]) - sum(EPS[i, j, k] * B[k] for k in range(3))
            bb = comm(B[i], B[j]) + sum(EPS[i, j, k] * R[k] for k in range(3))
            assert rr.abs().max() < TOL, f"[R{i},R{j}] violated"
            assert rb.abs().max() < TOL, f"[R{i},B{j}] violated"
            assert bb.abs().max() < TOL, f"[B{i},B{j}] violated"
    print("  ✓ Lorentz commutation relations hold")


def test_embedding_into_so33() -> None:
    assert torch.equal(ETA, ETA_SO33), "so3c ETA must equal so33 ETA"

    G = so3c_generator_stack()
    Jc = complex_structure()
    eta = torch.diag(ETA)
    for k, A in enumerate(G):
        cond = A.T @ eta + eta @ A
        assert cond.abs().max() < TOL, f"generator {k} not in so(3,3)"
        commJ = A @ Jc - Jc @ A
        assert commJ.abs().max() < TOL, f"generator {k} does not commute with Jc"
    print("  ✓ all 6 generators lie in so(3,3) and commute with Jc")


def test_expm_matches_matrix_exp() -> None:
    gen = torch.Generator().manual_seed(0)
    a = torch.complex(
        torch.randn(64, 3, dtype=torch.float64, generator=gen),
        torch.randn(64, 3, dtype=torch.float64, generator=gen) * 0.5,
    )
    Q = expm_so3c(a)
    Q_ref = torch.matrix_exp(cross_matrix(a))
    err = (Q - Q_ref).abs().max().item()
    assert err < 1e-10, f"Rodrigues vs matrix_exp: {err:.2e}"

    # Small-theta series branch.
    a_small = a * 1e-6
    err_small = (expm_so3c(a_small) - torch.matrix_exp(cross_matrix(a_small))).abs().max().item()
    assert err_small < 1e-12, f"series branch error: {err_small:.2e}"

    # Group properties: Q^T Q = I (complex bilinear orthogonality), det = 1.
    eye = torch.eye(3, dtype=Q.dtype)
    ortho = (Q.transpose(-1, -2) @ Q - eye).abs().max().item()
    det = (torch.linalg.det(Q) - 1.0).abs().max().item()
    assert ortho < 1e-10, f"Q^T Q != I: {ortho:.2e}"
    assert det < 1e-10, f"det Q != 1: {det:.2e}"
    print(f"  ✓ expm_so3c: vs matrix_exp {err:.1e}, orthogonality {ortho:.1e}, det {det:.1e}")


def test_real_representation_consistency() -> None:
    gen = torch.Generator().manual_seed(1)
    a = torch.complex(
        torch.randn(8, 3, dtype=torch.float64, generator=gen),
        torch.randn(8, 3, dtype=torch.float64, generator=gen) * 0.5,
    )
    v = torch.randn(8, 6, dtype=torch.float64, generator=gen)
    Q = expm_so3c(a)
    g = complex_matrix_to_real(Q)

    via_real = (g @ v.unsqueeze(-1)).squeeze(-1)
    via_complex = complex_to_real((Q @ real_to_complex(v).unsqueeze(-1)).squeeze(-1))
    err = (via_real - via_complex).abs().max().item()
    assert err < 1e-12, f"real/complex rep mismatch: {err:.2e}"
    print(f"  ✓ real 6x6 rep consistent with complex 3x3 action ({err:.1e})")


def test_invariance_of_bilinear_form() -> None:
    gen = torch.Generator().manual_seed(2)
    v = torch.randn(32, 6, dtype=torch.float64, generator=gen)
    q0 = bilinear_invariant(v)
    for boost in (0.0, 0.5, 1.5):
        g = random_group_element(rot_scale=1.0, boost_scale=boost, generator=gen)
        q1 = bilinear_invariant(v @ g.T)
        err = (q1 - q0).abs().max().item()
        assert err < 1e-9, f"z.z not invariant (boost={boost}): {err:.2e}"
    print("  ✓ z.z invariant under rotations and boosts")


if __name__ == "__main__":
    print("\n── so3c algebra tests ──")
    test_commutation_relations()
    test_embedding_into_so33()
    test_expm_matches_matrix_exp()
    test_real_representation_consistency()
    test_invariance_of_bilinear_form()
    print("All so3c algebra tests passed.\n")
