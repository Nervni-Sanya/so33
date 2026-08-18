"""
Test — bivector lift: representation correctness, invariants, jet lift.

Verifies:
  • lorentz_matrix produces genuine SO(3,1) elements (preserves +--- metric).
  • THE CRITICAL correspondence: bivec(Λp, Λq) = expm_so3c(rho + i beta) @
    bivec(p, q) with the same (rho, beta) — the lift intertwines the 4-vector
    rep with our C^3 rep to machine precision, rotations AND boosts.
  • z . z is Lorentz-invariant; Im(z . z) = 0 for simple bivectors and
    Im(z_a . z_b) = 0 for bivectors sharing a leg (documented structural
    facts the readout design relies on).
  • jet_bivectors: padded entries give z = 0 and do not affect real ones.

Run:
    python tests/test_so3c_lift.py
    python -m pytest tests/test_so3c_lift.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from so3c.algebra import complex_bilinear, expm_so3c
from so3c.lift import (
    bivector_lift,
    jet_bivectors,
    lorentz_matrix,
    minkowski_inner,
    random_lorentz_pair,
)

G_MINK = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))


def _random_p4(n: int, gen: torch.Generator) -> torch.Tensor:
    """Massive forward 4-momenta: E from mass shell, |p| ~ O(1)."""
    p = torch.randn(n, 3, dtype=torch.float64, generator=gen)
    m = torch.rand(n, dtype=torch.float64, generator=gen) + 0.5
    E = torch.sqrt(m * m + p.pow(2).sum(-1))
    return torch.cat([E.unsqueeze(-1), p], dim=-1)


def test_lorentz_matrix_preserves_metric() -> None:
    gen = torch.Generator().manual_seed(0)
    for boost in (0.0, 0.5, 1.5):
        L, _ = random_lorentz_pair(boost_scale=boost, generator=gen)
        err = (L.T @ G_MINK @ L - G_MINK).abs().max().item()
        assert err < 1e-10, f"not in SO(3,1) at boost={boost}: {err:.2e}"
    print("  ✓ lorentz_matrix preserves the +--- metric")


def test_representation_correspondence() -> None:
    gen = torch.Generator().manual_seed(1)
    p = _random_p4(64, gen)
    q = _random_p4(64, gen)
    for boost in (0.0, 0.5, 1.0):
        L, a = random_lorentz_pair(rot_scale=1.0, boost_scale=boost,
                                   generator=gen)
        Q = expm_so3c(a)
        lhs = bivector_lift(p @ L.T, q @ L.T)              # lift after transform
        rhs = (Q @ bivector_lift(p, q).unsqueeze(-1)).squeeze(-1)
        rel = ((lhs - rhs).abs().max() / rhs.abs().max()).item()
        assert rel < 1e-10, f"intertwiner broken (boost={boost}): {rel:.2e}"
        print(f"  ✓ bivec(Λp, Λq) = Q bivec(p, q) at boost_scale={boost} ({rel:.1e})")


def test_invariants() -> None:
    gen = torch.Generator().manual_seed(2)
    p = _random_p4(64, gen)
    q = _random_p4(64, gen)
    z = bivector_lift(p, q)
    zz = complex_bilinear(z, z)

    # Simple-bivector identity: pseudoscalar part vanishes.
    assert zz.imag.abs().max().item() < 1e-10, "Im(z.z) != 0 for simple bivector"

    # Lorentz invariance of z . z.
    L, _ = random_lorentz_pair(boost_scale=1.0, generator=gen)
    z_t = bivector_lift(p @ L.T, q @ L.T)
    zz_t = complex_bilinear(z_t, z_t)
    err = (zz_t - zz).abs().max().item()
    assert err < 1e-8, f"z.z not invariant: {err:.2e}"

    # Closed form: Re(z.z) = (p.q)^2 - p^2 q^2 in Minkowski products.
    expected = minkowski_inner(p, q) ** 2 - minkowski_inner(p, p) * minkowski_inner(q, q)
    err_cf = (zz.real - expected).abs().max().item()
    assert err_cf < 1e-8, f"Re(z.z) closed form mismatch: {err_cf:.2e}"

    # Shared-leg cross pseudoscalar vanishes: Im(bivec(p,P) . bivec(q,P)) = 0.
    P = _random_p4(64, gen)
    cross = complex_bilinear(bivector_lift(p, P), bivector_lift(q, P))
    assert cross.imag.abs().max().item() < 1e-9, "shared-leg cross-Im != 0"
    print("  ✓ invariants: Im(z.z)=0, Lorentz-invariance, closed form, shared-leg Im=0")


def test_jet_bivectors_masking() -> None:
    gen = torch.Generator().manual_seed(3)
    p4 = _random_p4(2 * 5, gen).reshape(2, 5, 4)
    mask = torch.tensor([[1., 1., 1., 0., 0.],
                         [1., 1., 1., 1., 1.]], dtype=torch.float64)
    z = jet_bivectors(p4, mask)
    assert z.shape == (2, 5, 3)
    assert z[0, 3:].abs().max().item() < 1e-12, "padded entries must be zero"

    # Real particles must match the truncated computation exactly.
    z_trunc = jet_bivectors(p4[:1, :3], torch.ones(1, 3, dtype=torch.float64))
    err = (z[0, :3] - z_trunc[0]).abs().max().item()
    assert err < 1e-12, f"padding leaked into real particles: {err:.2e}"
    print("  ✓ jet_bivectors: padding gives z=0 and does not leak")


if __name__ == "__main__":
    print("\n── so3c bivector-lift tests ──")
    test_lorentz_matrix_preserves_metric()
    test_representation_correspondence()
    test_invariants()
    test_jet_bivectors_masking()
    print("All lift tests passed.\n")
