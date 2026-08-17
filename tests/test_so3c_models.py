"""
Test — so3c benchmark models: invariance by construction, eta-blindness.

Verifies:
  • SO3CInvariantsClassifier and SO3CFlowClassifier produce IDENTICAL logits
    under random SO(3,C) transformations of the input (rotations + boosts).
  • EtaOnlyClassifier is invariant too (Re z.z is SO(3,C)-invariant) but is
    structurally constant w.r.t. changes of Im(z.z) at fixed Re(z.z).
  • The generator produces the prescribed invariants exactly.

Run:
    python tests/test_so3c_models.py
    python -m pytest tests/test_so3c_models.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.so3c_models import (
    EtaOnlyClassifier,
    SO3CFlowClassifier,
    SO3CInvariantsClassifier,
)
from benchmarks.so3c_synthetic import generate_em_invariant_dataset
from so3c.algebra import bilinear_invariant, random_group_element


def test_generator_invariants() -> None:
    X, y, _ = generate_em_invariant_dataset(n_samples=2000, seed=0,
                                            invariant_mode="im")
    q = bilinear_invariant(X)
    im0 = q.imag[y == 0]
    im1 = q.imag[y == 1]
    assert im0.min() > 0.5 - 1e-9 and im0.max() < 1.5 + 1e-9, "class-0 band broken"
    assert im1.min() > 2.5 - 1e-9 and im1.max() < 4.5 + 1e-9, "class-1 band broken"
    print(f"  ✓ generator bands: class0 Im(z.z) in [{im0.min():.2f}, {im0.max():.2f}], "
          f"class1 in [{im1.min():.2f}, {im1.max():.2f}]")


def test_classifier_invariance() -> None:
    gen = torch.Generator().manual_seed(1)
    X, _, _ = generate_em_invariant_dataset(n_samples=256, seed=1)
    torch.manual_seed(1)
    models = {
        "so3c_invariants": SO3CInvariantsClassifier(),
        "so3c_flow": SO3CFlowClassifier(),
        "eta_only": EtaOnlyClassifier(),
    }
    g = random_group_element(rot_scale=1.0, boost_scale=1.0, generator=gen)
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            base = model(X)
            transformed = model(X @ g.T)
        err = (transformed - base).abs().max().item()
        assert err < 1e-9, f"{name} not invariant: {err:.2e}"
        print(f"  ✓ {name}: logit shift under g = {err:.1e}")


def test_eta_only_blindness() -> None:
    """At fixed Re(z.z), changing Im(z.z) must not move EtaOnly's logits."""
    torch.manual_seed(2)
    model = EtaOnlyClassifier()
    # Two batches with identical Re(z.z) but different Im(z.z):
    # z = sqrt(q) * e1  =>  z.z = q.
    q_a = torch.complex(torch.linspace(-2, 2, 64, dtype=torch.float64),
                        torch.full((64,), 1.0, dtype=torch.float64))
    q_b = torch.complex(q_a.real,
                        torch.full((64,), 4.0, dtype=torch.float64))

    def embed(q):
        z = torch.zeros(64, 3, dtype=torch.complex128)
        z[:, 0] = torch.sqrt(q)
        return torch.cat([z.real, z.imag], dim=-1)

    with torch.no_grad():
        diff = (model(embed(q_a)) - model(embed(q_b))).abs().max().item()
    assert diff < 1e-12, f"eta_only saw Im(z.z): {diff:.2e}"
    print(f"  ✓ eta_only blind to Im(z.z) at fixed Re(z.z) ({diff:.1e})")


if __name__ == "__main__":
    print("\n── so3c benchmark-model tests ──")
    test_generator_invariants()
    test_classifier_invariance()
    test_eta_only_blindness()
    print("All so3c model tests passed.\n")
