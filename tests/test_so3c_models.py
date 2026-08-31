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


def test_tabular_factory_wiring() -> None:
    """build_model must return working flat tabular models for so3c names:
    forward on float32 (B, 28), finite backward, regularization contract."""
    from benchmarks.models import build_model, SO3C_MODELS

    torch.manual_seed(3)
    x = torch.randn(16, 28, dtype=torch.float32)
    y = torch.randint(0, 2, (16,))
    for name in SO3C_MODELS:
        model = build_model(name, in_features=28, out_features=2)
        logits = model(x)
        assert logits.shape == (16, 2), f"{name}: bad output shape"
        loss = torch.nn.functional.cross_entropy(logits, y) + model.regularization_loss()
        loss.backward()
        for pname, p in model.named_parameters():
            assert p.grad is not None, f"{name}: missing grad {pname}"
            assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad {pname}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {name}: forward/backward OK ({n_params} params)")

    # Constituents representation must be rejected explicitly.
    try:
        build_model("so3c", in_features=4, out_features=2,
                    representation="constituents")
    except ValueError:
        print("  ✓ constituents representation rejected for so3c")
    else:
        raise AssertionError("so3c must reject representation='constituents'")


def _random_jets(B: int, K: int, gen: torch.Generator) -> torch.Tensor:
    """(B, K, 5) physical-ish constituent sets with a ragged mask."""
    p = torch.randn(B, K, 3, dtype=torch.float64, generator=gen)
    m = torch.rand(B, K, dtype=torch.float64, generator=gen) * 0.1
    E = torch.sqrt(m * m + p.pow(2).sum(-1))
    mask = (torch.arange(K).unsqueeze(0)
            < torch.randint(3, K + 1, (B, 1), generator=gen)).double()
    x = torch.cat([E.unsqueeze(-1), p, mask.unsqueeze(-1)], dim=-1)
    return x * torch.cat([mask.unsqueeze(-1).expand(B, K, 4),
                          torch.ones(B, K, 1, dtype=torch.float64)], dim=-1)


def test_set_factory_wiring() -> None:
    """_build_deepsets branches for the so3c set family: forward/backward."""
    from benchmarks.models import build_model, SO3C_SET_MODELS

    torch.manual_seed(4)
    x = _random_jets(4, 8, torch.Generator().manual_seed(4))
    y = torch.randint(0, 2, (4,))
    for name in SO3C_SET_MODELS:
        model = build_model(name, in_features=4, out_features=2,
                            representation="constituents")
        logits = model(x)
        assert logits.shape == (4, 2), f"{name}: bad output shape"
        loss = torch.nn.functional.cross_entropy(logits, y) + model.regularization_loss()
        loss.backward()
        assert all(torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.grad is not None), \
            f"{name}: non-finite grads"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {name}: forward/backward OK ({n_params} params)")


def test_set_lorentz_invariance() -> None:
    """Logits must be invariant under Lorentz transformations of the jet —
    the defining property of the whole set family. Exact-flow models to
    near machine precision, the ODE model to solver tolerance."""
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    gen = torch.Generator().manual_seed(5)
    x = _random_jets(6, 10, gen)
    L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=0.5, generator=gen)
    p4_t = x[..., :4] @ L.T
    x_t = torch.cat([p4_t, x[..., 4:]], dim=-1)

    tolerances = {
        "so3c_invariant_set": 1e-9,
        "so3c_equivariant_set": 1e-9,
        "so3c_interaction_set": 1e-3,
    }
    for name, tol in tolerances.items():
        torch.manual_seed(6)
        model = build_model(name, in_features=4, out_features=2,
                            representation="constituents")
        model.eval()
        with torch.no_grad():
            base = model(x)
            transformed = model(x_t)
        err = (transformed - base).abs().max().item()
        assert err < tol, f"{name}: logits moved under Lorentz: {err:.2e}"
        print(f"  ✓ {name}: Lorentz-invariant logits ({err:.1e})")


def _excite_connection(model, seed: int = 0) -> None:
    """Give every flow in the model a non-zero connection.

    Freshly built models zero-initialise the connection MLP so the flow
    starts as the identity map. An invariance test on a fresh model
    therefore only exercises the readout -- the flow is a no-op and cannot
    break anything. Training moves the connection away from zero, so the
    property has to be checked there.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in model.modules():
            metric = getattr(mod, "metric", None)
            if metric is not None:
                metric.net[-1].weight.normal_(0, 0.5, generator=gen)
                metric.net[-1].bias.normal_(0, 0.5, generator=gen)
        if hasattr(model, "channel_weights"):
            model.channel_weights.normal_(0, 0.5, generator=gen)


def test_trained_regime_invariance() -> None:
    """Which set models stay Lorentz-invariant once the flow is switched on.

    so3c_invariant_set is invariant by construction: it reads pooled
    invariants of the bivector lift and has no flow.

    so3c_equivariant_set is NOT. Its connection a(s) is built from
    invariants, so under z -> Qz the invariants are unchanged, a is
    unchanged, and the flow applies the same rotation in the new frame
    instead of the conjugated one it would need. Per-particle invariants
    survive (an antisymmetric connection conserves z.z), but the pairwise
    z_a.z_b terms the readout depends on do not. Measured cost: a trained
    model loses 0.095 AUC under a rapidity-2 boost while the invariant
    model loses nothing.

    This test pins the behaviour so the distinction cannot be lost again;
    it is not an endorsement of the broken case.
    """
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    gen = torch.Generator().manual_seed(11)
    x = _random_jets(8, 12, gen)
    L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=0.5, generator=gen)
    x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)

    def shift(name):
        torch.manual_seed(7)
        m = build_model(name, in_features=4, out_features=2,
                        representation="constituents")
        m.eval()
        _excite_connection(m)
        with torch.no_grad():
            return (m(x_t) - m(x)).abs().max().item()

    inv = shift("so3c_invariant_set")
    assert inv < 1e-9, f"so3c_invariant_set must stay invariant: {inv:.2e}"
    print(f"  ✓ so3c_invariant_set invariant with a live flow ({inv:.1e})")

    eq = shift("so3c_equivariant_set")
    assert eq > 1e-5, (
        "so3c_equivariant_set is expected to BREAK invariance once its "
        f"connection is non-zero; got {eq:.2e}. If this now holds, the "
        "connection was made covariant and the docs must be updated."
    )
    print(f"  ✓ so3c_equivariant_set breaks invariance as documented ({eq:.1e})")


def test_interaction_set_is_equivariant_when_excited() -> None:
    """SO3CInteraction builds its connection from particle bivectors, which
    transform in the adjoint, so it stays equivariant with a live flow --
    the property so3c_equivariant_set lacks."""
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    gen = torch.Generator().manual_seed(12)
    x = _random_jets(4, 10, gen)
    L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=0.5, generator=gen)
    x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)

    torch.manual_seed(7)
    m = build_model("so3c_interaction_set", in_features=4, out_features=2,
                    representation="constituents")
    m.eval()
    with torch.no_grad():
        last = m.interaction.phi[-1]
        last.weight.normal_(0, 0.5, generator=gen)
        last.bias.normal_(0, 0.5, generator=gen)
    with torch.no_grad():
        err = (m(x_t) - m(x)).abs().max().item()
    assert err < 1e-5, f"interaction model lost equivariance: {err:.2e}"
    print(f"  ✓ so3c_interaction_set equivariant with a live flow ({err:.1e})")


if __name__ == "__main__":
    print("\n── so3c benchmark-model tests ──")
    test_generator_invariants()
    test_classifier_invariance()
    test_eta_only_blindness()
    test_tabular_factory_wiring()
    test_set_factory_wiring()
    test_set_lorentz_invariance()
    test_trained_regime_invariance()
    test_interaction_set_is_equivariant_when_excited()
    print("All so3c model tests passed.\n")
