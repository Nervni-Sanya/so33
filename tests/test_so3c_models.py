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


def test_covariant_set_is_equivariant_when_excited() -> None:
    """The repair: a connection built from cross products stays covariant.

    a_a = z_a x sum_b phi(invariants) z_b transforms as a -> Q a, because
    the cross product is covariant for SO(3, C) (det Q = 1) and the weights
    are invariants. Then exp(-T [Qa]_x)(Q z) = Q exp(-T [a]_x) z exactly.

    Contrast with test_trained_regime_invariance, where the same flow fed by
    a NON-covariant connection shifts the logits by ~3.6e-3. Trained on
    20k jets the two differ sharply under a rapidity-2 boost: the broken
    model drops 0.095 AUC, this one drops 0.0000 and is more accurate
    besides (0.9565 vs 0.9424).

    The reference vector must be an invariant-WEIGHTED sum: the plain total
    vanishes for this lift, since every z_b shares the leg P and
    sum_b bivec(p_b, P) = bivec(P, P) = 0.
    """
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    gen = torch.Generator().manual_seed(21)
    x = _random_jets(6, 12, gen)

    torch.manual_seed(7)
    m = build_model("so3c_covariant_set", in_features=4, out_features=2,
                    representation="constituents")
    m.eval()
    torch.manual_seed(7)
    identity = build_model("so3c_covariant_set", in_features=4, out_features=2,
                           representation="constituents").eval()
    with torch.no_grad():
        m.phi[-1].weight.normal_(0, 0.5, generator=gen)
        m.phi[-1].bias.normal_(0, 0.5, generator=gen)
        # The flow must actually do something, or equivariance is vacuous.
        moved = (m(x) - identity(x)).abs().max().item()
    assert moved > 1e-2, f"flow is inert, test would be vacuous: {moved:.2e}"

    for boost in (0.5, 1.0, 2.0):
        L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=boost,
                                   generator=gen)
        x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)
        with torch.no_grad():
            err = (m(x_t) - m(x)).abs().max().item()
        assert err < 1e-6, f"covariant model lost equivariance at {boost}: {err:.2e}"
        print(f"  ✓ so3c_covariant_set equivariant at boost {boost} ({err:.1e})")
    print(f"    (flow displaces logits by {moved:.2f}, so this is not vacuous)")


def test_message_set_is_equivariant_when_excited() -> None:
    """Multi-round covariant message passing keeps exact equivariance.

    Each round adds three things that could break it and do not:

    * a scalar channel h, updated from bilinear invariants only, so h is
      itself invariant and may be fed back into the edge features;
    * a complex channel mixing M, which commutes with Q because Q is
      complex linear;
    * a second and third application of the flow, whose connection is
      rebuilt covariantly from the already-rotated states.

    As in the single-round test, the weight heads are zero-initialised, so
    the flow must be excited by hand or the assertion would hold vacuously
    for the identity map.

    The tolerance is a float64 tolerance, and the generator is pinned for a
    reason. `random_lorentz_pair(boost_scale=s)` draws each beta component
    from N(0, s), so a draw at s = 2 can reach rapidity 5, where the lift's
    E_p q - E_q p cancels away real digits before the states are normalised.
    Errors of 1e-5 are reachable there in double precision. If this ever
    fails on a new draw, check the rapidity actually sampled before
    concluding the symmetry broke.
    """
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    gen = torch.Generator().manual_seed(21)
    x = _random_jets(6, 12, gen)

    torch.manual_seed(7)
    m = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents").eval()
    torch.manual_seed(7)
    identity = build_model("so3c_message_set", in_features=4, out_features=2,
                           representation="constituents").eval()
    with torch.no_grad():
        for r in range(m.rounds):
            m.w_head[r].weight.normal_(0, 0.5, generator=gen)
            m.w_head[r].bias.normal_(0, 0.5, generator=gen)
            m.mix[r].normal_(0, 0.3, generator=gen)
        base = m(x)
        moved = (base - identity(x)).abs().max().item()
    assert moved > 1e-2, f"flow is inert, test would be vacuous: {moved:.2e}"

    for boost in (0.5, 1.0, 2.0, 3.0):
        L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=boost,
                                   generator=gen)
        x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)
        with torch.no_grad():
            err = (m(x_t) - base).abs().max().item()
        assert err < 1e-6, f"message model lost equivariance at {boost}: {err:.2e}"
        print(f"  ✓ so3c_message_set equivariant at boost {boost} ({err:.1e})")
    print(f"    (3 rounds displace logits by {moved:.2f}, so this is not vacuous)")


def test_message_set_neighbour_graph_is_equivariant() -> None:
    """kNN sparsification preserves equivariance because the ranking key is
    an invariant: |Re z_a.z_b| does not move under z -> Qz, so every jet
    keeps the SAME neighbour index set in any frame. A key built from a
    frame-dependent quantity (angular distance, pT ordering) would silently
    reshuffle edges under a boost and break the symmetry with no error.
    """
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    gen = torch.Generator().manual_seed(5)
    x = _random_jets(6, 16, gen)

    torch.manual_seed(3)
    m = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"neighbors": 6}).eval()
    torch.manual_seed(3)
    identity = build_model("so3c_message_set", in_features=4, out_features=2,
                           representation="constituents",
                           so3c_kwargs={"neighbors": 6}).eval()
    with torch.no_grad():
        for r in range(m.rounds):
            m.w_head[r].weight.normal_(0, 0.5, generator=gen)
            m.w_head[r].bias.normal_(0, 0.5, generator=gen)
        base = m(x)
        moved = (base - identity(x)).abs().max().item()
    assert moved > 1e-2, f"flow is inert, test would be vacuous: {moved:.2e}"

    for boost in (1.0, 2.0):
        L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=boost,
                                   generator=gen)
        x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)
        with torch.no_grad():
            err = (m(x_t) - base).abs().max().item()
        assert err < 1e-6, f"kNN model lost equivariance at {boost}: {err:.2e}"
        print(f"  ✓ so3c_message_set(kNN) equivariant at boost {boost} ({err:.1e})")


def test_message_set_beams_are_covariant_inputs() -> None:
    """With beam particles the architecture stays exactly Lorentz covariant.

    LorentzNet and PELICAN append two beam particles (1, 0, 0, +-1) so the
    network can see lab-frame energies and transverse momenta. They enter as
    ordinary 4-vectors, so moving the jet AND the beams by one Lorentz matrix
    must leave the logits unchanged. Moving the jet alone must change them --
    otherwise the beams are not wired in -- except for a rotation about the
    beam axis, which fixes both beam vectors.
    """
    from benchmarks.models import build_model
    from so3c.lift import lorentz_matrix, random_lorentz_pair

    gen = torch.Generator().manual_seed(33)
    x = _random_jets(6, 12, gen)
    torch.manual_seed(5)
    m = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True}).eval()
    with torch.no_grad():
        for r in range(m.rounds):
            m.w_head[r].weight.normal_(0, 0.5, generator=gen)
            m.w_head[r].bias.normal_(0, 0.5, generator=gen)
            m.mix[r].normal_(0, 0.3, generator=gen)
        beams = m.beam_p4.clone()
        base = m(x)

        L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=1.0, generator=gen)
        x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)
        jet_only = (m(x_t) - base).abs().max().item()
        m.beam_p4.copy_(beams @ L.T)
        joint = (m(x_t) - base).abs().max().item()
        m.beam_p4.copy_(beams)

        theta = torch.tensor(0.7, dtype=torch.float64)
        zero = torch.zeros((), dtype=torch.float64)
        R = lorentz_matrix(torch.stack([zero, zero, theta]),
                           torch.zeros(3, dtype=torch.float64))
        x_r = torch.cat([x[..., :4] @ R.T, x[..., 4:]], dim=-1)
        z_rot = (m(x_r) - base).abs().max().item()

    assert joint < 1e-6, f"jet and beams moved together changed the logits: {joint:.2e}"
    assert jet_only > 1e-3, f"moving the jet alone changed nothing: beams not wired in ({jet_only:.2e})"
    assert z_rot < 1e-6, f"a rotation about the beam axis changed the logits: {z_rot:.2e}"

    m.train()
    m(x).sum().backward()
    assert m.h_init.weight.grad is not None and m.h_init.weight.grad.abs().sum() > 0
    print(f"  ✓ beams: joint {joint:.1e}, jet only {jet_only:.1e}, z-rotation {z_rot:.1e}")


def test_message_set_bundle_defaults_are_unchanged() -> None:
    """The four new switches default to the model the headline was measured with.

    Parameter counts are pinned to the published configurations, and a model
    built with the defaults written out explicitly produces identical logits.
    """
    from benchmarks.models import build_model

    counts = {
        (): 13862,
        (("beams", True),): 15038,
        (("beams", True), ("channels", 8)): 22834,
    }
    for kw, expected in counts.items():
        m = build_model("so3c_message_set", in_features=4, out_features=2,
                        representation="constituents", so3c_kwargs=dict(kw))
        n = sum(q.numel() for q in m.parameters())
        assert n == expected, f"{dict(kw)}: {n} params, expected {expected}"

    gen = torch.Generator().manual_seed(8)
    x = _random_jets(4, 10, gen)
    torch.manual_seed(2)
    a = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True}).eval()
    torch.manual_seed(2)
    b = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True, "mass_input": True,
                                 "self_edges": True, "relnorm_edge": False,
                                 "falpha": 0}).eval()
    with torch.no_grad():
        assert torch.equal(a(x), b(x)), "explicit defaults changed the logits"
    print("  ✓ bundle defaults reproduce the published models")


def test_message_set_bundle_is_exactly_covariant() -> None:
    """All four switches on, with and without beams: still exactly covariant.

    They only change which invariants are computed (no m^2, d_ab added, no
    self-edge) and how they are compressed (f_alpha), so moving the jet --
    and the beams, when present -- by one Lorentz matrix must leave the logits
    unchanged. Each switch must also actually change the output, or the test
    would pass for a switch that is never read.
    """
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    bundle = {"mass_input": False, "self_edges": False,
              "relnorm_edge": True, "falpha": 3}
    gen = torch.Generator().manual_seed(44)
    x = _random_jets(6, 12, gen)
    L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=1.0, generator=gen)
    x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)

    for beams in (False, True):
        torch.manual_seed(9)
        m = build_model("so3c_message_set", in_features=4, out_features=2,
                        representation="constituents",
                        so3c_kwargs=dict(bundle, beams=beams)).eval()
        with torch.no_grad():
            for r in range(m.rounds):
                m.w_head[r].weight.normal_(0, 0.5, generator=gen)
                m.w_head[r].bias.normal_(0, 0.5, generator=gen)
                m.mix[r].normal_(0, 0.3, generator=gen)
            base = m(x)
            if beams:
                beams0 = m.beam_p4.clone()
                m.beam_p4.copy_(beams0 @ L.T)
            err = (m(x_t) - base).abs().max().item()
            if beams:
                m.beam_p4.copy_(beams0)
        assert err < 1e-6, f"bundle (beams={beams}) lost covariance: {err:.2e}"
        print(f"  ✓ bundle, beams={beams}: covariant ({err:.1e})")

    gen = torch.Generator().manual_seed(45)
    x = _random_jets(6, 12, gen)
    for key, value in bundle.items():
        torch.manual_seed(3)
        ref = build_model("so3c_message_set", in_features=4, out_features=2,
                          representation="constituents",
                          so3c_kwargs={"beams": True}).eval()
        torch.manual_seed(3)
        alt = build_model("so3c_message_set", in_features=4, out_features=2,
                          representation="constituents",
                          so3c_kwargs={"beams": True, key: value}).eval()
        ref_sd, alt_sd = ref.state_dict(), alt.state_dict()
        shared = {k: v for k, v in ref_sd.items()
                  if k in alt_sd and alt_sd[k].shape == v.shape}
        alt.load_state_dict(shared, strict=False)
        with torch.no_grad():
            for mod in (ref, alt):
                g = torch.Generator().manual_seed(46)
                for r in range(mod.rounds):
                    mod.w_head[r].weight.normal_(0, 0.5, generator=g)
                    mod.w_head[r].bias.normal_(0, 0.5, generator=g)
            moved = (ref(x) - alt(x)).abs().max().item()
        assert moved > 1e-6, f"switch {key}={value} changed nothing"
        print(f"  ✓ switch {key}={value} is wired in ({moved:.1e})")


def test_message_set_falpha_learns() -> None:
    """The f_alpha exponents receive gradient, and f_alpha is odd and finite."""
    from benchmarks.models import build_model
    from benchmarks.so3c_models import _SignedFAlpha

    f = _SignedFAlpha(3, torch.float64)
    xs = torch.tensor([-50.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 50.0], dtype=torch.float64)
    ys = f(xs.unsqueeze(-1))
    assert torch.isfinite(ys).all()
    assert torch.allclose(f((-xs).unsqueeze(-1)), -ys)
    assert torch.all(ys[3] == 0)

    gen = torch.Generator().manual_seed(12)
    x = _random_jets(4, 10, gen)
    torch.manual_seed(4)
    m = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True, "falpha": 3})
    m(x).sum().backward()
    g = m.falpha_embed.alpha.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"  ✓ f_alpha gradient {g.abs().sum().item():.2e}")


def test_message_set_vector_channel_is_exactly_covariant() -> None:
    """The vector channel keeps exact covariance, with and without beams.

    v_a starts as p_a, edge invariants are Minkowski products, the update is an
    invariant-weighted linear combination of 4-vectors, and the readout pools
    Minkowski products. Moving the jet -- and the beams, when present -- by one
    Lorentz matrix must leave the logits unchanged. Every head is excited,
    including the zero-initialised vector heads, so the check cannot pass on an
    identity update.
    """
    from benchmarks.models import build_model
    from so3c.lift import random_lorentz_pair

    gen = torch.Generator().manual_seed(61)
    x = _random_jets(6, 12, gen)
    L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=1.0, generator=gen)
    x_t = torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)

    for beams in (False, True):
        torch.manual_seed(13)
        m = build_model("so3c_message_set", in_features=4, out_features=2,
                        representation="constituents",
                        so3c_kwargs={"vector_channel": True, "beams": beams}).eval()
        torch.manual_seed(13)
        still = build_model("so3c_message_set", in_features=4, out_features=2,
                            representation="constituents",
                            so3c_kwargs={"vector_channel": True, "beams": beams}).eval()
        with torch.no_grad():
            for r in range(m.rounds):
                m.w_head[r].weight.normal_(0, 0.5, generator=gen)
                m.w_head[r].bias.normal_(0, 0.5, generator=gen)
                m.mix[r].normal_(0, 0.3, generator=gen)
                m.v_head[r].weight.normal_(0, 0.5, generator=gen)
                m.v_head[r].bias.normal_(0, 0.5, generator=gen)
            still.load_state_dict({k: v for k, v in m.state_dict().items()
                                   if not k.startswith("v_head")}, strict=False)
            base = m(x)
            vector_moved = (base - still(x)).abs().max().item()
            if beams:
                beams0 = m.beam_p4.clone()
                m.beam_p4.copy_(beams0 @ L.T)
            err = (m(x_t) - base).abs().max().item()
            if beams:
                m.beam_p4.copy_(beams0)
        assert vector_moved > 1e-4, (
            f"the vector update changed nothing (beams={beams}): {vector_moved:.2e}")
        assert err < 1e-6, f"vector channel (beams={beams}) lost covariance: {err:.2e}"
        print(f"  ✓ vector channel, beams={beams}: covariant ({err:.1e}), "
              f"update moves logits by {vector_moved:.1e}")


def test_message_set_vector_channel_trains() -> None:
    """Parameters and gradients: the defaults are untouched and v_head learns.

    With the switch off the model has no vector heads and its parameter count
    is the published one. With it on, a backward pass reaches every vector head
    once the zero-init head of each round is nudged off zero -- otherwise the
    first backward could legitimately leave later heads at zero gradient.
    """
    from benchmarks.models import build_model

    off = build_model("so3c_message_set", in_features=4, out_features=2,
                      representation="constituents",
                      so3c_kwargs={"beams": True, "channels": 8})
    assert sum(q.numel() for q in off.parameters()) == 22834
    assert len(off.v_head) == 0

    gen = torch.Generator().manual_seed(62)
    x = _random_jets(4, 10, gen)
    torch.manual_seed(14)
    m = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True, "channels": 8,
                                 "vector_channel": True})
    extra = sum(q.numel() for q in m.parameters()) - 22834
    with torch.no_grad():
        for r in range(m.rounds):
            m.v_head[r].weight.normal_(0, 0.1, generator=gen)
    m(x).sum().backward()
    for r in range(m.rounds):
        g = m.v_head[r].weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, r
    print(f"  ✓ vector channel adds {extra} parameters; every vector head learns")

    import pytest
    with pytest.raises(ValueError):
        build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"vector_channel": True, "neighbors": 4})


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
    test_covariant_set_is_equivariant_when_excited()
    test_message_set_is_equivariant_when_excited()
    test_message_set_neighbour_graph_is_equivariant()
    test_message_set_beams_are_covariant_inputs()
    test_message_set_bundle_defaults_are_unchanged()
    test_message_set_bundle_is_exactly_covariant()
    test_message_set_falpha_learns()
    test_message_set_vector_channel_is_exactly_covariant()
    test_message_set_vector_channel_trains()
    print("All so3c model tests passed.\n")
