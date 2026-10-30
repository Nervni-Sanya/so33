"""
PFN / EFN baselines at ~5k parameters (benchmarks/pfn.py).

Verifies, on toy massless jets (benchmarks.synthetic), with randomly
initialised models whose biases are randomised too (at initialisation they are
zero, which makes every layer positively homogeneous):
  • Permutation invariance: shuffling the real constituents of every jet, or
    all K slots (padding interleaved), leaves the PFN and EFN logits unchanged.
  • Padding invariance: appending all-zero rows changes nothing, and whatever
    sits in a masked row (random momenta, NaN, inf) is ignored.
  • EFN IRC safety (1e-10 in float64): adding a zero-energy particle (a real,
    mask-1 row with p = 0, or pT = 0 along the beam) and splitting the hardest
    or a soft constituent into two collinear ones with pT fractions f and
    1 - f leave the EFN logits unchanged; the same perturbations change the
    PFN logits by O(1), so the check has teeth.
  • Both models are invariant under a global rescaling of the inputs, a common
    longitudinal boost (y -> y + dy) and a common rotation about the beam
    (phi -> phi + a), including rotations that put every jet across the
    phi = +-pi branch cut.
  • The kinematics on a hand-made jet: pT, rapidity (massless, massive, E a
    hair below |p|, E < |pz|, zero momentum) and the azimuth relative to the
    jet axis; masked rows give zeros whatever they contain.
  • The preprocessing: z = pT / sum pT sums to 1, wrap_phi maps into
    (-pi, pi], the coordinates are centred, aligned with the principal axis
    (sum z u v = 0, sum z u^2 >= sum z v^2) and reflected (at least half the
    pT in each positive half-plane), padding gets z = 0 and (0, 0), and the
    output equals energyflow's center_ptyphims("ptscheme") -> rotate_ptyphims
    -> reflect_ptyphims (skipped without energyflow); float32 inputs, where
    rounding puts E below |p| for many massless constituents, stay finite.
  • Parameter counts through benchmarks.models.build_model: PFN 5,106 and EFN
    5,082 trainable parameters (both in [4500, 5500]) with Phi 24-24-64 and
    F 24-24-24, ReLU after every layer, family "baseline_set"; the paper's
    sizes give 82,458 / 82,358. Initialisation as energyflow / Keras (zero
    biases, He-uniform Phi and F, Glorot-uniform output layer).
  • No-grad evaluation in chunks gives the same logits as one batch.
  • A short training run (massless_jet_split(3000, 32), 4 epochs,
    benchmarks.train.train_classifier) reaches test AUC >= 0.85 for both
    models (the toy jet-mass AUC is ~0.94).

Run:
    python -m pytest tests/test_pfn.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import numpy as np
import pytest
import torch

from benchmarks.models import build_model, count_parameters, is_set_model, model_family
from benchmarks.pfn import (
    EnergyFlowNetwork, ParticleFlowNetwork, PAPER_F_SIZES, PAPER_PHI_SIZES,
    constituent_kinematics, preprocess_constituents, wrap_phi,
)
from benchmarks.synthetic import generate_massless_jets, massless_jet_split
from benchmarks.tabular_runner import evaluate_test
from benchmarks.train import TrainConfig, train_classifier

MODELS = {"pfn": ParticleFlowNetwork, "efn": EnergyFlowNetwork}
N_PARAMS = {"pfn": 5106, "efn": 5082}
PAPER_N_PARAMS = {"pfn": 82458, "efn": 82358}


def _pack(X: torch.Tensor) -> torch.Tensor:
    """(n, K, 4) zero-padded momenta -> (n, K, 5) harness tensor with the mask."""
    mask = (X.abs().sum(dim=-1) > 0).to(X.dtype)
    return torch.cat([X, mask.unsqueeze(-1)], dim=-1)


@pytest.fixture(scope="module")
def jets() -> torch.Tensor:
    """48 toy jets in raw GeV (17-32 real constituents), float64."""
    X, _ = generate_massless_jets(48, 32, seed=11)
    return _pack(X)


@pytest.fixture(scope="module")
def toy_split():
    return massless_jet_split(3000, 32, seed=0)


def _generic(name: str, seed: int = 0) -> torch.nn.Module:
    """A random model with random biases, in eval mode."""
    torch.manual_seed(seed)
    model = MODELS[name]()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.bias.copy_(0.3 * torch.randn(m.bias.shape, generator=g,
                                               dtype=m.bias.dtype))
    return model.eval()


def _logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def _assert_same(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-10) -> None:
    """max |a - b| <= tol * max(1, max |b|): absolute for O(1) logits, relative above."""
    scale = max(1.0, b.abs().max().item())
    err = (a - b).abs().max().item()
    assert err <= tol * scale, f"max |diff| = {err:.3e} (scale {scale:.3g})"


def _boost_z(x: torch.Tensor, rapidity: float) -> torch.Tensor:
    """Longitudinal boost: every particle's rapidity y -> y + rapidity."""
    out = x.clone()
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    out[..., 0] = x[..., 0] * ch + x[..., 3] * sh
    out[..., 3] = x[..., 3] * ch + x[..., 0] * sh
    return out


def _rotate_z(x: torch.Tensor, alpha) -> torch.Tensor:
    """Rotation about the beam, phi -> phi + alpha (a number, or (B,) per jet)."""
    alpha = torch.as_tensor(alpha, dtype=x.dtype)
    if alpha.ndim == 1:
        alpha = alpha.unsqueeze(-1)
    c, s = torch.cos(alpha), torch.sin(alpha)
    out = x.clone()
    out[..., 1] = c * x[..., 1] - s * x[..., 2]
    out[..., 2] = s * x[..., 1] + c * x[..., 2]
    return out


def _across_branch_cut(x: torch.Tensor) -> torch.Tensor:
    """Every jet rotated so that its axis sits at phi = pi - 0.05."""
    phi_jet = torch.atan2(x[..., 2].sum(dim=1), x[..., 1].sum(dim=1))
    return _rotate_z(x, (math.pi - 0.05) - phi_jet)


def _append(x: torch.Tensor, row) -> torch.Tensor:
    """Append one (E, px, py, pz, mask) row to every jet."""
    r = torch.tensor(row, dtype=x.dtype).expand(x.shape[0], 1, 5)
    return torch.cat([x, r], dim=1)


def _split(x: torch.Tensor, slot: int, f: float) -> torch.Tensor:
    """Split constituent ``slot`` of every jet into f p and (1 - f) p."""
    out = x.clone()
    out[:, slot, :4] = f * x[:, slot, :4]
    extra = x[:, slot:slot + 1].clone()
    extra[..., :4] = (1.0 - f) * x[:, slot:slot + 1, :4]
    return torch.cat([out, extra], dim=1)


# ─────────────────────────────────────────────────────────────────────────
# (a), (b) permutations and padding
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", MODELS)
def test_permutation_invariance(jets, name):
    model = _generic(name)
    ref = _logits(model, jets)
    g = torch.Generator().manual_seed(0)
    n_real = jets[..., 4].sum(dim=1).long()
    K = jets.shape[1]
    # the real constituents of each jet shuffled among themselves
    idx = torch.stack([torch.cat([torch.randperm(int(n), generator=g),
                                  torch.arange(int(n), K)]) for n in n_real])
    shuffled = torch.gather(jets, 1, idx.unsqueeze(-1).expand_as(jets))
    assert not torch.equal(shuffled, jets)
    _assert_same(_logits(model, shuffled), ref)
    # all K slots shuffled: padding interleaved with the real constituents
    _assert_same(_logits(model, jets[:, torch.randperm(K, generator=g)]), ref)


@pytest.mark.parametrize("name", MODELS)
def test_padding_contributes_nothing(jets, name):
    model = _generic(name)
    ref = _logits(model, jets)
    padded = torch.cat([jets, torch.zeros(len(jets), 9, 5, dtype=jets.dtype)], dim=1)
    _assert_same(_logits(model, padded), ref)
    # masked rows are ignored whatever they contain
    garbage = jets.clone()
    pad = garbage[..., 4] == 0
    assert pad.any()
    vals = 50.0 * torch.randn(int(pad.sum()), 4, dtype=jets.dtype,
                              generator=torch.Generator().manual_seed(1))
    vals[0, 0], vals[-1, 2] = float("nan"), float("inf")
    garbage[..., :4][pad] = vals
    _assert_same(_logits(model, garbage), ref)
    # the padded slots of the preprocessing output are exactly zero
    z, coords, real = preprocess_constituents(garbage)
    assert torch.equal(real, jets[..., 4] > 0)
    assert (z[~real] == 0).all() and (coords[~real] == 0).all()


# ─────────────────────────────────────────────────────────────────────────
# (c) IRC safety
# ─────────────────────────────────────────────────────────────────────────

IRC_PERTURBATIONS = {
    "zero_energy_particle": lambda x: _append(x, [0.0, 0.0, 0.0, 0.0, 1.0]),
    "pt0_along_beam": lambda x: _append(x, [5.0, 0.0, 0.0, 5.0, 1.0]),
    "split_hardest": lambda x: _split(x, 0, 0.3),
    "split_soft": lambda x: _split(x, 9, 0.5),
}


@pytest.mark.parametrize("perturbation", IRC_PERTURBATIONS)
def test_efn_is_irc_safe_and_pfn_is_not(jets, perturbation):
    x_new = IRC_PERTURBATIONS[perturbation](jets)
    efn, pfn = _generic("efn"), _generic("pfn")
    ref = _logits(efn, jets)
    torch.testing.assert_close(_logits(efn, x_new), ref, rtol=0.0, atol=1e-10)
    # The same perturbation is visible to the PFN, so the check has teeth.
    assert (_logits(pfn, x_new) - _logits(pfn, jets)).abs().max() > 1e-3


def test_collinear_split_keeps_the_efn_inputs(jets):
    """Both halves of a split particle land on the original's coordinates."""
    z, coords, _ = preprocess_constituents(jets)
    z2, coords2, _ = preprocess_constituents(_split(jets, 0, 0.3))
    torch.testing.assert_close(z2[:, 0] + z2[:, -1], z[:, 0], rtol=0.0, atol=1e-14)
    for slot in (0, -1):
        torch.testing.assert_close(coords2[:, slot], coords[:, 0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(coords2[:, 1:-1], coords[:, 1:], rtol=0.0, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# (d) scale, longitudinal boost, rotation about the beam
# ─────────────────────────────────────────────────────────────────────────

SYMMETRIES = {
    "scale_1e-3": lambda x: torch.cat([1e-3 * x[..., :4], x[..., 4:]], dim=-1),
    "scale_37": lambda x: torch.cat([37.0 * x[..., :4], x[..., 4:]], dim=-1),
    "boost_+1.7": lambda x: _boost_z(x, 1.7),
    "boost_-0.9": lambda x: _boost_z(x, -0.9),
    "rotate_2.5": lambda x: _rotate_z(x, 2.5),
    "rotate_across_pi": _across_branch_cut,
}


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_invariance_under_scale_boost_rotation(jets, name, symmetry):
    model = _generic(name)
    _assert_same(_logits(model, SYMMETRIES[symmetry](jets)), _logits(model, jets))


def test_branch_cut_rotation_really_straddles_pi(jets):
    x = _across_branch_cut(jets)
    phi = torch.atan2(x[..., 2], x[..., 1])
    real = x[..., 4] > 0
    straddles = ((phi > 2.5) & real).any(dim=1) & ((phi < -2.5) & real).any(dim=1)
    assert straddles.all()


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────

def test_wrap_phi_maps_into_minus_pi_pi():
    phi = torch.tensor([math.pi, -math.pi, 1.5 * math.pi, -1.5 * math.pi, 7.0, -0.3, 0.0],
                       dtype=torch.float64)
    want = torch.tensor([math.pi, math.pi, -0.5 * math.pi, 0.5 * math.pi,
                         7.0 - 2.0 * math.pi, -0.3, 0.0], dtype=torch.float64)
    torch.testing.assert_close(wrap_phi(phi), want, rtol=0.0, atol=1e-15)
    many = torch.linspace(-20.0, 20.0, 4001, dtype=torch.float64)
    w = wrap_phi(many)
    assert (w > -math.pi).all() and (w <= math.pi).all()
    torch.testing.assert_close(torch.cos(w), torch.cos(many), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(torch.sin(w), torch.sin(many), rtol=0.0, atol=1e-12)


def _massless(pt: float, eta: float, phi: float) -> list[float]:
    return [pt * math.cosh(eta), pt * math.cos(phi), pt * math.sin(phi),
            pt * math.sinh(eta), 1.0]


def test_constituent_kinematics_by_hand():
    m, pt3, y3 = 10.0, 50.0, 0.2
    mt3 = math.hypot(pt3, m)
    rounded = _massless(30.0, 1.1, 0.0)
    rounded[0] *= 1.0 - 1e-6                     # E a hair below |p|: m^2 < 0
    rows = [
        _massless(100.0, 0.7, 0.3),
        _massless(100.0, -0.4, -0.3),            # jet axis stays at phi = 0
        [mt3 * math.cosh(y3), pt3, 0.0, mt3 * math.sinh(y3), 1.0],   # massive
        rounded,
        [39.99999, 1e-3, 0.0, 40.0, 1.0],        # E < |pz|: arctanh(pz/E) is NaN
        [0.0, 0.0, 0.0, 0.0, 1.0],               # zero momentum, real
        [7.0, float("nan"), 2.0, 1.0, 0.0],      # padding with garbage
    ]
    x = torch.tensor([rows], dtype=torch.float64)
    pt, y, phi, real = constituent_kinematics(x)
    want_pt = [100.0, 100.0, pt3, 30.0, 1e-3, 0.0, 0.0]
    want_y = [0.7, -0.4, y3, 1.1, math.asinh(40.0 / 1e-3), 0.0, 0.0]
    want_phi = [0.3, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    for got, want in ((pt, want_pt), (y, want_y), (phi, want_phi)):
        torch.testing.assert_close(got[0], torch.tensor(want, dtype=torch.float64),
                                   rtol=0.0, atol=1e-12)
    assert real[0].tolist() == [True] * 6 + [False]
    z, coords, _ = preprocess_constituents(x)
    assert torch.isfinite(z).all() and torch.isfinite(coords).all()
    for name in MODELS:
        assert torch.isfinite(_logits(_generic(name), x)).all()


def test_preprocessing_centres_aligns_and_reflects(jets):
    z, coords, real = preprocess_constituents(jets)
    v, u = coords[..., 0], coords[..., 1]
    torch.testing.assert_close(z.sum(dim=1), torch.ones(len(z), dtype=z.dtype))
    assert (z[real] > 0).all()
    tol = 1e-12
    assert (z * u).sum(dim=1).abs().max() < tol           # centred
    assert (z * v).sum(dim=1).abs().max() < tol
    assert (z * u * v).sum(dim=1).abs().max() < tol       # principal axes
    assert ((z * u * u).sum(dim=1) >= (z * v * v).sum(dim=1)).all()
    for c in (u, v):                                      # reflected
        pos = torch.where(c > 0, z, 0.0).sum(dim=1)
        neg = torch.where(c < 0, z, 0.0).sum(dim=1)
        assert (pos >= neg).all()


@pytest.mark.parametrize("across_pi", [False, True])
def test_preprocessing_matches_energyflow(jets, across_pi):
    ef = pytest.importorskip("energyflow")
    x = _across_branch_cut(jets) if across_pi else jets
    z, coords, real = preprocess_constituents(x)
    for i in range(len(x)):
        r = real[i]
        p4 = x[i, r, :4].numpy()
        event = ef.ptyphims_from_p4s(p4, phi_ref="hardest", mass=False)
        event = ef.center_ptyphims(event, center="ptscheme")
        event = ef.rotate_ptyphims(event, rotate="ptscheme")
        event = ef.reflect_ptyphims(event, which="both")
        np.testing.assert_allclose(z[i, r].numpy(), event[:, 0] / event[:, 0].sum(),
                                   rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(coords[i, r].numpy(), event[:, 1:3],
                                   rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("name", MODELS)
def test_float32_inputs_stay_finite(jets, name):
    x32 = jets.float()
    E = x32[..., 0].double()
    p = x32[..., 1:4].double().norm(dim=-1)
    real = x32[..., 4] > 0
    assert ((E < p) & real).any()             # rounding puts E below |p|
    model = _generic(name)
    out = _logits(model, x32)
    assert out.dtype == torch.float64 and torch.isfinite(out).all()
    ref = _logits(model, jets)
    assert (out - ref).abs().max() < 1e-3 * max(1.0, ref.abs().max().item())


# ─────────────────────────────────────────────────────────────────────────
# (e) parameter counts and registry
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", MODELS)
def test_parameter_counts_through_the_registry(name):
    model = build_model(name, in_features=4, out_features=2,
                        representation="constituents")
    assert isinstance(model, MODELS[name])
    n = count_parameters(model)
    assert 4500 <= n <= 5500
    assert n == N_PARAMS[name]
    widths = [(m.in_features, m.out_features) for m in model.modules()
              if isinstance(m, torch.nn.Linear)]
    first = 3 if name == "pfn" else 2
    assert widths == [(first, 24), (24, 24), (24, 64),
                      (64, 24), (24, 24), (24, 24), (24, 2)]
    # ReLU after every Phi layer (the latent one included) and every F layer
    for stack in (model.Phi, model.F):
        assert [type(m).__name__ for m in stack] == ["Linear", "ReLU"] * 3
    assert model_family(name) == "baseline_set" and is_set_model(name)
    assert all(p.dtype == torch.float64 for p in model.parameters())
    reg = model.regularization_loss()
    assert reg.shape == () and reg.dtype == torch.float64 and reg.item() == 0.0


@pytest.mark.parametrize("name", MODELS)
def test_initialisation_follows_energyflow(name):
    """Zero biases; He-uniform Phi / F weights, Glorot-uniform output weights."""
    torch.manual_seed(0)
    model = MODELS[name]()
    hidden = [m for m in (*model.Phi, *model.F) if isinstance(m, torch.nn.Linear)]
    scaled = []
    for lin in hidden + [model.output]:
        assert (lin.bias == 0).all()
        fan_out, fan_in = lin.weight.shape
        glorot = lin is model.output
        bound = math.sqrt(6.0 / (fan_in + fan_out) if glorot else 6.0 / fan_in)
        w = lin.weight.detach() / bound
        assert w.abs().max() <= 1.0
        if glorot:      # std of U(-1, 1) is 1/sqrt(3); 48 weights here
            assert abs(w.std().item() * math.sqrt(3.0) - 1.0) < 0.25
        else:
            scaled.append(w.flatten())
    # pooled over ~4.9k weights; PyTorch's default init would give ~0.41
    assert abs(torch.cat(scaled).std().item() * math.sqrt(3.0) - 1.0) < 0.05


@pytest.mark.parametrize("name", MODELS)
def test_paper_sizes_parameter_count(name):
    model = MODELS[name](Phi_sizes=PAPER_PHI_SIZES, F_sizes=PAPER_F_SIZES)
    assert count_parameters(model) == PAPER_N_PARAMS[name]


@pytest.mark.parametrize("name", MODELS)
def test_chunked_evaluation_matches_one_batch(jets, name):
    model = _generic(name)
    model.eval_chunk_size = None
    ref = _logits(model, jets)
    model.eval_chunk_size = 10                # 48 jets -> 5 chunks
    _assert_same(_logits(model, jets), ref, tol=1e-13)


# ─────────────────────────────────────────────────────────────────────────
# (f) short training on toy jets
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", MODELS)
def test_short_training_separates_toy_jets(toy_split, name):
    pytest.importorskip("sklearn")            # evaluate_test's AUC needs it
    s = toy_split
    torch.manual_seed(0)
    model = build_model(name, in_features=4, out_features=2,
                        representation="constituents")
    res = train_classifier(model, s.X_train, s.y_train, s.X_val, s.y_val,
                           TrainConfig(epochs=4, batch_size=128, lr=3e-3, seed=0))
    assert not res.diverged and res.nonfinite_steps == 0
    assert res.n_params == N_PARAMS[name]
    out = evaluate_test(model, s.X_test, s.y_test, n_classes=2)
    assert out["test_auc"] >= 0.85, out
