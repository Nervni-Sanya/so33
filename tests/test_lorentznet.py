"""
LorentzNet baseline at ~5k parameters (benchmarks/lorentznet.py).

Verifies, on toy massless jets (benchmarks.synthetic, globally RMS-normalised
as the harness delivers them; norms up to ~50), with randomly initialised
models whose biases, BatchNorm parameters and running statistics are
randomised and whose coordinate-update weights are made large (at
initialisation the coordinate update is typically ~1e-6 of the coordinates, so
errors in that path would go unseen):
  • Equivalence with the official algorithm: an independent edge-list
    implementation of the official models.py forward (concatenated edge
    input, index_add_ segment sums / means, nn.BatchNorm1d over the edge and
    node lists, the official clamp and normalisations) run on the ragged batch
    of the real nodes gives the same logits, the same parameter gradients and
    the same BatchNorm running statistics as the dense padded model, in
    training and evaluation mode, also with momenta large enough for the +-100
    clamp to act (1e-11 / 1e-10). The masked BatchNorm (hand-written backward)
    matches nn.BatchNorm1d on the selected rows. Decoder dropout acts in
    training mode only.
  • Symmetries (float64, 1e-8 relative to the logit scale, evaluation mode;
    training mode for the main ones). The beams are built in the jet's
    longitudinal rest frame (the default ``beam_frame="jet"``), so the logits
    are invariant under a common boost along z, a common rotation about z
    (one angle for all jets or one per jet), their product, and the
    reflections z -> -z, px -> -px, py -> -py. They are NOT invariant under a
    boost along x or y: the beams break that symmetry. The network after the
    graph construction is exactly Lorentz invariant: a general Lorentz
    transformation (rotation, transverse boost, rotation) applied to the
    constituents AND the beams leaves the logits unchanged, while applying it
    to the constituents alone changes them, which pins the symmetry breaking
    on the beams. With the official lab-frame beams (``beam_frame="lab"``)
    the rotations about z and the reflections stay exact, but a boost along z
    changes the logits (no vector in the (t, z) plane is invariant). The
    componentwise clamp of the coordinate update is not covariant, so exact
    invariance holds while it does not act: a check pins that the models used
    here stay clear of it.
  • The graph: the beams are (sqrt 2, 0, 0, +-1) in the lab frame and that
    vector boosted along z by the rapidity of the jet in the jet frame (both
    have mass 1 and <b+, b-> = 3); the scalars are psi(m^2, label); padded
    rows are zeroed whatever they contain; a jet without a real constituent
    or with M^2 = E^2 - pz^2 <= 0 gets the lab beams.
  • Permutation invariance of the real constituents (and of all K slots with
    padding interleaved).
  • Padding: appending zero rows, garbage (NaN, inf, large values) in masked
    rows, and extra padding slots change neither the logits, nor the
    parameter gradients, nor the BatchNorm running statistics, in training
    and evaluation mode.
  • psi, the Minkowski products, the official parameter names and shapes, and
    the argument checks.
  • Parameter counts through benchmarks.models.build_model: 5,088 trainable
    parameters (n_hidden 10, n_layers 6; in [4500, 5500]); the official
    top-tagging configuration (72, 6) gives 224,072.
  • Finite loss and gradients for a batch with real jets, a one-constituent
    jet, an all-padding jet and a jet along the beam; for a batch of only
    degenerate jets; for a one-jet batch; for the hardest constituents of the
    toy data (Euclidean norm ~50) and for constituents 20 times harder, in
    float64 and float32. The coordinate update is bounded by c * 100 per
    layer (the clamp).
  • float32 models run (outputs float32, close to the float64 ones); float32
    inputs stay finite; chunked no-grad evaluation equals one batch and does
    not apply to training mode; ``momentum_scale`` rescales the inputs.
  • A short training run (massless_jet_split(2000, 32), 3 epochs,
    benchmarks.train.train_classifier) reaches test AUC >= 0.85 (the toy
    jet-mass AUC is ~0.94). BatchNorm keeps its default momentum 0.1 in the
    model; the test raises it to 0.5, because 33 steps leave 3% of the
    initial running variance (1) in the running statistics, which swamps the
    first-layer channels whose batch variance is ~1e-4, so that with the
    default the evaluation-mode logits of such a short run are a lottery
    although the network has learned the task (see the module docstring).

Run:
    python -m pytest tests/test_lorentznet.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import copy
import math

import pytest
import torch
import torch.nn as nn

from benchmarks.lorentznet import (
    LGEB, LorentzNet, beam_spurions, lorentznet_graph, minkowski_dot,
    minkowski_normsq, pairwise_minkowski_dot, psi, _masked_batch_norm,
)
from benchmarks.models import build_model, count_parameters, is_set_model, model_family
from benchmarks.synthetic import massless_jet_split
from benchmarks.tabular_runner import evaluate_test
from benchmarks.train import TrainConfig, train_classifier, predict_logits

N_PARAMS = 5088
OFFICIAL_N_PARAMS = 224072
SQRT2 = math.sqrt(2.0)
LOG2 = math.log(2.0)
ETA = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))


@pytest.fixture(scope="module")
def toy_split():
    return massless_jet_split(2000, 32, seed=0)


@pytest.fixture(scope="module")
def jets(toy_split) -> torch.Tensor:
    """48 toy jets, RMS-normalised: 32 with padding (15-31 real constituents), 16 full."""
    X = toy_split.X_train
    n_real = X[..., 4].sum(dim=1)
    ragged = (n_real < 32).nonzero().flatten()[:32]
    full = (n_real == 32).nonzero().flatten()[:16]
    out = X[torch.cat([ragged, full])].clone()
    n = out[..., 4].sum(dim=1)
    assert len(out) == 48 and n.min() <= 20 and n.max() == 32
    return out


def _generic(seed: int = 0, phi_x_scale: float = 0.3, **kw) -> LorentzNet:
    """A random model in eval mode: random biases, BatchNorm affine parameters and
    running statistics, and large phi_x output weights (N(0, phi_x_scale^2)) with
    c = 0.2, so that every part of the network, the coordinate updates included,
    does something. The default keeps |x_diff phi_x| ~ 10 on the toy jets, far
    from the coordinate clamp at 100 (which breaks exact Lorentz invariance when
    it acts)."""
    torch.manual_seed(seed)
    kw.setdefault("c_weight", 0.2)
    model = LorentzNet(**kw)
    g = torch.Generator().manual_seed(seed + 1)

    def rnd(like: torch.Tensor, scale: float) -> torch.Tensor:
        return scale * torch.randn(like.shape, generator=g, dtype=like.dtype)

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.copy_(rnd(m.bias, 0.3))
            if isinstance(m, nn.BatchNorm1d):
                m.weight.copy_(1.0 + rnd(m.weight, 0.3))
                m.bias.copy_(rnd(m.bias, 0.3))
                m.running_mean.copy_(rnd(m.running_mean, 0.5))
                m.running_var.copy_(0.5 + torch.rand(m.running_var.shape, generator=g,
                                                     dtype=m.running_var.dtype))
        for lgeb in model.LGEBs:
            if not lgeb.last_layer:
                lgeb.phi_x[2].weight.copy_(rnd(lgeb.phi_x[2].weight, phi_x_scale))
    return model.eval()


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def _assert_same(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-8) -> None:
    """max |a - b| <= tol * max(1, max |b|): absolute for O(1) logits, relative above."""
    scale = max(1.0, b.abs().max().item())
    err = (a - b).abs().max().item()
    assert err <= tol * scale, f"max |diff| = {err:.3e} (scale {scale:.3g})"


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| relative to max(1, max |b|)."""
    return (a - b).abs().max().item() / max(1.0, b.abs().max().item())


def _boost(x: torch.Tensor, rapidity: float, axis: int) -> torch.Tensor:
    """Boost of every particle along axis 1 (x), 2 (y) or 3 (z) of (E, px, py, pz, mask)."""
    out = x.clone()
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    out[..., 0] = x[..., 0] * ch + x[..., axis] * sh
    out[..., axis] = x[..., axis] * ch + x[..., 0] * sh
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


def _reflect(x: torch.Tensor, axis: int) -> torch.Tensor:
    out = x.clone()
    out[..., axis] = -x[..., axis]
    return out


def _lorentz_matrix() -> torch.Tensor:
    """A general proper Lorentz transformation R_z(0.7) B_x(1.1) R_y(-0.4) on (E, px, py, pz)."""
    def rot_z(t):
        c, s = math.cos(t), math.sin(t)
        return torch.tensor([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
                            dtype=torch.float64)

    def rot_y(t):
        c, s = math.cos(t), math.sin(t)
        return torch.tensor([[1, 0, 0, 0], [0, c, 0, s], [0, 0, 1, 0], [0, -s, 0, c]],
                            dtype=torch.float64)

    def boost_x(e):
        ch, sh = math.cosh(e), math.sinh(e)
        return torch.tensor([[ch, sh, 0, 0], [sh, ch, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            dtype=torch.float64)

    return rot_z(0.7) @ boost_x(1.1) @ rot_y(-0.4)


def _real_shuffle(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """The real constituents of every jet shuffled among themselves (padding last)."""
    g = torch.Generator().manual_seed(seed)
    n_real = x[..., 4].sum(dim=1).long()
    K = x.shape[1]
    idx = torch.stack([torch.cat([torch.randperm(int(n), generator=g),
                                  torch.arange(int(n), K)]) for n in n_real])
    return torch.gather(x, 1, idx.unsqueeze(-1).expand_as(x))


def _pad(x: torch.Tensor, slots: int) -> torch.Tensor:
    return torch.cat([x, torch.zeros(len(x), slots, 5, dtype=x.dtype)], dim=1)


def _loss_and_grads(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    return loss, [p.grad.clone() for p in model.parameters()]


def _all_finite(loss: torch.Tensor, model: nn.Module) -> bool:
    return bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all())
                                              for p in model.parameters())


def _labels(n: int) -> torch.Tensor:
    return torch.arange(n) % 2


# ─────────────────────────────────────────────────────────────────────────
# Equivalence with the official algorithm (independent edge-list version)
# ─────────────────────────────────────────────────────────────────────────

def _normsq4(p: torch.Tensor) -> torch.Tensor:
    """Official ``normsq4``: 2 p_0^2 - sum_mu p_mu^2."""
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def _dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Official ``dotsq4``: 2 p_0 q_0 - sum_mu p_mu q_mu."""
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def _psi(a: torch.Tensor) -> torch.Tensor:
    """Official ``psi``: sgn(a) log(|a| + 1)."""
    return torch.sign(a) * torch.log(torch.abs(a) + 1)


def _segment_sum(data: torch.Tensor, ids: torch.Tensor, n: int) -> torch.Tensor:
    result = data.new_zeros((n, data.size(1)))
    result.index_add_(0, ids, data)
    return result


def _segment_mean(data: torch.Tensor, ids: torch.Tensor, n: int) -> torch.Tensor:
    count = data.new_zeros((n, data.size(1)))
    count.index_add_(0, ids, torch.ones_like(data))
    return _segment_sum(data, ids, n) / count.clamp(min=1)


def _official_logits(model: LorentzNet, coords, scalars, node_mask,
                     stats: dict | None = None) -> torch.Tensor:
    """The official LorentzNet forward (models.py) on the ragged batch of real nodes.

    Every graph keeps only its real nodes, concatenated into one list of M
    nodes, with the edge list of all ordered pairs i != j inside a graph;
    LGEB.forward is then models.py line by line (edge input
    cat[h_i, h_j, psi(norm), psi(dot)], attention weight, clamped coordinate
    mean, segment-summed messages), with the model's own modules, so that
    nn.BatchNorm1d sees the (E, H) edge list and the (M, H) node list. The
    pooling is the per-graph mean over the real nodes (the documented
    departure), not the official ``mean(h * node_mask)`` over all slots.
    ``stats["clamped"]`` counts the entries the clamp acts on.
    """
    B, N = node_mask.shape
    counts = node_mask.sum(dim=1).tolist()
    real = node_mask.reshape(-1)
    x = coords.reshape(B * N, 4)[real]
    s = scalars.reshape(B * N, -1)[real]
    rows, cols, offset = [], [], 0
    for n in counts:
        i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        keep = i != j
        rows.append(i[keep] + offset)
        cols.append(j[keep] + offset)
        offset += n
    i, j = torch.cat(rows), torch.cat(cols)
    graph_id = torch.repeat_interleave(torch.arange(B), torch.tensor(counts))
    M = len(x)

    h = model.embedding(s)
    for lgeb in model.LGEBs:
        x_diff = x[i] - x[j]
        norms = _psi(_normsq4(x_diff)).unsqueeze(1)
        dots = _psi(_dotsq4(x[i], x[j])).unsqueeze(1)
        m = lgeb.phi_e(torch.cat([h[i], h[j], norms, dots], dim=1))
        m = m * lgeb.phi_m(m)
        if not lgeb.last_layer:
            trans = x_diff * lgeb.phi_x(m)
            if stats is not None:
                stats["clamped"] = stats.get("clamped", 0) + int((trans.abs() > 100).sum())
            trans = torch.clamp(trans, min=-100, max=100)
            x = x + _segment_mean(trans, i, M) * lgeb.c_weight
        agg = _segment_sum(m, i, M)
        h = h + lgeb.phi_h(torch.cat([h, agg, s], dim=1))
    return model.graph_dec(_segment_mean(h, graph_id, B))


@pytest.mark.parametrize("mode,momentum_factor", [("train", 1.0), ("eval", 1.0), ("train", 8.0)])
def test_dense_model_equals_the_official_edge_list_algorithm(jets, mode, momentum_factor):
    """Logits, parameter gradients and BatchNorm running statistics agree with the
    official algorithm run on the ragged batch of real nodes (no dropout: the
    decoder dropout acts on the pooled vector in both and is checked below).
    c = 1 and large phi_x weights make the coordinate updates substantial; with
    the momenta 8 times larger the +-100 clamp acts on some edges (checked)."""
    model = _generic(0, phi_x_scale=0.6, c_weight=1.0, dropout=0.0)
    model.train(mode == "train")
    ref_model = copy.deepcopy(model)
    y = _labels(len(jets))
    x = jets.clone()
    x[..., :4] *= momentum_factor

    logits = model(x)
    coords, scalars, node_mask = ref_model.graph(x)
    assert node_mask.sum(dim=1).min() < node_mask.shape[1]          # padding present
    stats: dict = {}
    ref = _official_logits(ref_model, coords, scalars, node_mask, stats)
    assert (stats["clamped"] > 0) == (momentum_factor > 1.0)
    torch.testing.assert_close(logits, ref, rtol=0.0, atol=1e-11)

    nn.functional.cross_entropy(logits, y).backward()
    nn.functional.cross_entropy(ref, y).backward()
    scale = max(p.grad.abs().max().item() for p in ref_model.parameters())
    assert scale > 1e-3
    for (name, p), (_, q) in zip(model.named_parameters(), ref_model.named_parameters()):
        err = (p.grad - q.grad).abs().max().item()
        assert err <= 1e-10 * scale, f"{name}: |grad diff| {err:.3e} (scale {scale:.3g})"
    for (name, a), (_, b) in zip(model.named_buffers(), ref_model.named_buffers()):
        torch.testing.assert_close(a.double(), b.double(), rtol=1e-11, atol=1e-11,
                                   msg=lambda m, name=name: f"{name}: {m}")


def test_coordinate_updates_matter_in_the_equivalence_check(jets):
    """The model of the check above is sensitive to c, so the check covers the x path."""
    a = _generic(0, phi_x_scale=0.6, c_weight=1.0)
    b = _generic(0, phi_x_scale=0.6, c_weight=0.0)
    assert _max_diff(_logits(a, jets), _logits(b, jets)) > 1e-3        # ~6e-3


def test_decoder_dropout_is_active_in_training_only(jets):
    model = _generic(0, dropout=0.5)
    with torch.no_grad():
        model.train()
        torch.manual_seed(0)
        a = model(jets)
        torch.manual_seed(1)
        b = model(jets)
        assert (a - b).abs().max() > 1e-6
        model.eval()
        torch.testing.assert_close(model(jets), model(jets), rtol=0.0, atol=0.0)
    # dropout only in the decoder, as in the official code: the node features
    # of the LGEB stack do not depend on it
    model.train()
    with torch.no_grad():
        torch.manual_seed(0)
        h1, _ = model.node_features(jets)
        torch.manual_seed(1)
        h2, _ = model.node_features(jets)
    assert torch.equal(h1, h2)


@pytest.mark.parametrize("train", [True, False])
def test_masked_batch_norm_equals_batch_norm_on_the_selected_rows(train):
    g = torch.Generator().manual_seed(3)
    C = 7
    bn = nn.BatchNorm1d(C, dtype=torch.float64)
    with torch.no_grad():
        bn.weight.copy_(1.0 + 0.4 * torch.randn(C, generator=g, dtype=torch.float64))
        bn.bias.copy_(0.4 * torch.randn(C, generator=g, dtype=torch.float64))
        bn.running_mean.copy_(torch.randn(C, generator=g, dtype=torch.float64))
        bn.running_var.copy_(0.5 + torch.rand(C, generator=g, dtype=torch.float64))
    bn.train(train)
    ref_bn = copy.deepcopy(bn)

    x = (2.0 * torch.randn(5, 6, 6, C, generator=g, dtype=torch.float64) + 1.0
         ).requires_grad_(True)
    w = (torch.rand(5, 6, 6, generator=g) > 0.45).double()
    dy = torch.randn(5, 6, 6, C, generator=g, dtype=torch.float64) * w.unsqueeze(-1)
    rows = w.bool()
    assert 20 < int(rows.sum()) < 160

    out = _masked_batch_norm(bn, x, w)
    out.backward(dy)
    x2 = x.detach().clone().requires_grad_(True)
    ref = ref_bn(x2[rows])
    ref.backward(dy[rows])

    torch.testing.assert_close(out[rows], ref, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(x.grad[rows], x2.grad[rows], rtol=0.0, atol=1e-12)
    assert (x.grad[~rows] == 0).all()
    torch.testing.assert_close(bn.weight.grad, ref_bn.weight.grad, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(bn.bias.grad, ref_bn.bias.grad, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(bn.running_mean, ref_bn.running_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(bn.running_var, ref_bn.running_var, rtol=1e-12, atol=1e-12)
    assert int(bn.num_batches_tracked) == int(ref_bn.num_batches_tracked)


# ─────────────────────────────────────────────────────────────────────────
# Symmetries
# ─────────────────────────────────────────────────────────────────────────

BEAM_SYMMETRIES = {
    "boost_z_+1.7": lambda x: _boost(x, 1.7, 3),
    "boost_z_-0.9": lambda x: _boost(x, -0.9, 3),
    "rotate_z_2.5": lambda x: _rotate_z(x, 2.5),
    "rotate_z_per_jet": lambda x: _rotate_z(x, torch.linspace(-3.0, 3.0, len(x))),
    "boost_z_then_rotate_z": lambda x: _rotate_z(_boost(x, 1.2, 3), -0.7),
    "reflect_z": lambda x: _reflect(x, 3),
    "reflect_px": lambda x: _reflect(x, 1),
    "reflect_py": lambda x: _reflect(x, 2),
}


@pytest.mark.parametrize("symmetry", BEAM_SYMMETRIES)
def test_logits_are_invariant_under_the_symmetries_the_beams_leave(jets, symmetry):
    model = _generic(0)
    ref = _logits(model, jets)
    _assert_same(_logits(model, BEAM_SYMMETRIES[symmetry](jets)), ref, tol=1e-8)


@pytest.mark.parametrize("symmetry", ["boost_z_+1.7", "rotate_z_2.5", "boost_z_then_rotate_z"])
def test_invariance_holds_in_training_mode_too(jets, symmetry):
    """BatchNorm batch statistics of invariant edge / node features are invariant."""
    model = _generic(3, dropout=0.0).train()
    ref = _logits(model, jets)
    _assert_same(_logits(model, BEAM_SYMMETRIES[symmetry](jets)), ref, tol=1e-8)


def test_default_model_is_invariant_at_initialisation(jets):
    torch.manual_seed(0)
    model = LorentzNet().eval()
    ref = _logits(model, jets)
    for symmetry in ("boost_z_+1.7", "rotate_z_2.5"):
        _assert_same(_logits(model, BEAM_SYMMETRIES[symmetry](jets)), ref, tol=1e-8)


@pytest.mark.parametrize("axis", [1, 2])
def test_transverse_boost_changes_the_logits(jets, axis):
    """The beams break the symmetry: a boost along x or y is not a symmetry."""
    model = _generic(0)
    ref = _logits(model, jets)
    boosted = _logits(model, _boost(jets, 1.0, axis))
    # 1e-4 is four orders above the invariance tolerance and ~35 times below the effect
    assert _max_diff(boosted, ref) > 1e-4


def test_the_network_is_lorentz_invariant_and_the_beams_break_it(jets):
    """A general Lorentz transformation of the whole graph (beams included) is a
    symmetry of everything after the graph construction; applied to the
    constituents alone it is not."""
    model = _generic(0)
    L = _lorentz_matrix()
    assert torch.allclose(L.T @ ETA @ L, ETA, atol=1e-13)
    with torch.no_grad():
        coords, scalars, mask = model.graph(jets)
        ref = model.graph_logits(coords, scalars, mask)
        assert torch.equal(ref, model(jets))
        # the graph transformed as a whole: constituents and beams
        moved = model.graph_logits(coords @ L.T, scalars, mask)
        _assert_same(moved, ref, tol=1e-8)
        # the transformed constituents with freshly built beams (what the model sees)
        x = jets.clone()
        x[..., :4] = jets[..., :4] @ L.T
        coords2, scalars2, mask2 = model.graph(x)
        assert torch.equal(mask2, mask)
        # the scalars are invariants: unchanged up to rounding
        torch.testing.assert_close(scalars2, scalars, rtol=0.0, atol=1e-10)
        assert _max_diff(model(x), ref) > 1e-4


def test_exact_invariance_checks_stay_clear_of_the_clamp(jets, monkeypatch):
    """The coordinate update clamps x_diff * phi_x componentwise at +-100, which is not
    covariant: exact invariance holds as long as it does not act (in training mode
    BatchNorm rescales the messages, and a poor draw of the random weights, e.g. seed
    0, does reach the clamp under a boost of 1.2). The models of the checks above
    stay at |x_diff phi_x| ~ 10-40, so they test the invariance of the network and
    not the clamp."""
    peaks = []
    orig = LGEB.x_model

    def spy(self, x, x_diff, m, edge_w):
        peaks.append((x_diff * self.phi_x(m) * edge_w.unsqueeze(-1)).abs().max().item())
        return orig(self, x, x_diff, m, edge_w)

    monkeypatch.setattr(LGEB, "x_model", spy)
    L = _lorentz_matrix()
    for seed, train in ((0, False), (3, True)):
        model = _generic(seed, dropout=0.0).train(train)
        peaks.clear()
        with torch.no_grad():
            for name in ("boost_z_+1.7", "boost_z_-0.9", "boost_z_then_rotate_z", "rotate_z_2.5"):
                model(BEAM_SYMMETRIES[name](jets))
            coords, scalars, mask = model.graph(jets)
            model.graph_logits(coords @ L.T, scalars, mask)
        assert len(peaks) == 5 * 5
        assert 1.0 < max(peaks) < 35.0, (seed, train, max(peaks))


LAB_SYMMETRIES = {k: BEAM_SYMMETRIES[k] for k in
                  ("rotate_z_2.5", "rotate_z_per_jet", "reflect_z", "reflect_px", "reflect_py")}


@pytest.mark.parametrize("symmetry", LAB_SYMMETRIES)
def test_lab_frame_beams_keep_rotations_and_reflections(jets, symmetry):
    model = _generic(0, beam_frame="lab")
    ref = _logits(model, jets)
    _assert_same(_logits(model, LAB_SYMMETRIES[symmetry](jets)), ref, tol=1e-8)


def test_lab_frame_beams_break_the_longitudinal_boost(jets):
    model = _generic(0, beam_frame="lab")
    ref = _logits(model, jets)
    assert _max_diff(_logits(model, _boost(jets, 1.0, 3)), ref) > 1e-4
    # the jet-frame beams do not
    jet = _generic(0)
    _assert_same(_logits(jet, _boost(jets, 1.0, 3)), _logits(jet, jets), tol=1e-8)


# ─────────────────────────────────────────────────────────────────────────
# Graph construction and helpers
# ─────────────────────────────────────────────────────────────────────────

def _hand_jet() -> torch.Tensor:
    """(1, 5, 5): a massless, a massive and a massless constituent, two garbage rows."""
    rows = [
        [5.0, 3.0, 0.0, 4.0, 1.0],            # massless
        [4.0, 1.0, 1.0, 1.0, 1.0],            # m^2 = 16 - 3 = 13
        [2.0, 0.0, 2.0, 0.0, 1.0],            # massless
        [9.0, float("nan"), 2.0, 1.0, 0.0],   # padding with garbage
        [7.0, 1.0, float("inf"), 3.0, 0.0],   # padding with garbage
    ]
    return torch.tensor([rows], dtype=torch.float64)


def test_lab_frame_graph_by_hand():
    x = _hand_jet()
    coords, scalars, mask = lorentznet_graph(x, beam_frame="lab")
    assert coords.shape == (1, 7, 4) and scalars.shape == (1, 7, 2) and mask.shape == (1, 7)
    want = torch.tensor([[SQRT2, 0, 0, 1], [SQRT2, 0, 0, -1],
                         [5, 3, 0, 4], [4, 1, 1, 1], [2, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        dtype=torch.float64)
    torch.testing.assert_close(coords[0], want, rtol=0.0, atol=1e-15)
    assert mask[0].tolist() == [True] * 5 + [False] * 2
    # (psi(m^2), psi(label)): beams m^2 = 1 and label +1; constituents label -1
    want_s = torch.tensor([[LOG2, LOG2], [LOG2, LOG2], [0.0, -LOG2],
                           [math.log(14.0), -LOG2], [0.0, -LOG2], [0.0, -LOG2], [0.0, -LOG2]],
                          dtype=torch.float64)
    torch.testing.assert_close(scalars[0], want_s, rtol=0.0, atol=1e-13)


def test_jet_frame_beams_are_the_lab_beams_boosted_by_the_jet_rapidity():
    x = _hand_jet()
    E, pz = 11.0, 5.0                  # the real constituents sum to J = (11, 4, 3, 5)
    mt = math.sqrt(E * E - pz * pz)
    ch, sh = E / mt, pz / mt           # cosh and sinh of the jet's rapidity
    for frame, (c, s) in (("lab", (1.0, 0.0)), ("jet", (ch, sh))):
        b = lorentznet_graph(x, beam_frame=frame)[0][0, :2]
        # (sqrt 2, 0, 0, +-1) boosted along z with cosh y = c, sinh y = s
        want = torch.tensor([[SQRT2 * c + s, 0, 0, SQRT2 * s + c],
                             [SQRT2 * c - s, 0, 0, SQRT2 * s - c]], dtype=torch.float64)
        torch.testing.assert_close(b, want, rtol=0.0, atol=1e-13)
        # Lorentz invariants of the pair: mass 1 each and <b+, b-> = 3
        assert abs(minkowski_normsq(b[0]).item() - 1.0) < 1e-12
        assert abs(minkowski_normsq(b[1]).item() - 1.0) < 1e-12
        assert abs(minkowski_dot(b[0], b[1]).item() - 3.0) < 1e-12
    # unboosting the jet-frame beams by the jet's rapidity gives the lab beams back
    b = lorentznet_graph(x, beam_frame="jet")[0][0, :2]
    for sign, row in ((1.0, b[0]), (-1.0, b[1])):
        e_lab = ch * row[0] - sh * row[3]
        p_lab = ch * row[3] - sh * row[0]
        assert abs(e_lab.item() - SQRT2) < 1e-12 and abs(p_lab.item() - sign) < 1e-12
    # beam_spurions takes the momenta with the padding already zeroed
    p = torch.where((x[..., 4] > 0).unsqueeze(-1), x[..., :4], 0.0)
    torch.testing.assert_close(beam_spurions(p, frame="jet")[0], b, rtol=0.0, atol=0.0)


def test_beam_mass_and_degenerate_jets_fall_back_to_the_lab_frame():
    p = torch.zeros(4, 3, 4, dtype=torch.float64)
    p[1, 0] = torch.tensor([3.0, 0.0, 0.0, 3.0])         # one constituent along the beam: M^2 = 0
    p[2, 0] = torch.tensor([2.0, 1.0, 0.0, 1.0])         # one constituent, M^2 = 3
    # jets 0 and 3 have no constituent at all (E = 0)
    jet, lab = beam_spurions(p, frame="jet"), beam_spurions(p, frame="lab")
    for k in (0, 1, 3):
        assert torch.equal(jet[k], lab[k])
    assert (jet[2] - lab[2]).abs().max() > 0.1
    assert torch.isfinite(jet).all()
    # beam_mass sets the mass of the beams, <b, b> = beam_mass^2, in both frames
    for mass in (0.5, 2.0):
        for frame in ("lab", "jet"):
            b = beam_spurions(p, beam_mass=mass, frame=frame)
            for k in range(4):
                for beam in (0, 1):
                    assert abs(minkowski_normsq(b[k, beam]).item() - mass * mass) < 1e-12


def test_psi_and_minkowski_products_follow_the_official_formulas():
    a = torch.tensor([-1e6, -3.0, -1.0, -1e-9, 0.0, 1e-9, 0.5, 1.0, 7.0, 1e6], dtype=torch.float64)
    torch.testing.assert_close(psi(a), _psi(a), rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(psi(-a), -psi(a), rtol=0.0, atol=0.0)             # odd
    assert abs(psi(torch.tensor(1.0, dtype=torch.float64)).item() - LOG2) < 1e-15
    assert psi(torch.tensor(1e-12, dtype=torch.float64)).item() == pytest.approx(1e-12, rel=1e-9)
    g = torch.Generator().manual_seed(0)
    p = torch.randn(4, 6, 4, generator=g, dtype=torch.float64)
    q = torch.randn(4, 6, 4, generator=g, dtype=torch.float64)
    torch.testing.assert_close(minkowski_dot(p, q), _dotsq4(p, q), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(minkowski_normsq(p), _normsq4(p), rtol=1e-13, atol=1e-13)
    pair = pairwise_minkowski_dot(p)
    for b in range(4):
        for i in range(6):
            for j in range(6):
                assert abs(pair[b, i, j].item() - _dotsq4(p[b, i], p[b, j]).item()) < 1e-12
    # signature (+, -, -, -): a purely timelike vector has positive norm
    assert minkowski_normsq(torch.tensor([2.0, 0.0, 0.0, 0.0])).item() == 4.0
    assert minkowski_normsq(torch.tensor([0.0, 1.0, 1.0, 1.0])).item() == -3.0


def test_masked_rows_are_ignored_whatever_they_contain(jets):
    pad = jets[..., 4] == 0
    assert pad.any()
    garbage = jets.clone()
    g = torch.Generator().manual_seed(1)
    vals = 50.0 * torch.randn(int(pad.sum()), 4, generator=g, dtype=jets.dtype)
    vals[0, 0], vals[-1, 2], vals[1, 3] = float("nan"), float("inf"), -float("inf")
    garbage[..., :4][pad] = vals
    model = _generic(0)
    ref = _logits(model, jets)
    assert torch.isfinite(ref).all()
    _assert_same(_logits(model, garbage), ref, tol=1e-12)
    coords, scalars, mask = model.graph(garbage)
    assert torch.isfinite(coords).all() and torch.isfinite(scalars).all()
    assert torch.equal(mask[:, 2:], jets[..., 4] > 0) and mask[:, :2].all()
    assert (coords[:, 2:][~mask[:, 2:]] == 0).all()


# ─────────────────────────────────────────────────────────────────────────
# Permutations and padding
# ─────────────────────────────────────────────────────────────────────────

def test_permutation_invariance(jets):
    model = _generic(0)
    ref = _logits(model, jets)
    shuffled = _real_shuffle(jets)
    assert not torch.equal(shuffled, jets)
    _assert_same(_logits(model, shuffled), ref, tol=1e-10)
    # all K slots shuffled: padding interleaved with the real constituents
    g = torch.Generator().manual_seed(0)
    _assert_same(_logits(model, jets[:, torch.randperm(jets.shape[1], generator=g)]),
                 ref, tol=1e-10)


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_padding_changes_nothing(jets, mode):
    """Extra zero rows: same logits, same gradients, same BatchNorm statistics."""
    a = _generic(2, dropout=0.0)
    a.train(mode == "train")
    b = copy.deepcopy(a)
    y = _labels(len(jets))
    loss_a, grads_a = _loss_and_grads(a, jets, y)
    loss_b, grads_b = _loss_and_grads(b, _pad(jets, 9), y)
    assert abs(loss_a.item() - loss_b.item()) < 1e-12
    scale = max(g.abs().max().item() for g in grads_a)
    for (name, _), ga, gb in zip(a.named_parameters(), grads_a, grads_b):
        assert (ga - gb).abs().max().item() <= 1e-10 * scale, name
    for (name, u), (_, v) in zip(a.named_buffers(), b.named_buffers()):
        torch.testing.assert_close(u.double(), v.double(), rtol=1e-12, atol=1e-12,
                                   msg=lambda m, name=name: f"{name}: {m}")
    # and the padding really is there: the slots of the second batch are all zero
    assert (_pad(jets, 9)[:, jets.shape[1]:] == 0).all()


def test_logits_do_not_depend_on_the_number_of_slots(jets):
    model = _generic(0)
    ref = _logits(model, jets)
    for slots in (1, 9, 40):
        _assert_same(_logits(model, _pad(jets, slots)), ref, tol=1e-10)
    # trailing padding slots removed from the jets that have some (<= 31 constituents)
    ragged = jets[:32]
    assert ragged[..., 4].sum(dim=1).max() <= 31
    _assert_same(_logits(model, ragged[:, :31]), _logits(model, ragged), tol=1e-10)


# ─────────────────────────────────────────────────────────────────────────
# Parameter counts, names, arguments
# ─────────────────────────────────────────────────────────────────────────

def _count_formula(L: int, H: int) -> int:
    return (L - 1) * (7 * H * H + 14 * H + 1) + 7 * H * H + 18 * H + 3


def test_parameter_count_through_the_registry():
    model = build_model("lorentznet", in_features=4, out_features=2,
                        representation="constituents")
    assert isinstance(model, LorentzNet)
    n = count_parameters(model)
    assert 4500 <= n <= 5500
    assert n == N_PARAMS == _count_formula(6, 10)
    assert (model.n_hidden, model.n_layers) == (10, 6)
    assert [lgeb.c_weight for lgeb in model.LGEBs] == [5e-3] * 6
    assert model.graph_dec[2].p == 0.2
    assert model.embedding.in_features == 2 and model.beam_frame == "jet"
    assert model_family("lorentznet") == "baseline_set" and is_set_model("lorentznet")
    assert all(p.dtype == torch.float64 for p in model.parameters())
    reg = model.regularization_loss()
    assert reg.shape == () and reg.dtype == torch.float64 and reg.item() == 0.0
    assert [lgeb.last_layer for lgeb in model.LGEBs] == [False] * 5 + [True]


def test_official_configuration_parameter_count():
    """H = 72, L = 6, the official top-tagging configuration: the paper's 224k."""
    model = LorentzNet(n_hidden=72, n_layers=6)
    assert count_parameters(model) == OFFICIAL_N_PARAMS == _count_formula(6, 72)


def test_parameter_names_and_shapes_match_the_official_state_dict():
    H, L = 72, 2
    model = LorentzNet(n_hidden=H, n_layers=L)
    sd = model.state_dict()
    bn = lambda prefix, n: {f"{prefix}.{k}": s for k, s in
                            (("weight", (n,)), ("bias", (n,)), ("running_mean", (n,)),
                             ("running_var", (n,)), ("num_batches_tracked", ()))}
    want = {"embedding.weight": (H, 2), "embedding.bias": (H,),
            "graph_dec.0.weight": (H, H), "graph_dec.0.bias": (H,),
            "graph_dec.3.weight": (2, H), "graph_dec.3.bias": (2,)}
    for k in range(L):
        p = f"LGEBs.{k}"
        want.update({
            f"{p}.phi_e.0.weight": (H, 2 * H + 2),                  # no bias
            f"{p}.phi_e.3.weight": (H, H), f"{p}.phi_e.3.bias": (H,),
            f"{p}.phi_h.0.weight": (H, 2 * H + 2), f"{p}.phi_h.0.bias": (H,),
            f"{p}.phi_h.3.weight": (H, H), f"{p}.phi_h.3.bias": (H,),
            f"{p}.phi_m.0.weight": (1, H), f"{p}.phi_m.0.bias": (1,),
        })
        want.update(bn(f"{p}.phi_e.1", H))
        want.update(bn(f"{p}.phi_h.1", H))
        if k < L - 1:                       # the last LGEB has no phi_x
            want.update({f"{p}.phi_x.0.weight": (H, H), f"{p}.phi_x.0.bias": (H,),
                         f"{p}.phi_x.2.weight": (1, H)})      # no bias
    assert {k: tuple(v.shape) for k, v in sd.items()} == want


def test_phi_x_output_layer_is_initialised_with_gain_0001():
    torch.manual_seed(0)
    model = LorentzNet(n_hidden=72, n_layers=3)
    for lgeb in list(model.LGEBs)[:-1]:
        w = lgeb.phi_x[2].weight
        bound = 0.001 * math.sqrt(6.0 / (72 + 1))
        assert w.abs().max().item() <= bound and w.abs().max().item() > 0.7 * bound


@pytest.mark.parametrize("kwargs", [
    {"n_scalar": 3}, {"n_layers": 0}, {"beam_frame": "detector"},
])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LorentzNet(**kwargs)
    with pytest.raises(ValueError):
        beam_spurions(torch.zeros(1, 2, 4), frame="detector")


def test_momentum_scale_rescales_the_inputs(jets):
    torch.manual_seed(0)
    a = LorentzNet(momentum_scale=0.37, beam_frame="lab").eval()
    b = copy.deepcopy(a)
    b.momentum_scale = 1.0
    scaled = jets.clone()
    scaled[..., :4] *= 0.37
    _assert_same(_logits(a, jets), _logits(b, scaled), tol=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# Finite gradients and numerical safety
# ─────────────────────────────────────────────────────────────────────────

def _degenerate_jets(jets: torch.Tensor) -> torch.Tensor:
    """One-constituent jet, all-padding jet, jet along the beam, jet with two constituents."""
    out = torch.zeros(4, jets.shape[1], 5, dtype=jets.dtype)
    out[0, 0] = jets[0, 0]                                        # one constituent
    # out[1] stays all padding
    out[2, 0] = torch.tensor([3.0, 0.0, 0.0, 3.0, 1.0], dtype=jets.dtype)   # pT = 0, M^2 = 0
    out[3, :2] = jets[1, :2]
    return out


def test_finite_gradients_with_degenerate_jets_in_the_batch(jets):
    torch.manual_seed(0)
    model = build_model("lorentznet", in_features=4, out_features=2,
                        representation="constituents")
    model.train()
    batch = torch.cat([jets[:12], _degenerate_jets(jets)])
    y = _labels(len(batch))
    loss, grads = _loss_and_grads(model, batch, y)
    assert _all_finite(loss, model)
    assert torch.sqrt(sum(g.pow(2).sum() for g in grads)) > 0
    model.eval()
    assert torch.isfinite(_logits(model, batch)).all()


def test_finite_gradients_for_a_batch_of_only_degenerate_jets(jets):
    model = _generic(0, dropout=0.0).train()
    for batch in (_degenerate_jets(jets), _degenerate_jets(jets)[1:2], jets[:1],
                  torch.zeros(3, jets.shape[1], 5, dtype=jets.dtype)):
        loss, _ = _loss_and_grads(model, batch, _labels(len(batch)))
        assert _all_finite(loss, model)
        assert torch.isfinite(_logits(model.eval(), batch)).all()
        model.train()


def test_degenerate_jets_do_not_disturb_the_other_jets_in_eval_mode(jets):
    """Eval mode treats every jet on its own: the logits of a jet do not depend on its batch."""
    model = _generic(0)
    ref = _logits(model, jets)
    mixed = torch.cat([jets, _degenerate_jets(jets)])
    out = _logits(model, mixed)
    _assert_same(out[:len(jets)], ref, tol=1e-12)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_no_blow_up_on_the_hardest_constituents(toy_split, dtype):
    X = toy_split.X_train
    norm = X[..., :4].norm(dim=-1).max(dim=1).values
    hardest = torch.argsort(norm, descending=True)[:128]
    batch = X[hardest].to(dtype)
    assert 45.0 < norm.max().item() < 70.0                        # "norm up to ~50"
    y = _labels(len(batch))
    for scale in (1.0, 20.0):
        x = batch.clone()
        x[..., :4] *= scale                                       # norms up to ~1000
        torch.manual_seed(0)
        model = LorentzNet(dtype=dtype).train()
        loss, _ = _loss_and_grads(model, x, y)
        assert _all_finite(loss, model), (dtype, scale)
        assert loss.item() < 10.0
        model.eval()
        out = _logits(model, x)
        assert out.dtype == dtype and torch.isfinite(out).all()


def test_coordinate_update_is_bounded_by_the_clamp():
    """x_i += c mean_j clamp((x_i - x_j) phi_x(m_ij), -100, 100): at most c * 100 per component."""
    torch.manual_seed(0)
    lgeb = LGEB(4, 4, 4, n_node_attr=2, c_weight=0.5).double()
    with torch.no_grad():
        lgeb.phi_x[2].weight.fill_(10.0)                          # huge phi_x
        lgeb.phi_x[0].bias.fill_(1.0)
    x = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1e3, -1e3, 2e3, 5e2],
                       [-4e3, 0.0, 1e3, 3e3]]], dtype=torch.float64)
    m = torch.ones(1, 3, 3, 4, dtype=torch.float64)
    edge_w = 1.0 - torch.eye(3, dtype=torch.float64).unsqueeze(0)
    x_diff = x.unsqueeze(2) - x.unsqueeze(1)
    new = lgeb.x_model(x, x_diff, m, edge_w)
    shift = (new - x).abs()
    assert shift.max().item() <= 0.5 * 100.0 * (1 + 1e-12)
    assert shift.max().item() > 0.5 * 99.0                        # and it is reached
    # without large differences the update is the plain mean
    small = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0],
                           [0.0, 0.02, 0.0, 0.0]]], dtype=torch.float64)
    with torch.no_grad():
        lgeb.phi_x[2].weight.fill_(0.1)
    phi = lgeb.phi_x(m)                                           # (1, 3, 3, 1)
    d = small.unsqueeze(2) - small.unsqueeze(1)
    want = small + 0.5 * (d * phi * edge_w.unsqueeze(-1)).sum(dim=2) / 2.0
    torch.testing.assert_close(lgeb.x_model(small, d, m, edge_w), want, rtol=1e-13, atol=1e-15)


# ─────────────────────────────────────────────────────────────────────────
# dtype, evaluation chunks
# ─────────────────────────────────────────────────────────────────────────

def test_float32_model_runs_and_agrees_with_float64(jets):
    m64 = _generic(0)
    m32 = LorentzNet(dtype=torch.float32, c_weight=0.2)
    m32.load_state_dict({k: v.float() if v.is_floating_point() else v
                         for k, v in m64.state_dict().items()})
    assert all(p.dtype == torch.float32 for p in m32.parameters())
    assert m32.regularization_loss().dtype == torch.float32
    m32.eval()
    out32 = _logits(m32, jets)                      # float64 input into a float32 model
    out64 = _logits(m64, jets)
    assert out32.dtype == torch.float32 and torch.isfinite(out32).all()
    assert _max_diff(out32.double(), out64) < 1e-3
    assert out32.shape == (len(jets), 2)
    m32.train()
    loss, _ = _loss_and_grads(m32, jets.float(), _labels(len(jets)))
    assert loss.dtype == torch.float32 and _all_finite(loss, m32)
    assert all(p.grad.dtype == torch.float32 for p in m32.parameters())


def test_float32_inputs_stay_finite_in_a_float64_model(jets):
    x32 = jets.float()
    E, p = x32[..., 0].double(), x32[..., 1:4].double().norm(dim=-1)
    assert ((E < p) & (x32[..., 4] > 0)).any()          # rounding puts E below |p| (m^2 < 0)
    model = _generic(0)
    ref = _logits(model, jets)
    out = _logits(model, x32)
    assert out.dtype == torch.float64 and torch.isfinite(out).all()
    assert _max_diff(out, ref) < 1e-2


def test_chunked_evaluation_matches_one_batch(jets):
    model = _generic(0)
    model.eval_chunk_size = None
    ref = _logits(model, jets)
    model.eval_chunk_size = 10                                    # 48 jets -> 5 chunks
    _assert_same(_logits(model, jets), ref, tol=1e-12)
    out = predict_logits(model, jets, chunk_size=20)
    _assert_same(out, ref, tol=1e-12)
    # chunking only applies to evaluation mode without autograd: BatchNorm batch
    # statistics need the whole batch
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    with torch.no_grad():
        chunked = model(jets)
    model.eval_chunk_size = None
    with torch.no_grad():
        whole = model(jets)
    assert torch.equal(chunked, whole)


# ─────────────────────────────────────────────────────────────────────────
# Short training on toy jets
# ─────────────────────────────────────────────────────────────────────────

def test_short_training_separates_toy_jets(toy_split):
    pytest.importorskip("sklearn")            # evaluate_test's AUC needs it
    s = toy_split
    torch.manual_seed(0)
    model = build_model("lorentznet", in_features=4, out_features=2,
                        representation="constituents")
    # Running statistics that forget their initial values within a few steps
    # (see the module docstring: 33 steps at momentum 0.1 are not enough).
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = 0.5
    res = train_classifier(model, s.X_train, s.y_train, s.X_val, s.y_val,
                           TrainConfig(epochs=3, batch_size=128, lr=3e-3, seed=0))
    assert not res.diverged and res.nonfinite_steps == 0
    assert res.n_params == N_PARAMS and res.epochs_run == 3
    out = evaluate_test(model, s.X_test, s.y_test, n_classes=2)
    assert out["test_auc"] >= 0.85, out
