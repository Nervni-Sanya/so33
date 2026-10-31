"""
benchmarks.lorentznet
---------------------
LorentzNet baseline for top tagging at ~5k parameters: Gong, Meng, Zhang, Qu,
Li, Qian, Du, Ma, Liu, "An efficient Lorentz equivariant graph neural network
for jet tagging", JHEP 07 (2022) 030, arXiv:2201.08187. Reference
implementation: github.com/sdogsq/LorentzNet-release, ``models.py`` (classes
``LGEB`` and ``LorentzNet``; ``psi``, ``normsq4``, ``dotsq4``,
``unsorted_segment_sum`` / ``unsorted_segment_mean``) and ``top_tagging.py``
(how the network is fed and configured for top tagging).

Registered in ``benchmarks.models`` as "lorentznet" (family ``baseline_set``),
so ``benchmarks.run_top_tagging --representation constituents`` trains it under
the same protocol as every other per-constituent model: the leading 32
constituents, globally RMS-normalised, Adam 3e-3 with a cosine schedule, batch
128, 30 epochs, gradient clipping at 1, last-epoch model.

Input. ``forward`` takes the harness tensor x (B, K, 5): ``x[..., :4]`` =
(E, px, py, pz) divided by one global scale, ``x[..., 4]`` = mask (1 real
constituent, 0 padding). Masked rows are set to zero first, whatever they
contain. Unlike the PFN / EFN, the output depends on the global scale: psi is
not scale free and the beams have a fixed size in input units.

Graph, one per jet, N = K + 2 nodes (``lorentznet_graph``):
  - nodes 0, 1 are the two beam spurions, nodes 2 .. K + 1 the constituents;
  - coordinates x_i = four-momenta. The beams are
        b_+- = sqrt(1 + m_b^2) u +- w,   m_b = 1 (``beam_mass``).
    The official code uses u = e_t, w = e_z, i.e. b_+- = (sqrt 2, 0, 0, +-1)
    in the units of the momenta (``beam_frame="lab"``). The default
    ``beam_frame="jet"`` uses the same vectors in the jet's longitudinal rest
    frame: u = (E_J, 0, 0, pz_J) / M, w = (pz_J, 0, 0, E_J) / M,
    M^2 = E_J^2 - pz_J^2, J = sum_i p_i over the real constituents (e_t and
    e_z boosted along z by the jet rapidity);
  - scalars s_i = psi((m_i^2, l_i)), with m_i^2 = <x_i, x_i> and the label
    l = +1 for the beams, -1 for the constituents (and padding). For massless
    constituents and m_b = 1 this gives s = (log 2, log 2) for the beams and
    (0, -log 2) for every constituent;
  - edges: every ordered pair (i, j), i != j, of real nodes, beams included.
    Padded nodes send and receive no messages, take no part in the BatchNorm
    statistics and are left out of the pooling.

Architecture (models.py), with <p, q> = p0 q0 - p1 q1 - p2 q2 - p3 q3 (metric
(+, -, -, -)), ||p||^2 = <p, p> and psi(a) = sgn(a) log(|a| + 1):
    h_i = embedding(s_i)                                    Linear(2 -> H)
    for each of the L LGEBs (Lorentz group equivariant blocks):
      m_ij = phi_e(h_i, h_j, psi(||x_i - x_j||^2), psi(<x_i, x_j>))
             phi_e = Linear(2H + 2 -> H, no bias) - BN - ReLU - Linear - ReLU
      m_ij <- phi_m(m_ij) m_ij                              attention weight,
             phi_m = Linear(H -> 1) - sigmoid
      x_i  <- x_i + c mean_{j != i} clamp((x_i - x_j) phi_x(m_ij), -100, 100)
             phi_x = Linear - ReLU - Linear(H -> 1, no bias, Xavier gain
             0.001); the last LGEB has no phi_x and no coordinate update
      h_i  <- h_i + phi_h(h_i, sum_{j != i} m_ij, s_i)
             phi_h = Linear(2H + 2 -> H) - BN - ReLU - Linear(H -> H)
    logits = graph_dec(mean over the real nodes of h_i)
             graph_dec = Linear(H -> H) - ReLU - Dropout - Linear(H -> 2)
The sums and means over j run over the real neighbours of i (the official
``unsorted_segment_sum`` / ``unsorted_segment_mean`` over the edge list). All
edge features are Minkowski products and the coordinate update is covariant,
so everything after the graph construction (``graph_logits``) is Lorentz
invariant, exactly as long as the componentwise clamp of the coordinate update
(a safety net) does not act, and the network as a whole is invariant up to the
beams. Jet-frame beams move with the jet under boosts along the beam axis,
which makes the logits invariant under longitudinal boosts and rotations about
the beam axis (and the reflections z -> -z, px -> -px, py -> -py); a transverse
boost changes them. The official lab-frame beams are fixed four-vectors that a
longitudinal boost does not preserve (no nonzero vector in the (t, z) plane is
invariant), so with them only the rotations about the beam axis and the
reflections survive.

Sizes. Trainable parameters for L LGEBs of width H, 2 scalars, 2 classes:
    (L - 1)(7 H^2 + 14 H + 1) + 7 H^2 + 18 H + 3.
Default H = 10, L = 6: 5,088 (this keeps the official depth of 6 LGEBs and
scales the width 72 -> 10). The official top-tagging configuration H = 72,
L = 6 gives 224,072 (the paper's "224k"). Initialisation as in the official
code: PyTorch defaults, Xavier-uniform (gain 0.001) for the last phi_x layer;
the modules are created in the official order (the last LGEB builds phi_x and
deletes it), and parameter names match the official state dict.

Departures from the reference:
  - width H = 10 instead of 72 (5,088 instead of 224,072 parameters); the
    official top-tagging c = 5e-3 and dropout 0.2 are kept (``top_tagging.py``
    defaults; the class default in models.py is c = 1e-3);
  - beams in the jet's longitudinal rest frame by default, so that the logits
    are invariant under longitudinal boosts; the official lab-frame beams
    (``beam_frame="lab"``) break that symmetry, and the network then has to
    learn rapidity independence from the |eta| < 2 spread of the jets;
  - mass scalar m_i^2 = <x_i, x_i> instead of the mass m_i = sqrt|<x_i, x_i>|,
    which is what we take the first official node scalar to be (the ``top/``
    loader that builds it was not available; see the node-recipe item below).
    Both are 1 for the beams (m_b = 1) and 0 for massless constituents in exact
    arithmetic. For massless constituents, the square root turns the rounding
    noise of E^2 - |p|^2 into a spurious m_i ~ 1e-8 E_i (float64) to
    ~ 3e-4 E_i (the float32-stored data), which changes under boosts;
  - units: the protocol's RMS-normalised momenta instead of GeV. The beams
    keep the official components (sqrt 2, +-1), now in input units, so they
    are RMS / (1 GeV) times harder relative to the constituents than in the
    official code (the toy jets of ``benchmarks.synthetic`` have RMS ~ 46
    GeV), and the +-100 clamp is in input units (a safety net, as in the
    official code: it bounds every coordinate update by c * 100 = 0.5 per
    layer). ``momentum_scale`` multiplies the inputs; pass the training RMS in
    GeV to recover the official GeV convention;
  - padding: padded nodes send and receive no messages and are left out of
    the pooling and of every BatchNorm statistic. The pooling is the mean
    over the real nodes (beams and constituents). The official code
    multiplies h by the node mask and averages over all node slots
    (``view(-1, n_nodes, hidden).mean(1)``), so the padded slots count in the
    denominator, which is the padded length of the batch; that would make the
    output depend on the number of padding slots here. The phi_h BatchNorm
    statistics use the real nodes only (official: every node slot, padding
    included). The phi_e statistics use the real edges: the official LGEB
    applies no mask (``LorentzNet.forward`` takes an ``edge_mask`` and never
    uses it), so the edge list built by the ``top/`` loader has to hold
    only the real edges for padded nodes to be inert, and we take that to be
    the case (the loader was not available to check);
  - only the leading 32 constituents of a jet (official: all, up to 200);
  - training: this repository's protocol (see above, float64, batch 128 on
    one device) instead of AdamW (lr 1e-3, weight decay 0.01), 5 warm-up
    epochs + CosineAnnealingWarmRestarts(4, 2) and halving after epoch 31,
    35 epochs, batch 4 GPUs x 32 with SyncBatchNorm, the
    best-validation-accuracy checkpoint, float32. The BatchNorm batch is 128
    jets in both;
  - implementation: dense masked (B, N, N, H) tensors instead of edge lists.
    The first phi_e layer is applied as W_i h_i + W_j h_j + W_e (norms,
    dots), so the (B, N, N, 2H + 2) concatenation is never built. The edge
    mask is folded into the attention weight (m_ij = 0 off the edges). The
    masked BatchNorm has a hand-written backward (the same gradient, fewer
    passes over the edge tensor), and psi uses log1p. ``forward`` returns
    (B, out_features) logits (the official ``pred.squeeze(1)`` is the same
    for 2 classes), and in eval mode without autograd it evaluates batches
    larger than ``eval_chunk_size`` jets in chunks, which changes no logit and
    only bounds the memory and the cache footprint. tests/test_lorentznet.py
    checks the dense forward and its gradients against an edge-list
    implementation of the official algorithm;
  - the node recipe (beams, label) follows the paper and the LGN data
    pipeline that the official ``top/`` loader builds on, as far as we know
    it: beams (sqrt(1 + m_b^2), 0, 0, +-1) prepended to the constituents
    (``add_beams=True, beam_mass=1, scale=1``), label +1 beam / -1 particle,
    node mask E != 0, and ``top_tagging.py`` applies psi to the node scalars
    (``nodes = psi(nodes)``, which is what ``lorentznet_graph`` does). The
    ``top/`` code itself was not available to check, so the beam vectors,
    the label values and the mass convention are the part of the node recipe
    that rests on the paper and on memory of the LGN code rather than on the
    release;
  - the paper writes the coordinate update as x_i + c sum_j phi_x(m_ij) x_j.
    The official code, followed here, uses c mean_j phi_x(m_ij) (x_i - x_j).
    The ``dropout`` argument acts only in the decoder, as in the official
    code (LGEB accepts it but does not use it).

Short runs and per-epoch validation. BatchNorm keeps PyTorch's defaults, as in
the official code (momentum 0.1, running variance initialised to 1). At
initialisation some first-layer edge channels have a batch variance of only
~1e-4, so after a few dozen steps the running variance still holds a few
percent of its initial value 1, and while the learning rate is high the
running statistics lag the weights: the evaluation-mode logits can be far from
the training-mode ones even though the network has learned the task. On the
toy jets, 3 epochs on 1,400 jets give a test AUC between 0.43 and 0.97
depending on the seed (0.99 once the running statistics are refreshed with the
final weights; tests/test_lorentznet.py raises the momentum to 0.5 for that
reason), and over 6 epochs on 14,000 jets the validation accuracy goes
0.51, 0.51, 0.24, 0.71, 0.51, 0.98 with a training accuracy of 0.95 - 0.985
throughout (final test AUC 0.998). So the per-epoch ``val_acc`` of the
harness records is not meaningful for this model until the cosine schedule
has annealed the learning rate; the last-epoch model that the protocol
evaluates is not affected (canonical runs take hundreds of thousands of steps
and end at a learning rate of ~1e-5).

Run:
    # Canonical protocol, three seeds, with per-jet scores:
    python -m benchmarks.run_top_tagging --representation constituents \\
        --canonical-splits --models lorentznet --seeds 0-2 --save-scores

    # Tests (symmetries, padding, parameter count, official edge-list
    # cross-check, short toy training):
    python -m pytest tests/test_lorentznet.py -v
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "N_SCALAR", "N_HIDDEN", "N_LAYERS", "C_WEIGHT", "DROPOUT", "BEAM_MASS",
    "BEAM_FRAMES", "PAPER_N_HIDDEN", "PAPER_N_LAYERS", "EVAL_CHUNK_JETS",
    "psi", "minkowski_dot", "minkowski_normsq", "pairwise_minkowski_dot",
    "beam_spurions", "lorentznet_graph", "LGEB", "LorentzNet",
]

# Defaults (~5k parameters) and the official top-tagging configuration.
N_SCALAR = 2            # node scalars: (m^2, beam label)
N_HIDDEN = 10
N_LAYERS = 6
C_WEIGHT = 5e-3         # top_tagging.py --c_weight
DROPOUT = 0.2           # top_tagging.py --dropout
BEAM_MASS = 1.0         # collate_fn(..., add_beams=True, beam_mass=1)
BEAM_FRAMES = ("jet", "lab")
PAPER_N_HIDDEN = 72
PAPER_N_LAYERS = 6

# Jets per chunk when forward runs in eval mode without autograd. A chunk
# changes no logit (eval mode has no batch statistics); it bounds the memory
# (one (chunk, N, N, H) float64 edge tensor is ~12 MB at K = 32) and keeps the
# elementwise passes over the edge tensors in cache: on one CPU thread 128 jets
# per chunk evaluate ~30% faster than 512.
EVAL_CHUNK_JETS = 128

# Coordinate updates are clamped component-wise to +-CLAMP (official x_model).
CLAMP = 100.0


# ─────────────────────────────────────────────────────────────────────────
# Minkowski geometry and the graph
# ─────────────────────────────────────────────────────────────────────────

def psi(a: torch.Tensor) -> torch.Tensor:
    """psi(a) = sgn(a) log(|a| + 1), elementwise (official ``psi``)."""
    return torch.sign(a) * torch.log1p(a.abs())


def minkowski_dot(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """<p, q> = p0 q0 - p1 q1 - p2 q2 - p3 q3 over the last axis (``dotsq4``)."""
    return p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(dim=-1)


def minkowski_normsq(p: torch.Tensor) -> torch.Tensor:
    """||p||^2 = <p, p> over the last axis (``normsq4``)."""
    return minkowski_dot(p, p)


def pairwise_minkowski_dot(x: torch.Tensor) -> torch.Tensor:
    """(B, N, N) matrix of <x_i, x_j> for coordinates x (B, N, 4)."""
    metric = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=x.dtype, device=x.device)
    return torch.einsum("bik,bjk->bij", x * metric, x)


def beam_spurions(
    p: torch.Tensor,
    beam_mass: float = BEAM_MASS,
    frame: str = "jet",
) -> torch.Tensor:
    """Four-momenta (B, 2, 4) of the two beam spurions for jets p (B, K, 4).

    b_+- = sqrt(1 + beam_mass^2) u +- w. ``frame="lab"``: u = e_t, w = e_z,
    the official (sqrt(1 + m^2), 0, 0, +-1). ``frame="jet"``: u, w are e_t, e_z
    boosted along z by the rapidity of J = sum_i p_i (rows of p that are zero
    padding add nothing), u = (E_J, 0, 0, pz_J) / M, w = (pz_J, 0, 0, E_J) / M,
    M = sqrt((E_J - pz_J)(E_J + pz_J)): the official vectors in the frame where
    the jet has pz = 0. A jet with M^2 <= 0 or E_J <= 0 (no real constituent)
    gets the lab-frame beams.
    """
    B = p.shape[0]
    zero = p.new_zeros(B)
    if frame == "lab":
        cosh_y, sinh_y = p.new_ones(B), zero
    elif frame == "jet":
        jet = p.sum(dim=1)
        e_j, pz_j = jet[:, 0], jet[:, 3]
        mt2 = (e_j - pz_j) * (e_j + pz_j)
        ok = (mt2 > 0) & (e_j > 0)
        mt = torch.sqrt(torch.where(ok, mt2, 1.0))
        cosh_y = torch.where(ok, e_j / mt, 1.0)
        sinh_y = torch.where(ok, pz_j / mt, 0.0)
    else:
        raise ValueError(f"beam frame must be one of {BEAM_FRAMES}, got {frame!r}")
    u = torch.stack([cosh_y, zero, zero, sinh_y], dim=-1)
    w = torch.stack([sinh_y, zero, zero, cosh_y], dim=-1)
    e0 = math.sqrt(1.0 + beam_mass ** 2)
    return torch.stack([e0 * u + w, e0 * u - w], dim=1)


def lorentznet_graph(
    x: torch.Tensor,
    *,
    momentum_scale: float = 1.0,
    beam_mass: float = BEAM_MASS,
    beam_frame: str = "jet",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinates, scalars and node mask of the LorentzNet graph of x.

    Parameters
    ----------
    x : (B, K, 5) harness tensor (E, px, py, pz, mask).

    Returns
    -------
    coords    : (B, K + 2, 4) four-momenta: the beams (``beam_spurions``) in
                slots 0, 1, then momentum_scale * the constituents (zero rows
                on padding).
    scalars   : (B, K + 2, 2) psi((<x_i, x_i>, label)), label +1 for the
                beams and -1 for the constituent slots (padding included;
                padded nodes never reach a real node).
    node_mask : (B, K + 2) bool, True for the beams and the real
                constituents.
    """
    real = x[..., 4] > 0
    p = torch.where(real.unsqueeze(-1), x[..., :4], 0.0) * momentum_scale
    coords = torch.cat([beam_spurions(p, beam_mass, beam_frame), p], dim=1)
    B, K = real.shape
    node_mask = torch.cat([real.new_ones(B, 2), real], dim=1)
    label = torch.cat([x.new_ones(B, 2), -x.new_ones(B, K)], dim=1)
    scalars = psi(torch.stack([minkowski_normsq(coords), label], dim=-1))
    return coords, scalars, node_mask


# ─────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────

class _MaskedBatchNorm(torch.autograd.Function):
    """Training-mode BatchNorm of x (M, C) with statistics from the rows w = 1.

    Forward: mu, var = mean and biased variance over the rows with w = 1,
    y = weight (x - mu) / sqrt(var + eps) + bias on every row, and the running
    statistics updated in place as nn.BatchNorm1d does (factor = momentum,
    unbiased variance). Backward: the BatchNorm gradient with the statistics
    restricted to those rows, with xhat = (x - mu) rstd and sums over all rows,
        dx = weight rstd (dy - w / n (sum dy + xhat sum dy xhat)),
    which is what autograd gives for the same forward, in a few fused passes
    over the (M, C) tensor instead of a dozen.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor, running_mean: torch.Tensor,
                running_var: torch.Tensor, factor: float, eps: float) -> torch.Tensor:
        n = w.sum()
        mean = (w @ x) / n
        var = (w @ (x - mean).square_()) / n
        rstd = torch.rsqrt(var + eps)
        running_mean.lerp_(mean, factor)
        running_var.lerp_(var * n / (n - 1.0).clamp(min=1.0), factor)
        scale = weight * rstd
        ctx.save_for_backward(x, w / n, weight, mean, rstd)
        return torch.addcmul(bias - mean * scale, x, scale)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w_n, weight, mean, rstd = ctx.saved_tensors
        xhat = (x - mean).mul_(rstd)
        dbias = dy.sum(dim=0)
        dweight = (dy * xhat).sum(dim=0)
        corr = xhat.mul_(dweight).add_(dbias).mul_(w_n.unsqueeze(1))
        dx = torch.sub(dy, corr).mul_(weight * rstd)
        return dx, None, dweight, dbias, None, None, None, None


def _masked_batch_norm(bn: nn.BatchNorm1d, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``bn`` over the last axis of x (..., C) with statistics from real rows.

    w (...) is 1 on the rows that count (real nodes or edges) and 0 elsewhere.
    Training mode: batch statistics of those rows (``_MaskedBatchNorm``), and
    the running statistics updated as nn.BatchNorm1d does; eval mode: the
    running statistics. The other rows go through the same affine map and are
    never read.
    """
    if not bn.training:
        scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
        return torch.addcmul(bn.bias - bn.running_mean * scale, x, scale)
    bn.num_batches_tracked.add_(1)
    factor = (bn.momentum if bn.momentum is not None
              else 1.0 / float(bn.num_batches_tracked))
    y = _MaskedBatchNorm.apply(x.reshape(-1, x.shape[-1]), w.reshape(-1),
                               bn.weight, bn.bias, bn.running_mean,
                               bn.running_var, factor, bn.eps)
    return y.view(x.shape)


class LGEB(nn.Module):
    """Lorentz group equivariant block (official ``LGEB``) on dense graphs.

    Same modules, initialisation and parameter names as the official class;
    ``forward`` works on (B, N, ...) tensors with node / edge weights instead
    of an edge list. ``dropout`` is accepted and unused, as in the official
    code.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_node_attr: int = 0,
        dropout: float = 0.0,
        c_weight: float = 1.0,
        last_layer: bool = False,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.c_weight = c_weight
        n_edge_attr = 2          # psi(Minkowski norm), psi(inner product)

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False, dtype=dtype),
            nn.BatchNorm1d(n_hidden, dtype=dtype),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, dtype=dtype),
            nn.ReLU())

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden, dtype=dtype),
            nn.BatchNorm1d(n_hidden, dtype=dtype),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output, dtype=dtype))

        layer = nn.Linear(n_hidden, 1, bias=False, dtype=dtype)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden, dtype=dtype),
            nn.ReLU(),
            layer)

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1, dtype=dtype),
            nn.Sigmoid())

        # Built and then deleted, as in the official code, so that the random
        # initialisation of the following modules is the same.
        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

    def m_model(self, h: torch.Tensor, norms: torch.Tensor, dots: torch.Tensor,
                edge_w: torch.Tensor) -> torch.Tensor:
        """Attention-weighted messages m_ij (B, N, N, H), exactly 0 off the edges.

        phi_e's first layer on (h_i, h_j, norms, dots) is W_i h_i + W_j h_j +
        W_e (norms, dots), built in one (B, N, N, H) buffer. The edge mask is
        folded into the attention weight, so the neighbour sums of the
        callers are plain sums over j.
        """
        lin, bn, _, lin2, _ = self.phi_e
        H = h.shape[-1]
        W = lin.weight                                  # (H, 2H + 2)
        out = F.linear(h, W[:, :H]).unsqueeze(2) + F.linear(h, W[:, H:2 * H]).unsqueeze(1)
        edge_attr = torch.stack([norms, dots], dim=-1)
        out.view(-1, H).addmm_(edge_attr.view(-1, 2), W[:, 2 * H:].t())
        out = F.relu_(_masked_batch_norm(bn, out, edge_w))
        out = F.relu_(lin2(out))
        return out * (self.phi_m(out) * edge_w.unsqueeze(-1))

    def x_model(self, x: torch.Tensor, x_diff: torch.Tensor, m: torch.Tensor,
                edge_w: torch.Tensor) -> torch.Tensor:
        """x_i + c mean_j clamp((x_i - x_j) phi_x(m_ij)) over real neighbours j."""
        trans = torch.clamp(x_diff * (self.phi_x(m) * edge_w.unsqueeze(-1)),
                            min=-CLAMP, max=CLAMP)
        count = edge_w.sum(dim=2).unsqueeze(-1).clamp(min=1.0)
        return x + trans.sum(dim=2) / count * self.c_weight

    def h_model(self, h: torch.Tensor, m: torch.Tensor, node_attr: torch.Tensor,
                node_w: torch.Tensor) -> torch.Tensor:
        """h_i + phi_h(h_i, sum_j m_ij, s_i) over real neighbours j."""
        lin, bn, act, lin2 = self.phi_h
        out = lin(torch.cat([h, m.sum(dim=2), node_attr], dim=-1))
        out = lin2(act(_masked_batch_norm(bn, out, node_w)))
        return h + out

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_attr: torch.Tensor,
        node_w: torch.Tensor,
        edge_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One block: (h, x) (B, N, H), (B, N, 4) -> updated (h, x).

        node_attr (B, N, S) are the scalars s_i; node_w (B, N) and edge_w
        (B, N, N) are 1 on real nodes / edges (i != j, both real) and 0
        elsewhere.
        """
        x_diff = x.unsqueeze(2) - x.unsqueeze(1)                    # x_i - x_j
        norms = psi(minkowski_normsq(x_diff))
        dots = psi(pairwise_minkowski_dot(x))
        m = self.m_model(h, norms, dots, edge_w)
        if not self.last_layer:
            x = self.x_model(x, x_diff, m, edge_w)
        h = self.h_model(h, m, node_attr, node_w)
        return h, x


class LorentzNet(nn.Module):
    """LorentzNet jet tagger on the harness tensor (B, K, 5).

    Parameters
    ----------
    out_features   : number of classes (2: logits for background / signal).
    dtype          : parameter and compute dtype (default float64).
    n_scalar       : node scalars per particle; must be 2 (m^2, beam label),
                     as in the official top-tagging script.
    n_hidden       : latent width H (official top tagging: 72).
    n_layers       : number of LGEBs L (official top tagging: 6).
    c_weight       : coordinate-update weight c (official top tagging: 5e-3).
    dropout        : decoder dropout (official top tagging: 0.2).
    beam_mass      : invariant mass of the beam spurions (official: 1).
    beam_frame     : "jet" (default) to attach the beams in the jet's
                     longitudinal rest frame, "lab" for the official
                     lab-frame beams (see the module docstring).
    momentum_scale : factor applied to the input momenta (input units -> the
                     units of the beams); 1 keeps the harness units.
    eval_chunk_size: jets per chunk in eval mode without autograd (None:
                     no chunking).

    Defaults: 5,088 trainable parameters with ``out_features=2``.
    """

    def __init__(
        self,
        out_features: int = 2,
        *,
        dtype: torch.dtype = torch.float64,
        n_scalar: int = N_SCALAR,
        n_hidden: int = N_HIDDEN,
        n_layers: int = N_LAYERS,
        c_weight: float = C_WEIGHT,
        dropout: float = DROPOUT,
        beam_mass: float = BEAM_MASS,
        beam_frame: str = "jet",
        momentum_scale: float = 1.0,
        eval_chunk_size: int | None = EVAL_CHUNK_JETS,
    ) -> None:
        super().__init__()
        if n_scalar != N_SCALAR:
            raise ValueError(f"n_scalar must be {N_SCALAR} (m^2 and the beam label), "
                             f"got {n_scalar}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if beam_frame not in BEAM_FRAMES:
            raise ValueError(f"beam_frame must be one of {BEAM_FRAMES}, got {beam_frame!r}")
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dtype = dtype
        self.beam_mass = float(beam_mass)
        self.beam_frame = beam_frame
        self.momentum_scale = float(momentum_scale)
        self.eval_chunk_size = eval_chunk_size

        self.embedding = nn.Linear(n_scalar, n_hidden, dtype=dtype)
        self.LGEBs = nn.ModuleList([
            LGEB(n_hidden, n_hidden, n_hidden, n_node_attr=n_scalar,
                 dropout=dropout, c_weight=c_weight,
                 last_layer=(i == n_layers - 1), dtype=dtype)
            for i in range(n_layers)])
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, out_features, dtype=dtype))

    def graph(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``lorentznet_graph`` of x with this model's beam settings."""
        return lorentznet_graph(x.to(self.dtype), momentum_scale=self.momentum_scale,
                                beam_mass=self.beam_mass, beam_frame=self.beam_frame)

    def graph_node_features(self, coords: torch.Tensor, scalars: torch.Tensor,
                            node_mask: torch.Tensor) -> torch.Tensor:
        """Final node features h (B, N, H) of a graph: the LGEB stack.

        coords (B, N, 4) four-momenta (the beams included), scalars (B, N, S)
        the node scalars and node_mask (B, N) bool the real nodes, as returned
        by ``graph``. Everything after the graph construction is Lorentz
        invariant: replacing ``coords`` by Lambda coords for any Lorentz
        transformation Lambda (the beams included) leaves h unchanged, as long
        as the componentwise clamp of the coordinate update does not act.
        """
        node_w = node_mask.to(coords.dtype)
        N = node_w.shape[1]
        no_self = 1.0 - torch.eye(N, dtype=coords.dtype, device=coords.device)
        edge_w = node_w.unsqueeze(2) * node_w.unsqueeze(1) * no_self
        h = self.embedding(scalars)
        for lgeb in self.LGEBs:
            h, coords = lgeb(h, coords, scalars, node_w, edge_w)
        return h

    def graph_logits(self, coords: torch.Tensor, scalars: torch.Tensor,
                     node_mask: torch.Tensor) -> torch.Tensor:
        """Logits (B, out_features) of a graph: LGEBs, mean pooling over the
        real nodes, decoder."""
        h = self.graph_node_features(coords, scalars, node_mask)
        w = node_mask.to(h.dtype)
        pooled = torch.einsum("bnh,bn->bh", h, w) / w.sum(dim=1, keepdim=True)
        return self.graph_dec(pooled)

    def node_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Final node features h (B, K + 2, H) and the node mask (B, K + 2)."""
        coords, scalars, node_mask = self.graph(x)
        return self.graph_node_features(coords, scalars, node_mask), node_mask

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.graph_logits(*self.graph(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B, out_features) of a (B, K, 5) batch (E, px, py, pz, mask)."""
        x = x.to(self.dtype)
        chunk = self.eval_chunk_size
        if (chunk and not self.training and not torch.is_grad_enabled()
                and x.shape[0] > chunk):
            return torch.cat([self._logits(xc) for xc in x.split(chunk)], dim=0)
        return self._logits(x)

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype)
