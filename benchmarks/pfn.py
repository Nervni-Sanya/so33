"""
benchmarks.pfn
--------------
Particle Flow Network (PFN) and Energy Flow Network (EFN) baselines for top
tagging at ~5k parameters: Komiske, Metodiev, Thaler, "Energy Flow Networks:
Deep Sets for Particle Jets", JHEP 01 (2019) 121, arXiv:1810.05165. Reference
implementation: energyflow 1.4.0, ``energyflow.archs.efn`` (classes ``PFN`` and
``EFN``) and the preprocessing helpers of ``energyflow.utils.particle_utils``
(``center_ptyphims``, ``rotate_ptyphims``, ``reflect_ptyphims``).

Both models are registered in ``benchmarks.models`` as "pfn" and "efn" (family
``baseline_set``), so ``benchmarks.run_top_tagging --representation
constituents`` trains them under the same protocol as every other
per-constituent model: the leading 32 constituents, globally RMS-normalised,
Adam 3e-3 with a cosine schedule, batch 128, 30 epochs, gradient clipping at 1,
last-epoch model.

Input. ``forward`` takes the harness tensor x (B, K, 5): ``x[..., :4]`` =
(E, px, py, pz) divided by one global scale, ``x[..., 4]`` = mask (1 real
constituent, 0 padding). The networks only see pT fractions and angles, so the
output does not depend on that scale (nor on whether the inputs are normalised
at all).

Kinematics, per constituent (masked rows are set to zero first):
    pT_i   = hypot(px_i, py_i),          z_i = pT_i / sum_j pT_j  (real j only),
    y_i    = asinh(pz_i / mT_i),         mT_i^2 = max(E_i^2 - pz_i^2, pT_i^2),
    phi_i  = azimuth relative to the jet's transverse momentum
             (sum_j px_j, sum_j py_j), as atan2(cross, dot).
The rapidity convention is that of ``benchmarks.run_efp_baseline``: it equals
energyflow's arctanh(pz / E) for any m^2 >= 0 and stays finite for the
float32-rounded massless constituents of the data, whose E can come out a hair
below |p| (the slightly negative m^2 is clamped, giving y = eta). The data's
constituents are massless, so y = eta: this is the "(pT, eta, phi)"
preprocessing. A particle with mT = 0 (zero momentum, or pT = 0 along the
beam) gets y = 0; it has z = 0 and drops out of every z-weighted sum.

Preprocessing (no learned parameters), the three particle_utils steps in the
order centre -> rotate -> reflect:
  1. centre (center_ptyphims(center="ptscheme")): subtract the pT-weighted
     centroid,
         Dy_i = y_i - sum_j z_j y_j,   Dphi_i = wrap(phi_i - sum_j z_j phi_j),
     with wrap() into (-pi, pi].
  2. rotate (rotate_ptyphims(rotate="ptscheme")): express every particle in
     the eigenbasis of the pT-weighted second-moment matrix
         M = sum_i z_i (Dy_i, Dphi_i)^T (Dy_i, Dphi_i).
     Its principal axis (larger eigenvalue) is e1 = (cos t, sin t) with
     t = atan2(2 M_y,phi, M_y,y - M_phi,phi) / 2, the minor axis
     e2 = (-sin t, cos t). The new coordinates are (v_i, u_i) = (r_i . e2,
     r_i . e1), r_i = (Dy_i, Dphi_i): the principal axis becomes the second
     ("vertical", phi) coordinate, as in energyflow, which returns
     r_i . eigvecs with numpy.linalg.eigh's ascending eigenvalue order.
  3. reflect (rotate_ptyphims's sign flip, then
     reflect_ptyphims(which="both")): u -> -u if sum_{u_i > 0} z_i <
     sum_{u_i < 0} z_i, and likewise v -> -v; afterwards the positive
     half-plane of each axis carries at least as much pT as the negative one.
     (energyflow negates both coordinates on the u test, then v on the v test
     and u on the u test, which is then a no-op; the end result is the same.)
The sign rules fix the eigenvector signs that eigh leaves arbitrary, so the
output equals energyflow's centre -> rotate -> reflect output up to rounding
(tests/test_pfn.py checks this on toy jets) except in exact ties: an isotropic
M (here e1 = y axis) or equal pT in two half-planes (here no flip), where
energyflow keeps LAPACK's choice. Masked rows get z = 0 and (v, u) = (0, 0).

Architecture (energyflow.archs.efn):
    PFN:  logits = out(F(sum_i mask_i Phi(z_i, v_i, u_i)))
    EFN:  logits = out(F(sum_i z_i    Phi(v_i, u_i)))
Phi and F are stacks of Linear + ReLU; the last Phi layer, which defines the
latent space, is ReLU-activated as well (energyflow applies Phi_acts to every
Phi layer). ``out`` is a Linear layer to the logits; energyflow adds a softmax
and trains with categorical cross-entropy, which is what nn.CrossEntropyLoss
computes from the logits. Initialisation as in energyflow / Keras: He-uniform
weights in Phi and F, Glorot-uniform weights in the output layer, zero
biases. Padding contributes exactly nothing: its Phi input is zero and its
weight (mask or z) is exactly 0. The EFN is IRC safe: a z = 0 particle has
weight 0, and splitting a particle collinearly (z -> f z, (1 - f) z at the
same position) leaves sum_i z_i Phi(r_i) unchanged; the centroid, M and the
half-plane pT sums are z-weighted, hence IRC safe as well. The PFN, whose Phi
also sees z_i and whose sum counts particles, is not.

Sizes. Defaults Phi = (24, 24, 64), F = (24, 24, 24): the paper's
Phi = (100, 100, l = 256), F = (100, 100, 100) with every width scaled by ~1/4,
keeping the depth (three Phi and three F layers) and a latent space wider than
the hidden layers. Trainable parameters:
    PFN   3 -> 24 -> 24 -> 64 | sum | 64 -> 24 -> 24 -> 24 -> 2    5,106
    EFN   2 -> 24 -> 24 -> 64 | sum | 64 -> 24 -> 24 -> 24 -> 2    5,082
(paper sizes: 82,458 for the PFN on (z, y, phi), 82,358 for the EFN).

Departures from the reference:
  - ~5k instead of ~82k parameters (sizes above);
  - training: this repository's protocol for every model (see above), in
    float64, instead of energyflow's defaults (Keras Adam, lr 1e-3, float32)
    and the paper's own schedule;
  - input: only the leading 32 constituents (by pT) of each jet, as for every
    model here (energyflow takes all particles of a jet; the Kasieczka files
    store up to 200), and no particle-ID channel (the data have none);
  - the preprocessing runs inside ``forward`` on (E, px, py, pz, mask), with
    the closed-form principal axis instead of eigh (same result, see above),
    the azimuths of every jet in one chart around the jet's pT direction
    (energyflow averages raw phi values and leaves the chart to the caller)
    and the rapidity convention above (energyflow: arctanh(pz / E), NaN for
    E < |pz| and infinite for pT = 0 along the beam);
  - padding is identified by the mask channel, not by all-zero feature rows
    (PFN, mask_val = 0) or z = 0 (EFN); for the EFN this is the same, for the
    PFN it only differs for a real particle at exactly (z, y, phi) = 0;
  - ``forward`` returns logits (no softmax layer), and without autograd it
    evaluates batches larger than ``eval_chunk_size`` jets in chunks (the
    harness evaluates the full canonical val / test sets, ~400k jets, in one
    call); that only bounds the memory and changes no number.

Run:
    # Canonical protocol, three seeds, with per-jet scores:
    python -m benchmarks.run_top_tagging --representation constituents \\
        --canonical-splits --models pfn,efn --seeds 0-2 --save-scores

    # Tests (invariances, IRC safety, parameter counts, short toy training):
    python -m pytest tests/test_pfn.py -v
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

__all__ = [
    "PHI_SIZES", "F_SIZES", "PAPER_PHI_SIZES", "PAPER_F_SIZES",
    "EVAL_CHUNK_JETS", "wrap_phi", "constituent_kinematics",
    "preprocess_constituents", "ParticleFlowNetwork", "EnergyFlowNetwork",
]

# Default widths (~5k parameters) and the paper's widths, for reference.
PHI_SIZES: tuple[int, ...] = (24, 24, 64)
F_SIZES: tuple[int, ...] = (24, 24, 24)
PAPER_PHI_SIZES: tuple[int, ...] = (100, 100, 256)
PAPER_F_SIZES: tuple[int, ...] = (100, 100, 100)

# Jets per chunk when forward runs without autograd (memory bound only).
EVAL_CHUNK_JETS = 4096

TWO_PI = 2.0 * math.pi


# ─────────────────────────────────────────────────────────────────────────
# Kinematics and preprocessing
# ─────────────────────────────────────────────────────────────────────────

def wrap_phi(phi: torch.Tensor) -> torch.Tensor:
    """Angles mapped into (-pi, pi] (equal to phi modulo 2 pi)."""
    return math.pi - torch.remainder(math.pi - phi, TWO_PI)


def constituent_kinematics(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pT, y, phi, real) of every slot of the harness tensor x (..., K, 5).

    ``real`` is ``x[..., 4] > 0``; masked rows are zeroed before anything is
    computed, so they get pT = y = phi = 0 whatever they contain. y is
    asinh(pz / mT) with mT^2 = max(E^2 - pz^2, pT^2), and 0 where mT = 0. phi
    is the azimuth relative to the jet's transverse momentum (the sum over the
    real constituents), atan2(P x p, P . p) with |phi| <= pi, so the jets
    never straddle the branch cut of an absolute azimuth.
    """
    real = x[..., 4] > 0
    p4 = torch.where(real.unsqueeze(-1), x[..., :4], torch.zeros_like(x[..., :4]))
    E, px, py, pz = p4.unbind(dim=-1)
    pt = torch.hypot(px, py)
    mt = torch.sqrt(torch.maximum(E * E - pz * pz, pt * pt))
    has_mt = mt > 0
    y = torch.where(has_mt, torch.asinh(pz / torch.where(has_mt, mt, 1.0)), 0.0)
    jet_px = px.sum(dim=-1, keepdim=True)
    jet_py = py.sum(dim=-1, keepdim=True)
    phi = torch.atan2(jet_px * py - jet_py * px, jet_px * px + jet_py * py)
    return pt, y, phi, real


def _half_plane_sign(z: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    """-1 where the negative half-plane carries more pT than the positive one.

    energyflow's ``_do_reflection``: sum(z[coord > 0]) < sum(z[coord < 0]).
    """
    pos = torch.where(coord > 0, z, 0.0).sum(dim=-1, keepdim=True)
    neg = torch.where(coord < 0, z, 0.0).sum(dim=-1, keepdim=True)
    return torch.where(pos < neg, -1.0, 1.0).to(coord.dtype)


def preprocess_constituents(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Momentum fractions and centred, rotated, reflected (y, phi) coordinates.

    Parameters
    ----------
    x : (B, K, 5) harness tensor (E, px, py, pz, mask), any global scale.

    Returns
    -------
    z      : (B, K) pT_i / sum_j pT_j over the real constituents (0 on padding).
    coords : (B, K, 2) per-particle (v, u): the coordinates along the minor
             and the principal axis of the jet's pT-weighted (y, phi)
             distribution, centred on its pT-weighted centroid, each axis
             oriented so that its positive half carries at least as much pT
             as its negative half ((0, 0) on padding).
    real   : (B, K) bool, the mask.

    Computed in the dtype of ``x``; see the module docstring for the
    correspondence with energyflow's center / rotate / reflect.
    """
    pt, y, phi, real = constituent_kinematics(x)
    pt_sum = pt.sum(dim=-1, keepdim=True)
    has_pt = pt_sum > 0
    z = torch.where(has_pt, pt / torch.where(has_pt, pt_sum, 1.0), 0.0)

    # 1. centre on the pT-weighted centroid (Delta phi wrapped into (-pi, pi]).
    dy = y - (z * y).sum(dim=-1, keepdim=True)
    dphi = wrap_phi(phi - (z * phi).sum(dim=-1, keepdim=True))

    # 2. rotate into the eigenbasis of M; the principal axis (larger
    #    eigenvalue) maximises (a + c)/2 + (a - c)/2 cos 2t + b sin 2t.
    m_yy = (z * dy * dy).sum(dim=-1, keepdim=True)
    m_pp = (z * dphi * dphi).sum(dim=-1, keepdim=True)
    m_yp = (z * dy * dphi).sum(dim=-1, keepdim=True)
    t = 0.5 * torch.atan2(2.0 * m_yp, m_yy - m_pp)
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    u = cos_t * dy + sin_t * dphi          # principal axis -> "phi" slot
    v = cos_t * dphi - sin_t * dy          # minor axis     -> "y" slot

    # 3. reflect: more pT in the positive half-plane of each axis.
    u = u * _half_plane_sign(z, u)
    v = v * _half_plane_sign(z, v)

    coords = torch.stack([v, u], dim=-1)
    coords = torch.where(real.unsqueeze(-1), coords, 0.0)
    return z, coords, real


# ─────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────

def _relu_stack(in_dim: int, sizes: Sequence[int], dtype: torch.dtype) -> nn.Sequential:
    """Linear + ReLU per size, He-uniform weights and zero biases (Keras
    ``he_uniform``: U(-sqrt(6 / fan_in), sqrt(6 / fan_in)))."""
    layers: list[nn.Module] = []
    for size in sizes:
        lin = nn.Linear(in_dim, size, dtype=dtype)
        nn.init.kaiming_uniform_(lin.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
        in_dim = size
    return nn.Sequential(*layers)


class _SymmetricPerParticleNN(nn.Module):
    """Phi per particle -> weighted sum over particles -> F -> logits.

    The common skeleton of energyflow's ``SymmetricPerParticleNN``. Subclasses
    choose the per-particle input of Phi and the weight of each particle in
    the sum.
    """

    phi_in: int = 0      # per-particle input width of Phi

    def __init__(
        self,
        out_features: int,
        *,
        Phi_sizes: Sequence[int],
        F_sizes: Sequence[int],
        dtype: torch.dtype,
        eval_chunk_size: int | None,
    ) -> None:
        super().__init__()
        self.Phi_sizes = tuple(int(s) for s in Phi_sizes)
        self.F_sizes = tuple(int(s) for s in F_sizes)
        if not self.Phi_sizes:
            raise ValueError("Phi_sizes must name at least the latent dimension")
        self.dtype = dtype
        self.eval_chunk_size = eval_chunk_size
        self.Phi = _relu_stack(self.phi_in, self.Phi_sizes, dtype)
        self.F = _relu_stack(self.Phi_sizes[-1], self.F_sizes, dtype)
        f_out = self.F_sizes[-1] if self.F_sizes else self.Phi_sizes[-1]
        self.output = nn.Linear(f_out, out_features, dtype=dtype)
        nn.init.xavier_uniform_(self.output.weight)      # Keras glorot_uniform
        nn.init.zeros_(self.output.bias)

    def particle_inputs(self, z: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def particle_weights(self, z: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """(B, l) latent observables sum_i w_i Phi_i of a (B, K, 5) batch."""
        z, coords, real = preprocess_constituents(x.to(self.dtype))
        h = self.Phi(self.particle_inputs(z, coords))             # (B, K, l)
        w = self.particle_weights(z, real)                        # (B, K), 0 on padding
        return (w.unsqueeze(-1) * h).sum(dim=1)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.F(self.latent(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B, out_features) of a (B, K, 5) batch (E, px, py, pz, mask)."""
        x = x.to(self.dtype)
        chunk = self.eval_chunk_size
        if chunk and not torch.is_grad_enabled() and x.shape[0] > chunk:
            return torch.cat([self._logits(xc) for xc in x.split(chunk)], dim=0)
        return self._logits(x)

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype)


class ParticleFlowNetwork(_SymmetricPerParticleNN):
    """PFN: logits = out(F(sum_i mask_i Phi(z_i, v_i, u_i))).

    Per-particle input (z, v, u) from ``preprocess_constituents``: the pT
    fraction and the centred / rotated / reflected (y, phi) coordinates
    (energyflow ``PFN`` with input_dim = 3). Defaults Phi = (24, 24, 64),
    F = (24, 24, 24): 5,106 trainable parameters with ``out_features=2``
    (the paper's Phi = (100, 100, 256), F = (100, 100, 100): 82,458).
    Padding rows (mask 0) have weight 0. Not IRC safe.
    """

    phi_in = 3

    def __init__(
        self,
        out_features: int = 2,
        *,
        dtype: torch.dtype = torch.float64,
        Phi_sizes: Sequence[int] = PHI_SIZES,
        F_sizes: Sequence[int] = F_SIZES,
        eval_chunk_size: int | None = EVAL_CHUNK_JETS,
    ) -> None:
        super().__init__(out_features, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                         dtype=dtype, eval_chunk_size=eval_chunk_size)

    def particle_inputs(self, z: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.unsqueeze(-1), coords], dim=-1)

    def particle_weights(self, z: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return real.to(z.dtype)


class EnergyFlowNetwork(_SymmetricPerParticleNN):
    """EFN: logits = out(F(sum_i z_i Phi(v_i, u_i))).

    Phi sees only the centred / rotated / reflected (y, phi) coordinates and
    each particle enters with weight z_i = pT_i / sum_j pT_j (energyflow
    ``EFN`` with input_dim = 2), which makes the latent observables, and the
    whole network, IRC safe. Defaults Phi = (24, 24, 64), F = (24, 24, 24):
    5,082 trainable parameters with ``out_features=2`` (the paper's sizes:
    82,358). Padding rows have z = 0.
    """

    phi_in = 2

    def __init__(
        self,
        out_features: int = 2,
        *,
        dtype: torch.dtype = torch.float64,
        Phi_sizes: Sequence[int] = PHI_SIZES,
        F_sizes: Sequence[int] = F_SIZES,
        eval_chunk_size: int | None = EVAL_CHUNK_JETS,
    ) -> None:
        super().__init__(out_features, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                         dtype=dtype, eval_chunk_size=eval_chunk_size)

    def particle_inputs(self, z: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return coords

    def particle_weights(self, z: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return z
