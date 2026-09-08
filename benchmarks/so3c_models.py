"""
benchmarks.so3c_models
----------------------
Classifier heads for the SO(3, C) (complexified-SO(3)) benchmark battery.

Three roles, mirroring the Arch-A / control / Arch-B split of the parent
so33 benchmarks:

- SO3CInvariantsClassifier : reads BOTH real invariants (Re z.z, Im z.z) —
  SO(3, C)-invariant by construction. The complexified analogue of
  EtaInvariantsClassifier.
- EtaOnlyClassifier        : reads only Re(z.z) = v^T eta v — what an
  eta-based (so33-style) invariant readout sees on complexified data.
  Structurally blind to labels carried by Im(z.z).
- SO3CFlowClassifier       : equivariant feature extractor — complex channel
  lift, per-channel SO3CActivation geodesic flow, cross-channel invariant
  readout. Exactly invariant end-to-end; the flow is load-bearing because
  cross-channel invariants z_c(T) . z_d(T) are NOT conserved (only each
  channel's own z_c . z_c is), so the readout is strictly richer than the
  input invariants. This realises the "expand channels, not the metric"
  capacity direction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from so3c.activation import SO3CActivation
from so3c.algebra import expm_so3c, invariant_features, real_to_complex
from so3c.interaction import SO3CInteraction
from so3c.lift import jet_bivectors, minkowski_inner


# ─────────────────────────────────────────────────────────────────────────
# Shared pooled-invariant readouts for set (per-constituent) models.
# Design informed by the so33 Arch-B failure (readout omitted the pairwise
# term -> chance level): every readout here includes pairwise statistics.
# ─────────────────────────────────────────────────────────────────────────

# Feature counts, so callers size their first Linear correctly.
N_POOLED_FULL = 11
N_POOLED_SIMPLE = 5


def _pooled_bivector_invariants(
    z: torch.Tensor,
    mask: torch.Tensor,
    simple: bool = False,
) -> torch.Tensor:
    """(B, K, 3) complex states + (B, K) mask -> (B, 11) invariant features,
    or (B, 5) when ``simple=True``.

    All features are SO(3, C)-invariant and permutation-invariant: masked
    moments of the per-particle invariants z_a . z_a, of the pairwise
    invariants z_a . z_b, and the jet-total invariant. arcsinh-compressed.

    ``simple=True`` is for RAW jet bivectors, i.e. before any flow. Six of the
    eleven features are then identically zero, by algebra rather than by
    accident, and feeding them to a Linear only wastes weights:

      * every Im feature vanishes. A simple bivector p ^ q has E.B = 0, so
        Im(z_a . z_a) = 0; and two bivectors sharing the leg P give
        Im(z_a . z_b) ~ eps(p_a, P, p_b, P) = 0.
      * the jet-total invariant vanishes. bivector_lift is linear in its
        first argument and jet_bivectors pairs every constituent with the
        same P = sum_a p_a, so
            z_tot = sum_a bivec(p_a, P) = bivec(P, P) = 0
        exactly, hence q_tot = 0 in both real and imaginary parts.

    Measured on 2000 real jets: the six dropped features have magnitude
    1e-13 or smaller (q_tot below 1e-26), while the five kept ones are O(1).
    After a flow the states are no longer simple and all eleven carry
    signal — that is precisely the flow's contribution to the readout — so
    the flow models keep ``simple=False``.
    """
    B, K, _ = z.shape
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)          # (B, K, K)
    n = mask.sum(dim=-1).clamp_min(1.0)                          # (B,)
    eye = torch.eye(K, dtype=mask.dtype, device=mask.device)
    off_mask = pair_mask * (1.0 - eye)
    n_off = off_mask.sum(dim=(-2, -1)).clamp_min(1.0)

    S = z @ z.transpose(-1, -2)                                  # (B, K, K)
    s_diag = torch.diagonal(S, dim1=-2, dim2=-1)                 # (B, K)
    s_diag = s_diag * mask
    S_off = S * off_mask

    z_tot = (z * mask.unsqueeze(-1)).sum(dim=1)                  # (B, 3)
    q_tot = (z_tot * z_tot).sum(dim=-1)                          # (B,) complex

    live = [
        s_diag.real.sum(-1) / n,
        s_diag.real.abs().sum(-1) / n,
        S_off.real.sum((-2, -1)) / n_off,
        S_off.real.pow(2).sum((-2, -1)) / n_off,
        S_off.real.abs().amax(dim=(-2, -1)),
    ]
    if simple:
        return torch.asinh(torch.stack(live, dim=-1))            # (B, 5)

    feats = live[:2] + [s_diag.imag.sum(-1) / n] + live[2:] + [
        S_off.imag.sum((-2, -1)) / n_off,
        S_off.imag.pow(2).sum((-2, -1)) / n_off,
        S_off.imag.abs().amax(dim=(-2, -1)),
        q_tot.real,
        q_tot.imag,
    ]
    return torch.asinh(torch.stack(feats, dim=-1))               # (B, 11)


def _minkowski_stats(p4: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, K, 4) + (B, K) -> (B, 7): the eta_invariants feature set computed
    directly in Minkowski form (m^2 = <p,p>, pairwise <p_a,p_b>, jet <P,P>).
    arcsinh-compressed."""
    p4 = p4 * mask.unsqueeze(-1)
    m2 = minkowski_inner(p4, p4) * mask                          # (B, K)
    n = mask.sum(dim=-1).clamp_min(1.0)

    # Pairwise <p_a, p_b> via the metric split: E_a E_b - p_a . p_b.
    E = p4[..., 0]
    pv = p4[..., 1:]
    s = E.unsqueeze(-1) * E.unsqueeze(-2) - pv @ pv.transpose(-1, -2)
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    s_off = s * pair_mask * (1.0 - eye)
    n_off = (pair_mask * (1.0 - eye)).sum(dim=(-2, -1)).clamp_min(1.0)

    P = p4.sum(dim=1)                                            # (B, 4)
    feats = torch.stack([
        m2.sum(-1) / n,
        m2.pow(2).sum(-1) / n,
        m2.abs().sum(-1) / n,
        s_off.sum((-2, -1)) / n_off,
        s_off.pow(2).sum((-2, -1)) / n_off,
        s_off.abs().amax(dim=(-2, -1)),
        minkowski_inner(P, P),
    ], dim=-1)                                                   # (B, 7)
    return torch.asinh(feats)


class SO3CBottleneck(nn.Module):
    """Linear(in -> 6) -> SO3CActivation -> Linear(6 -> out).

    The direct so3c analogue of SO33Network's matched-bottleneck wiring for
    flat tabular data (HIGGS, Adult). Uses the closed-form "exact" flow — no
    ODE solver, so unlike the so33 counterpart it cannot diverge on
    heavy-tailed real-data inputs (the connection is soft-normalised and the
    flow is a bounded group element), and needs neither input bounding nor a
    norm cap on the flat path.

    mode="dynamic": connection from the invariant-fed HermitianMetric MLP.
    mode="static" : 6 learnable scalars — the closest analogue of so33's
                    15-coefficient activation (matched parameter class).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "dynamic",
        T: float = 1.0,
        hidden_metric: int = 16,
        bound_input: str = "none",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.l1 = nn.Linear(in_features, 6).to(dtype)
        self.act = SO3CActivation(
            T=T, mode=mode, method="exact",
            bound_input=bound_input, hidden=hidden_metric, dtype=dtype,
        )
        self.l2 = nn.Linear(6, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        return self.l2(self.act(self.l1(x)))

    def regularization_loss(self) -> torch.Tensor:
        return self.act.regularization_loss()


class SO3CInvariantSetClassifier(nn.Module):
    """Arch-A analogue on the bivector lift: invariant by construction.

    (B, K, 5) constituents -> jet bivectors z_a = bivec(p_a, P) -> pooled
    complex invariants (11) + Minkowski eta-stats (7) -> MLP. No flow: this
    isolates whether the bivector pair invariants beat the eta feature set.
    """

    def __init__(
        self,
        out_features: int = 2,
        hidden: int = 64,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        # 5 live bivector invariants (the other 6 are identically zero for
        # raw bivectors — see _pooled_bivector_invariants) + 7 Minkowski.
        self.mlp = nn.Sequential(
            nn.Linear(N_POOLED_SIMPLE + 7, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4, mask = x[..., :4], x[..., 4]
        z = jet_bivectors(p4, mask)
        return torch.cat(
            [_pooled_bivector_invariants(z, mask, simple=True),
             _minkowski_stats(p4, mask)],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._features(x))

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=next(self.parameters()).device)


class SO3CEquivariantSetClassifier(nn.Module):
    """Arch B done right: bivector lift -> channel lift -> shared geodesic
    flow -> pooled invariant readout WITH pairwise and cross-channel terms.

    Every step is exactly equivariant / invariant: complex channel scalars
    commute with the group action, the flow's connection is built from
    invariants, and the readout consumes only complex bilinear invariants.
    The flow is load-bearing: it de-simplifies the bivectors, populating
    Im features and cross-channel invariants that are trivial at the input.
    """

    def __init__(
        self,
        out_features: int = 2,
        channels: int = 4,
        hidden: int = 64,
        act_hidden: int = 16,
        T: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.channels = channels

        gen = torch.Generator().manual_seed(0)
        w = torch.stack([
            1.0 + 0.1 * torch.randn(channels, dtype=dtype, generator=gen),
            0.1 * torch.randn(channels, dtype=dtype, generator=gen),
        ], dim=-1)
        self.channel_weights = nn.Parameter(w)                   # (C, 2)

        self.act = SO3CActivation(
            T=T, mode="dynamic", method="exact",
            bound_input="bilinear", hidden=act_hidden, dtype=dtype,
        )

        n_ch_pairs = channels * (channels + 1) // 2
        in_features = channels * 11 + 2 * n_ch_pairs + 7
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

        iu = torch.triu_indices(channels, channels)
        self.register_buffer("pair_rows", iu[0])
        self.register_buffer("pair_cols", iu[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4, mask = x[..., :4], x[..., 4]
        z = jet_bivectors(p4, mask)                              # (B, K, 3)
        B, K, _ = z.shape
        C = self.channels

        w = torch.complex(self.channel_weights[:, 0],
                          self.channel_weights[:, 1])            # (C,)
        zc = w[None, :, None, None] * z[:, None, :, :]           # (B, C, K, 3)
        zc = self.act(zc.reshape(B * C * K, 3)).reshape(B, C, K, 3)

        # Per-channel pooled invariants over the particle set.
        per_ch = torch.stack(
            [_pooled_bivector_invariants(zc[:, c], mask) for c in range(C)],
            dim=1,
        ).reshape(B, C * 11)

        # Cross-channel invariants of the jet-total states.
        z_tot = (zc * mask[:, None, :, None]).sum(dim=2)         # (B, C, 3)
        T_cc = z_tot @ z_tot.transpose(-1, -2)                   # (B, C, C)
        T_pairs = T_cc[:, self.pair_rows, self.pair_cols]
        cross = torch.cat(
            [torch.asinh(T_pairs.real), torch.asinh(T_pairs.imag)], dim=-1
        )

        feats = torch.cat([per_ch, cross, _minkowski_stats(p4, mask)], dim=-1)
        return self.mlp(feats)

    def regularization_loss(self) -> torch.Tensor:
        return self.act.regularization_loss()


class SO3CInteractionSetClassifier(nn.Module):
    """Bivector lift -> SO3CInteraction (equivariant multi-particle geodesic
    flow, adaptive solver) -> pooled invariant readout. The only ODE model
    in the so3c set family — the direct precursor of the LGEB integration.
    """

    def __init__(
        self,
        out_features: int = 2,
        hidden: int = 64,
        interaction_hidden: int = 16,
        T: float = 1.0,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.interaction = SO3CInteraction(
            hidden=interaction_hidden, T=T, rtol=rtol, atol=atol, dtype=dtype,
        )
        self.mlp = nn.Sequential(
            nn.Linear(18, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4, mask = x[..., :4], x[..., 4]
        z = jet_bivectors(p4, mask)                              # (B, K, 3) complex
        # Soft invariant bound before the ODE: bivectors are quadratic in p
        # and the jet-total leg makes them O(K); rescale by the invariant
        # magnitude so the flow operates on O(1) states (equivariant).
        q = (z * z).sum(dim=-1)
        scale = 1.0 + (q.real.pow(2) + q.imag.pow(2) + 1e-12).sqrt().sqrt()
        z = z / scale.unsqueeze(-1)
        z_out = self.interaction(z, mask=mask)
        feats = torch.cat(
            [_pooled_bivector_invariants(z_out, mask), _minkowski_stats(p4, mask)],
            dim=-1,
        )
        return self.mlp(feats)

    def regularization_loss(self) -> torch.Tensor:
        return self.interaction.regularization_loss()


class SO3CCovariantSetClassifier(nn.Module):
    """Equivariant flow whose connection is built COVARIANTLY.

    Why the previous design failed
    ------------------------------
    SO3CEquivariantSetClassifier feeds SO3CActivation, whose connection
    a(s) is a function of invariants. Under z -> Qz the invariants do not
    move, so a does not move -- but equivariance needs a -> Qa. The flow
    then applies the same rotation in the new frame instead of the
    conjugated one. Per-particle invariants survive (an antisymmetric
    connection conserves z.z) but the pairwise z_a.z_b terms the readout
    reads do not, and a trained model loses 0.095 AUC under a rapidity-2
    boost.

    Why a single particle cannot be fixed
    -------------------------------------
    The only covariant vector available from one state z is z itself (times
    any function of invariants), and [z]_x z = z x z = 0. Rotating about
    your own axis is the identity, so a genuinely equivariant flow MUST be
    multi-particle. There is no single-particle repair.

    The construction
    ----------------
    The cross product is covariant for SO(3, C): Q(z_a x z_b) =
    (Q z_a) x (Q z_b), since det Q = 1. Weighting it by invariants keeps
    that, so

        a_a = z_a  x  sum_b phi(s_aa, s_bb, s_ab) z_b

    transforms as a_a -> Q a_a, and therefore

        exp(-T [Q a]_x) (Q z) = Q exp(-T [a]_x) z

    exactly. Note the reference vector must be an invariant-WEIGHTED sum:
    the plain total sum_b z_b vanishes identically for this lift, because
    every z_b shares the leg P and sum_b bivec(p_b, P) = bivec(P, P) = 0.

    The connection is evaluated once at t = 0 and held fixed, so the flow
    is a single closed-form group element (complex Rodrigues) rather than
    an ODE solve -- equivariant, exact, and solver-free.

    Channels differ in a way that matters here: each carries its own phi,
    so each produces a different rotation. In the broken model channels
    were complex scalar rescalings z_c = w_c z, which is a large part of
    why adding them bought nothing (AUC flat to 0.0002 from 9k to 198k
    parameters).
    """

    def __init__(
        self,
        out_features: int = 2,
        channels: int = 4,
        hidden: int = 64,
        act_hidden: int = 16,
        T: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.channels = channels
        self.T = T
        # phi: pairwise invariants -> one real coefficient per channel.
        self.phi = nn.Sequential(
            nn.Linear(6, act_hidden), nn.Tanh(),
            nn.Linear(act_hidden, channels),
        ).to(dtype)
        nn.init.zeros_(self.phi[-1].weight)   # identity flow at init
        nn.init.zeros_(self.phi[-1].bias)

        n_ch_pairs = channels * (channels + 1) // 2
        in_features = channels * 11 + 2 * n_ch_pairs + 7
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)
        iu = torch.triu_indices(channels, channels)
        self.register_buffer("pair_rows", iu[0])
        self.register_buffer("pair_cols", iu[1])

    def _connection(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, K, 3) complex -> (B, C, K, 3) complex covariant connections."""
        S = z @ z.transpose(-1, -2)                          # (B, K, K)
        s_diag = torch.diagonal(S, dim1=-2, dim2=-1)         # (B, K)
        B, K = s_diag.shape
        feats = torch.stack([
            s_diag.real.unsqueeze(-1).expand(B, K, K),
            s_diag.imag.unsqueeze(-1).expand(B, K, K),
            s_diag.real.unsqueeze(-2).expand(B, K, K),
            s_diag.imag.unsqueeze(-2).expand(B, K, K),
            S.real, S.imag,
        ], dim=-1)
        phi = self.phi(torch.asinh(feats))                   # (B, K, K, C)
        pair_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(-1)
        phi = phi * pair_mask
        n = mask.sum(-1).clamp_min(1.0)[:, None, None, None]

        # Reference vector per (channel, particle): an invariant-weighted
        # sum of the other states. Covariant because the weights are
        # invariants and the sum is over a covariant object.
        ref = torch.einsum("bklc,bld->bckd", phi.to(z.real.dtype), z.real)             + 1j * torch.einsum("bklc,bld->bckd", phi.to(z.real.dtype), z.imag)
        ref = ref / n
        zc = z.unsqueeze(1).expand(-1, self.channels, -1, -1)   # (B, C, K, 3)
        return torch.linalg.cross(zc, ref, dim=-1)              # covariant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4, mask = x[..., :4], x[..., 4]
        z = jet_bivectors(p4, mask)                          # (B, K, 3)
        a = self._connection(z, mask)                        # (B, C, K, 3)

        B, C, K, _ = a.shape
        Q = expm_so3c(a.reshape(-1, 3), t=-self.T)           # (B*C*K, 3, 3)
        zc = z.unsqueeze(1).expand(-1, C, -1, -1).reshape(-1, 3, 1)
        zc = (Q @ zc).squeeze(-1).reshape(B, C, K, 3)

        per_ch = torch.stack(
            [_pooled_bivector_invariants(zc[:, c], mask) for c in range(C)],
            dim=1).reshape(B, C * 11)
        z_tot = (zc * mask[:, None, :, None]).sum(dim=2)     # (B, C, 3)
        T_cc = z_tot @ z_tot.transpose(-1, -2)
        T_pairs = T_cc[:, self.pair_rows, self.pair_cols]
        cross = torch.cat([torch.asinh(T_pairs.real),
                           torch.asinh(T_pairs.imag)], dim=-1)
        return self.mlp(torch.cat(
            [per_ch, cross, _minkowski_stats(p4, mask)], dim=-1))

    def regularization_loss(self) -> torch.Tensor:
        last = self.phi[-1]
        return 1e-3 * (last.weight.pow(2).sum() + last.bias.pow(2).sum())


class SO3CMessageSetClassifier(nn.Module):
    """Covariant message passing: several rounds of the SO3CCovariantSet
    connection, with a scalar channel running alongside the vector one.

    The single-round covariant model has one bottleneck the readout cannot
    work around: every bit of information about a constituent travels
    through three complex numbers. Real Lorentz-equivariant taggers
    (LorentzNet's LGEB, PELICAN) carry a *scalar* embedding beside the
    vector and update both, several times. This is that construction on the
    so3c algebra.

    Per round, for particles a, b and channel c:

        e_ab   = MLP([Re/Im z_a.z_b, Re/Im z_a.z_a, Re/Im z_b.z_b, h_a, h_b])
        h_a   <- h_a + MLP([h_a, mean_b msg(e_ab)])            (invariant)
        ref_a  = mean_b w_c(e_ab) z_b                          (covariant)
        z_a   <- exp(-T [z_a x ref_a]_x) z_a                   (covariant)
        z     <- M z                                           (covariant)

    Equivariance holds round by round and therefore end to end:

    * every edge feature is a bilinear invariant z.z or a scalar h, so the
      weights w_c and messages are invariant -- they do not move under
      z -> Qz;
    * an invariant-weighted sum of covariant vectors is covariant;
    * the cross product is covariant for SO(3, C), since det Q = 1;
    * exp(-T [Qa]_x) (Qz) = Q exp(-T [a]_x) z, because [Qa]_x = Q [a]_x Q^T;
    * a constant complex channel mixing M commutes with Q, which is complex
      linear.

    The readout is the covariant model's, plus masked mean/max of the final
    scalar states.

    Bounding. exp of a complex generator is a boost as well as a rotation,
    so the Hermitian norm |z| can grow without bound over rounds. It cannot
    be normalised away: the only scalars available are the bilinear
    invariants, and a null vector has z.z = 0 at arbitrary magnitude. What
    IS available is an invariant soft bound -- divide by 1 + |z.z|^(1/2),
    a function of invariants times a covariant vector -- applied to the
    input lift and to each round's connection. Combined with the zero-init
    weight head (identity flow at step 0) that has kept the rounds stable.

    Sparsification. ``neighbors=k`` restricts messages to each particle's k
    strongest partners, cutting the per-round cost from K^2 to K*k. The
    ranking key is |Re(z_a . z_b)| on the input bivectors -- an INVARIANT,
    so every jet keeps the same neighbour set in every frame and the graph
    itself is equivariant. A key built from a frame-dependent quantity
    (angular distance in the lab, pT ordering) would silently reshuffle
    edges under a boost and break the symmetry with nothing to flag it. The
    graph is built once from the input and reused across rounds; the states
    rotate, but their pairwise invariants -- and hence the ranking -- are
    only changed by the flow, not by the frame.
    """

    def __init__(
        self,
        out_features: int = 2,
        channels: int = 4,
        rounds: int = 3,
        hidden: int = 64,
        act_hidden: int = 16,
        scalar_dim: int = 8,
        msg_dim: int = 8,
        T: float = 1.0,
        channel_mixing: bool = True,
        neighbors: int | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.channels = channels
        self.rounds = rounds
        self.scalar_dim = scalar_dim
        self.neighbors = neighbors
        self.T = T

        gen = torch.Generator().manual_seed(0)
        w = torch.stack([
            1.0 + 0.1 * torch.randn(channels, dtype=dtype, generator=gen),
            0.1 * torch.randn(channels, dtype=dtype, generator=gen),
        ], dim=-1)
        self.channel_weights = nn.Parameter(w)                   # (C, 2)

        # Scalar states are seeded from per-particle invariants only:
        # Re/Im (z.z) and the Minkowski mass <p, p>.
        self.h_init = nn.Linear(3, scalar_dim).to(dtype)

        edge_in = 6 * channels + 2 * scalar_dim
        self.edge = nn.ModuleList()
        self.w_head = nn.ModuleList()
        self.msg_head = nn.ModuleList()
        self.node = nn.ModuleList()
        for _ in range(rounds):
            self.edge.append(nn.Sequential(
                nn.Linear(edge_in, act_hidden), nn.Tanh()).to(dtype))
            head = nn.Linear(act_hidden, channels).to(dtype)
            nn.init.zeros_(head.weight)      # identity flow at init
            nn.init.zeros_(head.bias)
            self.w_head.append(head)
            self.msg_head.append(nn.Linear(act_hidden, msg_dim).to(dtype))
            self.node.append(nn.Sequential(
                nn.Linear(scalar_dim + msg_dim, act_hidden), nn.ReLU(),
                nn.Linear(act_hidden, scalar_dim)).to(dtype))

        if channel_mixing:
            mix = torch.zeros(rounds, channels, channels, 2, dtype=dtype)
            mix[..., 0] = torch.eye(channels, dtype=dtype)       # identity init
            self.mix = nn.Parameter(mix)
        else:
            self.register_parameter("mix", None)

        n_ch_pairs = channels * (channels + 1) // 2
        in_features = channels * 11 + 2 * n_ch_pairs + 2 * scalar_dim + 7
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)
        iu = torch.triu_indices(channels, channels)
        self.register_buffer("pair_rows", iu[0])
        self.register_buffer("pair_cols", iu[1])

    @staticmethod
    def _invariant_scale(v: torch.Tensor) -> torch.Tensor:
        """1 + |v.v|^(1/2), an invariant soft bound. (..., 3) -> (...)."""
        q = (v * v).sum(dim=-1)
        return 1.0 + (q.real.pow(2) + q.imag.pow(2) + 1e-12).sqrt().sqrt()

    def _neighbour_graph(self, z, mask):
        """(B, K, 3), (B, K) -> neighbour indices (B, K, k) and their mask.

        Ranked by the invariant |Re(z_a . z_b)|, self-edges excluded.
        """
        k = self.neighbors
        S = (z @ z.transpose(-1, -2)).real.abs()                 # (B, K, K)
        pair = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        eye = torch.eye(z.shape[1], dtype=pair.dtype, device=pair.device)
        pair = pair * (1.0 - eye)
        key = torch.where(pair.bool(), S, torch.full_like(S, -1.0))
        val, idx = key.topk(k, dim=-1)                            # (B, K, k)
        return idx, (val >= 0).to(z.real.dtype) * mask.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4, mask = x[..., :4], x[..., 4]
        z = jet_bivectors(p4, mask)                              # (B, K, 3)
        z = z / self._invariant_scale(z).unsqueeze(-1)

        q = (z * z).sum(dim=-1)
        m2 = minkowski_inner(p4, p4) * mask
        h = self.h_init(torch.asinh(
            torch.stack([q.real, q.imag, m2], dim=-1)))          # (B, K, D)
        h = h * mask.unsqueeze(-1)

        w = torch.complex(self.channel_weights[:, 0],
                          self.channel_weights[:, 1])            # (C,)
        zc = w[None, :, None, None] * z[:, None, :, :]           # (B, C, K, 3)

        B, C, K, _ = zc.shape
        D = self.scalar_dim
        node_mask = mask.unsqueeze(-1)
        sparse = self.neighbors is not None and self.neighbors < K
        if sparse:
            idx, edge_mask = self._neighbour_graph(z, mask)      # (B, K, k)
            bat = torch.arange(B, device=z.device).view(B, 1, 1)
            k = idx.shape[-1]
            denom = edge_mask.sum(-1).clamp_min(1.0)             # (B, K)
        else:
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, K, K)
            denom = mask.sum(dim=-1).clamp_min(1.0).unsqueeze(-1).expand(B, K)

        for r in range(self.rounds):
            if sparse:
                zp = zc.permute(0, 2, 1, 3)                      # (B, K, C, 3)
                zb = zp[bat, idx]                                # (B, K, k, C, 3)
                sd = (zp * zp).sum(dim=-1)                       # (B, K, C)
                s_ab = (zp.unsqueeze(2) * zb).sum(dim=-1)        # (B, K, k, C)
                row = sd.unsqueeze(2).expand(B, K, k, C)
                col = sd[bat, idx]                               # (B, K, k, C)
                hb = h[bat, idx]                                 # (B, K, k, D)
                ha = h.unsqueeze(2).expand(B, K, k, D)
                emask = edge_mask.unsqueeze(-1)
            else:
                S = zc @ zc.transpose(-1, -2)                    # (B, C, K, K)
                sd = torch.diagonal(S, dim1=-2, dim2=-1).permute(0, 2, 1)
                s_ab = S.permute(0, 2, 3, 1)                     # (B, K, K, C)
                row = sd.unsqueeze(2).expand(B, K, K, C)
                col = sd.unsqueeze(1).expand(B, K, K, C)
                hb = h.unsqueeze(1).expand(B, K, K, D)
                ha = h.unsqueeze(2).expand(B, K, K, D)
                emask = pair_mask.unsqueeze(-1)

            e = torch.cat([
                torch.asinh(s_ab.real), torch.asinh(s_ab.imag),
                torch.asinh(row.real), torch.asinh(row.imag),
                torch.asinh(col.real), torch.asinh(col.imag),
                ha, hb,
            ], dim=-1)                                           # (B,K,*,6C+2D)

            hid = self.edge[r](e)
            wts = self.w_head[r](hid) * emask                    # (B, K, *, C)
            msg = self.msg_head[r](hid) * emask                  # (B, K, *, M)

            h = h + self.node[r](torch.cat(
                [h, msg.sum(dim=2) / denom.unsqueeze(-1)], dim=-1))
            h = h * node_mask

            if sparse:
                ref = torch.complex(
                    torch.einsum("bakc,bakcx->bacx", wts, zb.real),
                    torch.einsum("bakc,bakcx->bacx", wts, zb.imag),
                ).permute(0, 2, 1, 3)                            # (B, C, K, 3)
            else:
                wc = wts.permute(0, 3, 1, 2)                     # (B, C, K, K)
                ref = torch.complex(wc @ zc.real, wc @ zc.imag)
            ref = ref / denom[:, None, :, None]
            a = torch.linalg.cross(zc, ref, dim=-1)              # covariant
            a = a / self._invariant_scale(a).unsqueeze(-1)

            Q = expm_so3c(a.reshape(-1, 3), t=-self.T)           # (B*C*K, 3, 3)
            zc = (Q @ zc.reshape(-1, 3, 1)).squeeze(-1).reshape(B, C, K, 3)
            zc = zc * mask[:, None, :, None]

            if self.mix is not None:
                M = torch.complex(self.mix[r, ..., 0], self.mix[r, ..., 1])
                zc = torch.einsum("cd,bdkx->bckx", M, zc)

        per_ch = torch.stack(
            [_pooled_bivector_invariants(zc[:, c], mask) for c in range(C)],
            dim=1).reshape(B, C * 11)
        z_tot = (zc * mask[:, None, :, None]).sum(dim=2)         # (B, C, 3)
        T_cc = z_tot @ z_tot.transpose(-1, -2)
        T_pairs = T_cc[:, self.pair_rows, self.pair_cols]
        cross = torch.cat([torch.asinh(T_pairs.real),
                           torch.asinh(T_pairs.imag)], dim=-1)

        n = mask.sum(dim=-1).clamp_min(1.0)
        h_mean = (h * node_mask).sum(dim=1) / n.unsqueeze(-1)
        h_max = torch.where(node_mask.bool(), h,
                            torch.full_like(h, -1e30)).amax(dim=1)
        return self.mlp(torch.cat(
            [per_ch, cross, h_mean, h_max, _minkowski_stats(p4, mask)],
            dim=-1))

    def regularization_loss(self) -> torch.Tensor:
        loss = torch.zeros((), dtype=self.dtype,
                           device=self.channel_weights.device)
        for head in self.w_head:
            loss = loss + head.weight.pow(2).sum() + head.bias.pow(2).sum()
        return 1e-3 * loss


class MultiChannelSO3C(nn.Module):
    """Multi-channel so3c block: C parallel Linear(in -> 6) + SO3CActivation,
    concatenated to 6*C, then Linear(6*C -> out). Mirror of MultiChannelSO33.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        channels: int = 4,
        mode: str = "dynamic",
        T: float = 1.0,
        hidden_metric: int = 16,
        bound_input: str = "none",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.lifts = nn.ModuleList(
            [nn.Linear(in_features, 6).to(dtype) for _ in range(channels)]
        )
        self.acts = nn.ModuleList(
            [
                SO3CActivation(
                    T=T, mode=mode, method="exact",
                    bound_input=bound_input, hidden=hidden_metric, dtype=dtype,
                )
                for _ in range(channels)
            ]
        )
        self.head = nn.Linear(6 * channels, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        feats = [act(lift(x)) for lift, act in zip(self.lifts, self.acts)]
        return self.head(torch.cat(feats, dim=-1))

    def regularization_loss(self) -> torch.Tensor:
        return sum(act.regularization_loss() for act in self.acts)


class SO3CInvariantsClassifier(nn.Module):
    """SO(3, C)-INVARIANT-by-construction classifier (both invariants)."""

    def __init__(
        self,
        out_features: int = 2,
        hidden: int = 32,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(invariant_features(x.to(self.dtype)))

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=next(self.parameters()).device)


class EtaOnlyClassifier(nn.Module):
    """Control: sees only Re(z.z) — the eta-invariant of the parent so33.

    On labels carried by Im(z.z) this model is structurally at chance; it
    quantifies exactly what the complexification adds over the SO(3,3) prior.
    """

    def __init__(
        self,
        out_features: int = 2,
        hidden: int = 32,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = invariant_features(x.to(self.dtype))[..., :1]   # Re part only
        return self.mlp(feats)

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=next(self.parameters()).device)


class SO3CFlowClassifier(nn.Module):
    """Equivariant multi-channel geodesic-flow classifier.

    Pipeline (every step commutes with, or is invariant under, SO(3, C)):
      1. Channel lift  z_c = w_c z  with learnable complex scalars w_c —
         scalar weights act on the channel index, the group on the vector
         index, so the lift is exactly equivariant.
      2. Shared SO3CActivation applied per channel (exact closed-form mode).
         Each channel flows differently because its connection a(s(z_c))
         sees a different invariant s(z_c) = w_c^2 (z . z).
      3. Readout: all pairwise cross-channel invariants z_c(T) . z_d(T)
         (arcsinh-normalised Re/Im) -> MLP -> logits.
    """

    def __init__(
        self,
        out_features: int = 2,
        channels: int = 4,
        hidden: int = 32,
        act_hidden: int = 16,
        T: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.channels = channels

        # Complex channel weights, initialised near 1 (near-identity lift).
        gen = torch.Generator().manual_seed(0)
        w = torch.stack([
            1.0 + 0.1 * torch.randn(channels, dtype=dtype, generator=gen),
            0.1 * torch.randn(channels, dtype=dtype, generator=gen),
        ], dim=-1)
        self.channel_weights = nn.Parameter(w)              # (C, 2) = (Re, Im)

        self.act = SO3CActivation(
            T=T, mode="dynamic", method="exact",
            bound_input="bilinear", hidden=act_hidden, dtype=dtype,
        )

        n_pairs = channels * (channels + 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_pairs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

        iu = torch.triu_indices(channels, channels)
        self.register_buffer("pair_rows", iu[0])
        self.register_buffer("pair_cols", iu[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = real_to_complex(x.to(self.dtype))                       # (B, 3)
        w = torch.complex(self.channel_weights[:, 0],
                          self.channel_weights[:, 1])               # (C,)
        zc = w[None, :, None] * z[:, None, :]                       # (B, C, 3)

        B, C, _ = zc.shape
        zc_flat = self.act(zc.reshape(B * C, 3))                    # complex path
        zc = zc_flat.reshape(B, C, 3)

        S = zc @ zc.transpose(-1, -2)                               # (B, C, C)
        S_pairs = S[:, self.pair_rows, self.pair_cols]              # (B, C(C+1)/2)
        feats = torch.cat(
            [torch.asinh(S_pairs.real), torch.asinh(S_pairs.imag)], dim=-1
        )
        return self.mlp(feats)

    def regularization_loss(self) -> torch.Tensor:
        return self.act.regularization_loss()
