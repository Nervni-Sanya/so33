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
from so3c.algebra import invariant_features, real_to_complex
from so3c.interaction import SO3CInteraction
from so3c.lift import jet_bivectors, minkowski_inner


# ─────────────────────────────────────────────────────────────────────────
# Shared pooled-invariant readouts for set (per-constituent) models.
# Design informed by the so33 Arch-B failure (readout omitted the pairwise
# term -> chance level): every readout here includes pairwise statistics.
# ─────────────────────────────────────────────────────────────────────────

def _pooled_bivector_invariants(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, K, 3) complex states + (B, K) mask -> (B, 11) invariant features.

    All features are SO(3, C)-invariant and permutation-invariant: masked
    moments of the per-particle invariants z_a . z_a, of the pairwise
    invariants z_a . z_b, and the jet-total invariant. arcsinh-compressed.
    For raw (simple) bivectors the Im parts vanish identically; after a
    geodesic flow the states are no longer simple and Im features carry
    signal — which is exactly the flow's contribution to the readout.
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

    feats = torch.stack([
        s_diag.real.sum(-1) / n,
        s_diag.real.abs().sum(-1) / n,
        s_diag.imag.sum(-1) / n,
        S_off.real.sum((-2, -1)) / n_off,
        S_off.real.pow(2).sum((-2, -1)) / n_off,
        S_off.real.abs().amax(dim=(-2, -1)),
        S_off.imag.sum((-2, -1)) / n_off,
        S_off.imag.pow(2).sum((-2, -1)) / n_off,
        S_off.imag.abs().amax(dim=(-2, -1)),
        q_tot.real,
        q_tot.imag,
    ], dim=-1)                                                   # (B, 11)
    return torch.asinh(feats)


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
        self.mlp = nn.Sequential(
            nn.Linear(18, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4, mask = x[..., :4], x[..., 4]
        z = jet_bivectors(p4, mask)
        return torch.cat(
            [_pooled_bivector_invariants(z, mask), _minkowski_stats(p4, mask)],
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
