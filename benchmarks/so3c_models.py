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
        return torch.zeros((), dtype=self.dtype)


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
        return torch.zeros((), dtype=self.dtype)


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
