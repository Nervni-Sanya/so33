"""
benchmarks.models
-----------------
Model factory used by the benchmark harness.

Two families:
- Matched-bottleneck (Linear -> hidden=6 -> activation -> Linear).
  Every model in this family sees the same compression before the
  activation runs, so accuracy comparisons isolate the effect of the
  activation itself. Used for the "scientific" comparison.
- Natural-width MLP (Linear(in -> H) -> activation -> Linear(H -> out)).
  Standard 256-256 baseline that does NOT bottleneck through 6.
  Used for the "engineering" comparison: is SO33 competitive in
  practice given its hard 6-dim constraint?

The factory takes a string name + dimensions and returns an
``nn.Module``. SO33 variants automatically expose
``regularization_loss()``; pointwise activations get a no-op via
BottleneckClassifier.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from so33 import SO33Activation, SO33Network, BottleneckClassifier
from so33.basis import DIM, ETA


def _lift_4to6(p4: torch.Tensor) -> torch.Tensor:
    """Equivariant lift of a (..., 4) Minkowski 4-vector into R^{3,3}.

    Layout matches ETA = diag(+1,+1,+1,-1,-1,-1):
        out[..., 0:3] = (px, py, pz)        spacelike (+)
        out[...,   3] = E                    timelike  (-)
        out[..., 4:6] = 0                    unused timelike axes

    This is deterministic (no learned parameters), so SO(3,3) boosts
    acting on the lifted vector correspond exactly to the standard
    Lorentz action on the original (E, p). A learned Linear(4->6) would
    not commute with SO(3,3) and is why the previous OOD experiment
    showed no advantage for SO33-based models.
    """
    shape = p4.shape[:-1] + (DIM,)
    out = p4.new_zeros(shape)
    out[..., 0] = p4[..., 1]
    out[..., 1] = p4[..., 2]
    out[..., 2] = p4[..., 3]
    out[..., 3] = p4[..., 0]
    return out


# Names the factory understands.
MATCHED_MODELS = (
    "so33",
    "so33_signature_only",
    "so33_frozen",
    "relu_bottleneck",
    "tanh_bottleneck",
    "gelu_bottleneck",
)
NATURAL_MODELS = (
    "relu_mlp",
    "tanh_mlp",
    "gelu_mlp",
    "so33_multi",          # multi-channel SO33: capacity comparison vs wide MLPs
)
SET_MODELS = (
    # Set-based equivariant classifiers — only valid for representation="constituents".
    "eta_invariants",      # SO(3,3)-invariant features (Arch A)
    "so33_equivariant",    # equivariant lift + SO33 + invariant readout (Arch B)
    "so33_equivariant_frozen",       # Arch B with freeze_coeffs=True
    "so33_equivariant_unbounded",    # Arch B with bound_input=False
    "so33_equivariant_eta_bounded",  # Arch B with eta-invariant input bound
)
ALL_MODELS = MATCHED_MODELS + NATURAL_MODELS + SET_MODELS

# Default number of parallel SO(3,3) blocks for the "so33_multi" variant.
SO33_MULTI_CHANNELS = 4


# ─────────────────────────────────────────────────────────────────────────
# Natural-width MLP
# ─────────────────────────────────────────────────────────────────────────

class NaturalWidthMLP(nn.Module):
    """Linear(in -> H) -> activation -> Linear(H -> out), no bottleneck.

    Used as the engineering-fairness baseline: MLPs use a wide hidden
    layer (default 256) while SO33 stays at 6. Tests whether the
    geometric model is competitive *in practice* despite its compression.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        activation:   nn.Module,
        hidden:       int = 256,
        dtype:        torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.l1 = nn.Linear(in_features, hidden).to(dtype)
        self.activation = activation
        self.l2 = nn.Linear(hidden, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        return self.l2(self.activation(self.l1(x)))

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype)


# ─────────────────────────────────────────────────────────────────────────
# Per-constituent Deep Sets (Top Tagging headline)
# ─────────────────────────────────────────────────────────────────────────

class DeepSetsClassifier(nn.Module):
    """Per-particle phi -> masked mean pool -> rho head.

    The headline Top Tagging architecture. Each jet arrives as a
    (B, K, 5) tensor where ``[..., :4]`` is the standardised
    (E, px, py, pz) of each of K constituents and ``[..., 4]`` is a
    mask (1.0 real, 0.0 padding). The same per-particle function phi =
    activation(Linear(4 -> hidden)) is applied to every constituent;
    SO33 uses ``hidden=6`` so the activation is the SO(3,3) geodesic
    flow acting on each particle's embedded 4-momentum. A masked mean
    pool over constituents feeds the linear classifier head.

    Unlike the aggregated jet-level loader, this keeps the per-particle
    substructure, so the geometric prior has something to exploit.
    All baselines (ReLU/Tanh/GELU) share this exact skeleton with their
    activation swapped in, keeping the comparison apples-to-apples.
    """

    def __init__(
        self,
        activation:   nn.Module,
        out_features: int,
        in_features:  int = 4,
        hidden:       int = DIM,
        dtype:        torch.dtype = torch.float64,
        channels:     list[nn.Module] | None = None,
        pool:         str = "mean",
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.pool = pool
        # Single-channel (default) uses ``activation``; multi-channel passes a
        # list of parallel activations, each with its own embedding, whose
        # pooled outputs are concatenated before the head.
        if channels is None:
            self.embeds = nn.ModuleList([nn.Linear(in_features, hidden).to(dtype)])
            self.acts   = nn.ModuleList([activation])
            head_in = hidden
        else:
            self.embeds = nn.ModuleList(
                [nn.Linear(in_features, hidden).to(dtype) for _ in channels]
            )
            self.acts = nn.ModuleList(channels)
            head_in = hidden * len(channels)
        self.head = nn.Linear(head_in, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        feat = x[..., :4]                       # (B, K, 4)
        mask = x[..., 4:5]                       # (B, K, 1)
        B, K, _ = feat.shape

        pooled_channels = []
        for embed, act in zip(self.embeds, self.acts):
            h = embed(feat)                                   # (B, K, hidden)
            h = act(h.reshape(B * K, -1)).reshape(B, K, -1)
            h = h * mask                                      # zero out padding
            if self.pool == "sum":
                pooled = h.sum(dim=1)
            else:                                             # masked mean
                pooled = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled_channels.append(pooled)
        return self.head(torch.cat(pooled_channels, dim=-1))

    def regularization_loss(self) -> torch.Tensor:
        total = torch.zeros((), dtype=self.dtype)
        for act in self.acts:
            reg = getattr(act, "regularization_loss", None)
            if callable(reg):
                total = total + reg()
        return total


# ─────────────────────────────────────────────────────────────────────────
# Multi-channel SO33 (capacity comparison)
# ─────────────────────────────────────────────────────────────────────────

class MultiChannelSO33(nn.Module):
    """C parallel SO(3,3) blocks, concatenated, then a linear head.

    A single SO33 block is locked to 6 hidden dims by the algebra, which
    caps its capacity well below a wide MLP. This runs C independent
    Linear(in->6) -> SO33Activation blocks in parallel and concatenates
    their 6-D outputs (C*6 features) before the classifier. Each block
    keeps the full geometric prior; together they provide the width that
    lets SO33 compete with natural-width MLPs on raw accuracy.

    Works on flat inputs (B, in_features). For per-constituent data use
    ``DeepSetsClassifier`` with ``channels > 1``.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        *,
        channels: int = SO33_MULTI_CHANNELS,
        T: float = 0.3,
        adjoint: bool = True,
        dtype: torch.dtype = torch.float64,
        bound_input: bool = False,
        so33_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        so33_kwargs = so33_kwargs or {}
        self.embeds = nn.ModuleList(
            [nn.Linear(in_features, DIM).to(dtype) for _ in range(channels)]
        )
        self.acts = nn.ModuleList([
            SO33Activation(T=T, adjoint=adjoint, dtype=dtype,
                           bound_input=bound_input, **so33_kwargs)
            for _ in range(channels)
        ])
        self.head = nn.Linear(DIM * channels, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        h = torch.cat([a(e(x)) for e, a in zip(self.embeds, self.acts)], dim=-1)
        return self.head(h)

    def regularization_loss(self) -> torch.Tensor:
        return sum(a.regularization_loss() for a in self.acts)


# ─────────────────────────────────────────────────────────────────────────
# Equivariant architectures (the OOD-headline experiments)
# ─────────────────────────────────────────────────────────────────────────

class EtaInvariantsClassifier(nn.Module):
    """SO(3,3)-INVARIANT-by-construction classifier (Arch A).

    Per-particle features are reduced to the only quantities that survive
    an SO(3,3) transformation: the η-norm  m_i² = <p_i, η p_i>  of each
    constituent and the pairwise η-inner products  s_ij = <p_i, η p_j>.
    Those scalars are pooled into permutation-invariant statistics and
    fed to a plain MLP. The whole network commutes with any SO(3,3) boost
    BY CONSTRUCTION (no approximation), so the model is structurally
    immune to the boost-OOD failure mode that hits Linear-wrapped designs.

    No SO33Activation here on purpose: this is the maximally-conservative
    "use only the η metric, nothing else" baseline. If even this loses on
    OOD, the inductive bias hypothesis is wrong; if it wins, we have a
    clean demonstration of where the prior helps.
    """

    def __init__(
        self,
        out_features: int,
        hidden: int = 64,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.register_buffer("eta", ETA.to(dtype))

        # 7 pooled invariant statistics (see forward) -> small MLP -> logits.
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4   = x[..., :4]                                  # (B, K, 4)
        mask = x[..., 4]                                    # (B, K)
        p    = _lift_4to6(p4)                               # (B, K, 6)

        eta = self.eta                                      # (6,)
        # Per-particle η-norm  m_i^2 = <p_i, η p_i>
        m2 = (p * eta * p).sum(dim=-1)                      # (B, K)
        m2 = m2 * mask
        # Pairwise η-inner products  s_ij = <p_i, η p_j>
        s  = torch.einsum("bki,i,bli->bkl", p, eta, p)      # (B, K, K)
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)   # (B, K, K)
        s = s * pair_mask
        eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
        s_off = s * (1.0 - eye)                             # off-diagonal pairs

        n_real      = mask.sum(dim=-1).clamp_min(1.0)
        n_real_pair = pair_mask.sum(dim=(-2, -1)).clamp_min(1.0)

        feats = torch.stack([
            m2.sum(dim=-1) / n_real,                        # mean per-particle mass²
            m2.pow(2).sum(dim=-1) / n_real,                 # mean (mass²)²
            m2.abs().sum(dim=-1) / n_real,                  # mean |mass²|
            s_off.sum(dim=(-2, -1)) / n_real_pair,          # mean pairwise inner
            s_off.pow(2).sum(dim=(-2, -1)) / n_real_pair,   # mean squared pairwise
            s_off.abs().max(dim=-1).values.max(dim=-1).values,    # max |pair|
            (p.sum(dim=1) * eta * p.sum(dim=1)).sum(dim=-1),       # jet-total <P,ηP>
        ], dim=-1)                                          # (B, 7)
        return self.mlp(feats)

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype)


class EquivariantSO33Classifier(nn.Module):
    """Equivariant lift + per-particle SO33Activation + INVARIANT readout (Arch B).

    Unlike DeepSetsClassifier (whose Linear(4->6) breaks SO(3,3) equivariance),
    this:
      1. lifts each 4-momentum to R^{3,3} by deterministic placement
         (``_lift_4to6``), which IS equivariant;
      2. applies the SO33Activation per particle (empirically equivariant to
         <0.13% relative error at init, ~3.4% after heavy training);
      3. reads out only SO(3,3)-INVARIANT scalars built from the activated
         outputs: per-particle η-norm and pairwise η-inner products with the
         jet-total vector, pooled per channel;
      4. classifies the pooled invariants with a small MLP.

    Multi-channel: each of ``channels`` SO33Activations sees the same
    lifted input but learns its own connection, contributing independent
    invariant features for the readout. This is the "test the ODE kernel
    inside a structurally equivariant frame" architecture.
    """

    def __init__(
        self,
        out_features: int,
        *,
        channels: int = SO33_MULTI_CHANNELS,
        T: float = 0.3,
        adjoint: bool = True,
        dtype: torch.dtype = torch.float64,
        bound_input: bool | str = True,
        head_hidden: int = 32,
        so33_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.register_buffer("eta", ETA.to(dtype))
        so33_kwargs = so33_kwargs or {}
        self.acts = nn.ModuleList([
            SO33Activation(T=T, adjoint=adjoint, dtype=dtype,
                           bound_input=bound_input, **so33_kwargs)
            for _ in range(channels)
        ])
        # 3 invariants per channel: mean m^2, mean |m|, <jet-total, η jet-total>
        n_feats = 3 * channels
        self.head = nn.Sequential(
            nn.Linear(n_feats, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, out_features),
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4   = x[..., :4]
        mask = x[..., 4]
        p    = _lift_4to6(p4)                               # (B, K, 6) equivariant

        B, K, _ = p.shape
        eta = self.eta
        n_real = mask.sum(dim=-1).clamp_min(1.0)

        feats = []
        for act in self.acts:
            h = act(p.reshape(B * K, DIM)).reshape(B, K, DIM)
            h = h * mask.unsqueeze(-1)                      # zero out padding
            m2 = (h * eta * h).sum(dim=-1)                  # (B, K) — INVARIANT
            jet_total = h.sum(dim=1)                        # (B, 6)
            jet_inv = (jet_total * eta * jet_total).sum(-1)  # (B,) — INVARIANT
            feats.extend([
                m2.sum(dim=-1) / n_real,
                m2.abs().sum(dim=-1) / n_real,
                jet_inv,
            ])
        return self.head(torch.stack(feats, dim=-1))

    def regularization_loss(self) -> torch.Tensor:
        return sum(a.regularization_loss() for a in self.acts)


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────

def _build_deepsets(
    name: str,
    in_features: int,
    out_features: int,
    *,
    T: float,
    adjoint: bool,
    dtype: torch.dtype,
    natural_hidden: int,
    so33_kwargs: dict,
    bound_input: bool,
    pool: str,
) -> nn.Module:
    """Construct the per-constituent Deep Sets variant for a model name."""
    def make_so33(**extra):
        return SO33Activation(
            T=T, adjoint=adjoint, dtype=dtype,
            bound_input=bound_input, **extra, **so33_kwargs,
        )

    if name in ("so33", "so33_signature_only", "so33_frozen"):
        act = make_so33(
            signature_only=(name == "so33_signature_only"),
            freeze_coeffs=(name == "so33_frozen"),
        )
        return DeepSetsClassifier(
            activation=act, out_features=out_features,
            in_features=in_features, hidden=DIM, dtype=dtype, pool=pool,
        )

    if name == "so33_multi":
        chans = [make_so33() for _ in range(SO33_MULTI_CHANNELS)]
        return DeepSetsClassifier(
            activation=None, out_features=out_features,
            in_features=in_features, hidden=DIM, dtype=dtype,
            channels=chans, pool=pool,
        )

    if name == "eta_invariants":
        return EtaInvariantsClassifier(out_features=out_features, dtype=dtype)

    if name == "so33_equivariant":
        return EquivariantSO33Classifier(
            out_features=out_features, T=T, adjoint=adjoint, dtype=dtype,
            bound_input=bound_input, so33_kwargs=so33_kwargs,
        )

    if name == "so33_equivariant_frozen":
        # Ablation 1: freeze the connection coefficients. Tests whether
        # training-time drift in Γ is what destroys OOD generalisation.
        # If OOD AUC stays ~0.663, learnable Γ is not the cause.
        return EquivariantSO33Classifier(
            out_features=out_features, T=T, adjoint=adjoint, dtype=dtype,
            bound_input=bound_input,
            so33_kwargs=dict(so33_kwargs, freeze_coeffs=True),
        )

    if name == "so33_equivariant_unbounded":
        # Ablation 2: disable the bound_input normalisation. The default
        # bound_input divides by  1 + ||x||_2 , and the Euclidean norm is
        # NOT an SO(3,3) invariant -- boosted inputs are scaled by a
        # different factor, so the activation loses equivariance with
        # input rapidity. Removing the bound restores exact equivariance
        # at the risk of geodesic blow-up; if OOD AUC recovers, the
        # bound is confirmed as the cause.
        return EquivariantSO33Classifier(
            out_features=out_features, T=T, adjoint=adjoint, dtype=dtype,
            bound_input=False, so33_kwargs=so33_kwargs,
        )

    if name == "so33_equivariant_eta_bounded":
        # Principled fix: replace the Euclidean-norm bound with the
        # eta-norm bound  x / (1 + sqrt(|x . eta . x|)). This is
        # SO(3,3)-invariant by construction (x . eta . x is the
        # indefinite-metric invariant), so equivariance is preserved
        # exactly while the input is still bounded for ODE stability on
        # real-data settings (HIGGS, top tagging) where the unbounded
        # variant diverges.
        return EquivariantSO33Classifier(
            out_features=out_features, T=T, adjoint=adjoint, dtype=dtype,
            bound_input="eta", so33_kwargs=so33_kwargs,
        )

    pointwise: dict[str, Callable[[], nn.Module]] = {
        "relu_bottleneck": nn.ReLU, "tanh_bottleneck": nn.Tanh,
        "gelu_bottleneck": nn.GELU,
        "relu_mlp": nn.ReLU, "tanh_mlp": nn.Tanh, "gelu_mlp": nn.GELU,
    }
    if name in pointwise:
        hidden = DIM if name.endswith("_bottleneck") else natural_hidden
        return DeepSetsClassifier(
            activation=pointwise[name](), out_features=out_features,
            in_features=in_features, hidden=hidden, dtype=dtype, pool=pool,
        )

    raise ValueError(f"Unknown model name: {name!r}. Known: {ALL_MODELS}")


def build_model(
    name: str,
    in_features: int,
    out_features: int,
    *,
    T: float = 0.3,
    natural_hidden: int = 256,
    dtype: torch.dtype = torch.float64,
    adjoint: bool = True,
    so33_method: str = "rk4",
    so33_step_size: float | None = None,
    representation: str = "flat",
    bound_input: bool | None = None,
    max_input_norm: float | None = 8.0,
    pool: str = "mean",
) -> nn.Module:
    """Construct a model by string name.

    Matched-bottleneck names use hidden=6 throughout. Natural-width
    names use ``natural_hidden`` (default 256).

    Parameters
    ----------
    name        : one of ALL_MODELS.
    in_features : input dimensionality. For ``representation="constituents"``
                  this is the per-particle feature count (4 = E,px,py,pz).
    out_features: number of classes (or regression outputs).
    T           : SO33 ODE integration horizon (so33 variants only).
    natural_hidden : hidden size for natural-width MLPs.
    dtype       : parameter dtype.
    adjoint     : adjoint backprop flag for SO33 variants.
    representation : "flat" (default; input is (B, in_features)) or
                  "constituents" (input is (B, K, 5) — a Deep Sets model
                  applies the activation per particle then masked-pools).

    Returns
    -------
    nn.Module with ``forward(x)`` and ``regularization_loss()``.
    """
    # SO33 variants: default to fixed-step rk4 in the benchmark harness.
    # Real-data inputs (HIGGS 28-d, Adult 14-d, Top-Tagging jets) push the
    # connection coefficients into a regime where the indefinite-metric
    # geodesic ODE diverges fast enough that adaptive dopri5 underflows
    # dt to zero. rk4 with a small fixed step never underflows, runs in
    # O(T/step) RHS evals, and is plenty accurate for classification.
    if so33_method == "rk4":
        step = so33_step_size if so33_step_size is not None else max(T / 10.0, 1e-3)
        so33_kwargs = dict(method="rk4", solver_options={"step_size": step})
    else:
        so33_kwargs = dict(method=so33_method)
        if so33_step_size is not None:
            so33_kwargs["solver_options"] = {"step_size": so33_step_size}

    # bound_input is representation-aware: on per-constituent 4-momenta the
    # geodesic diverges (boosts -> large ||v|| -> NaN) so it must be on;
    # on flat already-standardised features (HIGGS/Adult) it squashes signal
    # and hurts, so it stays off. Callers may override explicitly.
    if bound_input is None:
        bound_input = (representation == "constituents")

    if representation == "constituents":
        return _build_deepsets(
            name, in_features, out_features,
            T=T, adjoint=adjoint, dtype=dtype,
            natural_hidden=natural_hidden, so33_kwargs=so33_kwargs,
            bound_input=bound_input, pool=pool,
        )

    # Flat real-data path: when bound_input is off (which it is by default,
    # to avoid squashing standardised features), add a soft norm cap as a
    # safety net. Real HIGGS has heavy-tailed features that occasionally
    # drive the geodesic to a stochastic NaN (so33 -> 0.473 in one run,
    # 0.752 in another). The cap only rescales outliers, so it does not
    # hurt the typical-input signal the way bound_input does.
    if not bound_input and max_input_norm is not None:
        so33_kwargs = dict(so33_kwargs, max_input_norm=max_input_norm)

    if name == "so33":
        return SO33Network(
            in_features=in_features, out_features=out_features,
            T=T, adjoint=adjoint, dtype=dtype, bound_input=bound_input,
            **so33_kwargs,
        )
    if name == "so33_signature_only":
        return SO33Network(
            in_features=in_features, out_features=out_features,
            T=T, adjoint=adjoint, dtype=dtype, signature_only=True,
            bound_input=bound_input, **so33_kwargs,
        )
    if name == "so33_frozen":
        return SO33Network(
            in_features=in_features, out_features=out_features,
            T=T, adjoint=adjoint, dtype=dtype, freeze_coeffs=True,
            bound_input=bound_input, **so33_kwargs,
        )
    if name == "so33_multi":
        return MultiChannelSO33(
            in_features=in_features, out_features=out_features,
            channels=SO33_MULTI_CHANNELS, T=T, adjoint=adjoint, dtype=dtype,
            bound_input=bound_input, so33_kwargs=so33_kwargs,
        )

    pointwise: dict[str, Callable[[], nn.Module]] = {
        "relu_bottleneck": nn.ReLU,
        "tanh_bottleneck": nn.Tanh,
        "gelu_bottleneck": nn.GELU,
        "relu_mlp":        nn.ReLU,
        "tanh_mlp":        nn.Tanh,
        "gelu_mlp":        nn.GELU,
    }
    if name in pointwise:
        act = pointwise[name]()
        if name.endswith("_bottleneck"):
            return BottleneckClassifier(
                in_features=in_features, out_features=out_features,
                activation=act, hidden=6, dtype=dtype,
            )
        return NaturalWidthMLP(
            in_features=in_features, out_features=out_features,
            activation=act, hidden=natural_hidden, dtype=dtype,
        )

    raise ValueError(
        f"Unknown model name: {name!r}. "
        f"Known: {ALL_MODELS}"
    )


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
