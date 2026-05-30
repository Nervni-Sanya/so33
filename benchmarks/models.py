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
from so33.basis import DIM


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
ALL_MODELS = MATCHED_MODELS + NATURAL_MODELS

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
