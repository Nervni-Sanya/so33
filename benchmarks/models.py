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
)
ALL_MODELS = MATCHED_MODELS + NATURAL_MODELS


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
# Factory
# ─────────────────────────────────────────────────────────────────────────

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
) -> nn.Module:
    """Construct a model by string name.

    Matched-bottleneck names use hidden=6 throughout. Natural-width
    names use ``natural_hidden`` (default 256).

    Parameters
    ----------
    name        : one of ALL_MODELS.
    in_features : input dimensionality.
    out_features: number of classes (or regression outputs).
    T           : SO33 ODE integration horizon (so33 variants only).
    natural_hidden : hidden size for natural-width MLPs.
    dtype       : parameter dtype.
    adjoint     : adjoint backprop flag for SO33 variants.

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

    if name == "so33":
        return SO33Network(
            in_features=in_features, out_features=out_features,
            T=T, adjoint=adjoint, dtype=dtype, **so33_kwargs,
        )
    if name == "so33_signature_only":
        return SO33Network(
            in_features=in_features, out_features=out_features,
            T=T, adjoint=adjoint, dtype=dtype, signature_only=True,
            **so33_kwargs,
        )
    if name == "so33_frozen":
        return SO33Network(
            in_features=in_features, out_features=out_features,
            T=T, adjoint=adjoint, dtype=dtype, freeze_coeffs=True,
            **so33_kwargs,
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
