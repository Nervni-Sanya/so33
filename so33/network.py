"""
network.py
----------
SO33Network: convenience wrapper using SO33Activation as a drop-in
activation layer inside a standard Linear -> Activation -> Linear pipeline.

BottleneckClassifier: generic Linear -> activation -> Linear wrapper that
mirrors the SO33Network architecture for any nn.Module activation. Used to
build matched-bottleneck baselines (ReLU/Tanh/GELU/SO(3)/frozen-Γ) so
comparisons are on equal footing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .basis import DIM
from .activation import SO33Activation


class SO33Network(nn.Module):
    """Three-layer network:  Linear(in_features -> 6) -> SO33Activation -> Linear(6 -> out_features).

    The inner dimension is always 6 because SO33Activation operates in R^{3,3}.

    Parameters
    ----------
    in_features  : int    input dimension  (projected to 6 before activation)
    out_features : int    output dimension
    T            : float  ODE integration horizon passed to SO33Activation
    adjoint      : bool   whether to use adjoint backprop
    dtype        : torch.dtype  parameter dtype (default float64)
    signature_only : bool       restrict to so(3) ⊕ so(3) (Euclidean ablation)
    freeze_coeffs  : bool       freeze the connection coefficients (fixed-Γ ablation)

    Example
    -------
    >>> net = SO33Network(in_features=6, out_features=2, T=0.5)
    >>> logits = net(torch.randn(8, 6, dtype=torch.float64))
    >>> logits.shape
    torch.Size([8, 2])
    """

    def __init__(
        self,
        in_features:  int   = 6,
        out_features: int   = 1,
        T:            float = 0.5,
        adjoint:      bool  = True,
        dtype:        torch.dtype = torch.float64,
        signature_only: bool = False,
        freeze_coeffs:  bool = False,
    ) -> None:
        super().__init__()
        self.dtype = dtype

        self.input_proj  = nn.Linear(in_features, DIM).to(dtype)
        self.activation  = SO33Activation(
            T=T, adjoint=adjoint, method="dopri5", rtol=1e-4, atol=1e-5,
            dtype=dtype,
            signature_only=signature_only,
            freeze_coeffs=freeze_coeffs,
        )
        self.output_proj = nn.Linear(DIM, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        h = self.input_proj(x)     # (B, 6)
        h = self.activation(h)     # (B, 6)
        return self.output_proj(h) # (B, out_features)

    def regularization_loss(self) -> torch.Tensor:
        """Frobenius regularization loss from the activation layer."""
        return self.activation.regularization_loss()


class BottleneckClassifier(nn.Module):
    """Generic Linear -> activation -> Linear pipeline through a fixed bottleneck.

    Mirrors the SO33Network architecture so ReLU/Tanh/GELU and the SO(3)
    Euclidean / frozen-Γ ablations all use the same Linear(in -> hidden)
    -> activation -> Linear(hidden -> out) shape. This makes "matched
    bottleneck" baselines honest: every model sees the same compression
    before its activation runs.

    Parameters
    ----------
    in_features  : int        input dimension
    out_features : int        output dimension
    activation   : nn.Module  activation module operating on `hidden`-dim vectors
    hidden       : int        bottleneck dimension (default 6, matching SO(3,3))
    dtype        : torch.dtype  parameter dtype (default float64)

    Notes
    -----
    The activation must accept and return tensors of shape (B, hidden) with
    the configured dtype. For SO33Activation use `hidden=6` and a matching
    dtype. For pointwise activations (ReLU/Tanh/GELU) any hidden width works.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        activation:   nn.Module,
        hidden:       int = DIM,
        dtype:        torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype

        self.input_proj  = nn.Linear(in_features, hidden).to(dtype)
        self.activation  = activation
        self.output_proj = nn.Linear(hidden, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        h = self.input_proj(x)
        h = self.activation(h)
        return self.output_proj(h)

    def regularization_loss(self) -> torch.Tensor:
        """Forward to activation.regularization_loss if present, else 0.

        Lets training code call `model.regularization_loss()` uniformly
        regardless of whether the activation has a regulariser.
        """
        reg = getattr(self.activation, "regularization_loss", None)
        if callable(reg):
            return reg()
        return torch.zeros((), dtype=self.dtype)
