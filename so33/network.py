"""
network.py
----------
SO33Network: convenience wrapper using SO33Activation as a drop-in
activation layer inside a standard Linear -> Activation -> Linear pipeline.
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
    ) -> None:
        super().__init__()

        self.input_proj  = nn.Linear(in_features, DIM).double()
        self.activation  = SO33Activation(
            T=T, adjoint=adjoint, method="dopri5", rtol=1e-4, atol=1e-5
        )
        self.output_proj = nn.Linear(DIM, out_features).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.double()
        h = self.input_proj(x)     # (B, 6)
        h = self.activation(h)     # (B, 6)
        return self.output_proj(h) # (B, out_features)

    def regularization_loss(self) -> torch.Tensor:
        """Frobenius regularization loss from the activation layer."""
        return self.activation.regularization_loss()
