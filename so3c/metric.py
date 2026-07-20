"""
metric.py
---------
Dynamic Hermitian metric on C^3, parameterised by SO(3, C) invariants.

Construction
------------
The metric is Hermitian with a fixed Euclidean real part and a learned,
state-dependent imaginary (Kahler) part:

    g(s) = I_3 + i [beta(s)]_x ,      beta(s) in R^3

[beta]_x is real antisymmetric, so i [beta]_x is Hermitian and g(s) is a
well-defined Hermitian form for every s — complex "lengths" never appear.

Alongside beta the module predicts a rotational component rho(s) in R^3; the
pair combines into the connection (an so(3, C) algebra element)

    a(s) = rho(s) + i beta(s)   <->   [a]_x ,

which drives the geodesic flow dz/dt = -[a]_x z in the activation layer.
The boost part beta is exactly the imaginary part of the dynamic metric; the
rotation part rho is the metric-compatible rotational connection.

Equivariance discipline (the central lesson of the so33 paper)
--------------------------------------------------------------
The coefficients depend on the state ONLY through the SO(3, C) invariants
s = (Re z.z, Im z.z). Any dependence on raw coordinates would silently break
the group structure — this is the complexified analogue of the eta-bound
result in the parent work (Euclidean bound: OOD AUC 0.663; invariant bound:
1.000).

Initialisation: the final linear layer starts at zero, so a(s) = 0 and the
induced flow is the identity map — training moves away from a safe start.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .algebra import cross_matrix, invariant_features


class HermitianMetric(nn.Module):
    """MLP from invariants s = (Re z.z, Im z.z) to a connection a = rho + i beta.

    Parameters
    ----------
    hidden      : int   width of the single hidden layer (default 16)
    dtype       : torch.dtype   real parameter dtype (default float64)

    forward(v) : (…, 6) real state -> (…, 3) complex algebra element a(s(v)).
    """

    def __init__(self, hidden: int = 16, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 6, dtype=dtype),
        )
        # Zero-init the output layer: a(s) = 0 -> identity flow at start.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        s = invariant_features(v)                      # (…, 2), invariant
        out = self.net(s)                              # (…, 6)
        rho, beta = out[..., :3], out[..., 3:]
        return torch.complex(rho, beta)                # (…, 3) complex

    def metric_tensor(self, v: torch.Tensor) -> torch.Tensor:
        """The Hermitian metric g(s) = I + i [beta(s)]_x, shape (…, 3, 3).

        Diagnostic / interpretability accessor; the flow itself consumes the
        full connection a(s) from forward().
        """
        beta = self.forward(v).imag
        eye = torch.eye(3, dtype=beta.dtype, device=beta.device)
        return eye + 1j * cross_matrix(beta)

    def weight_penalty(self) -> torch.Tensor:
        """Sum of squared output-layer weights — regularisation proxy that
        bounds ||a(s)|| uniformly (tanh hidden activations are in [-1, 1])."""
        last: nn.Linear = self.net[-1]
        return last.weight.pow(2).sum() + last.bias.pow(2).sum()
