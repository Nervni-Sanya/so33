"""
ode_func.py
-----------
ODEFunc: right-hand side of the geodesic-like ODE in R^{3,3}.

The ODE:
    dv^mu / dt = -sum_{nu, lambda}  omega^mu_{nu lambda}  v^nu  v^lambda

is a state-dependent quadratic vector field on R^{3,3}.  The quadratic term
v x v (outer product) generates dynamics richer than any linear activation:
possible limit cycles, symmetry-breaking bifurcations, and sensitivity to the
causal structure of the initial condition.  Despite having no explicit metric
in the equation, the geometry of R^{3,3} is encoded in the so(3,3)
antisymmetry constraints on omega, which are enforced by construction in basis.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .basis import get_connection_tensor


class ODEFunc(nn.Module):
    """Right-hand side  f(t, v) = -omega(v) v  for torchdiffeq integration.

    This is a stateless nn.Module: it holds references to the trainable
    coefficients and the fixed basis, so that PyTorch's autograd can
    differentiate through the ODE solve w.r.t. the coefficients.

    Parameters
    ----------
    coeffs      : (15,) nn.Parameter — reference from SO33Activation
    basis_stack : (15, 6, 6, 6) buffer — fixed basis tensors
    scale       : float — global multiplier for stability (adaptive in activation.py)
    """

    def __init__(
        self,
        coeffs: torch.Tensor,
        basis_stack: torch.Tensor,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.coeffs      = coeffs
        self.basis_stack = basis_stack
        self.scale       = scale

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Evaluate dv/dt at current state v.

        Parameters
        ----------
        t : scalar tensor   (required by torchdiffeq API; ODE is autonomous)
        v : (B, 6) float64  current velocity batch

        Returns
        -------
        dvdt : (B, 6) float64
            dvdt[b, mu] = -omega[mu, nu, lambda] * v[b, nu] * v[b, lambda]
        """
        # Rebuild omega on every call so autograd can differentiate through coeffs
        omega = get_connection_tensor(self.coeffs, self.basis_stack)  # (6, 6, 6)

        if self.scale != 1.0:
            omega = omega * self.scale

        # Efficient quadratic contraction via Einstein summation
        return -torch.einsum("mnl, bn, bl -> bm", omega, v, v)
