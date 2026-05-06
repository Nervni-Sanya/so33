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

    This is a stateless nn.Module: it holds a reference to the precomputed
    connection tensor omega so that autograd can differentiate through the
    ODE solve back to whatever produced omega (e.g. SO33Activation.coeffs).

    omega is a non-leaf tensor produced by SO33Activation.forward; the same
    omega is reused for every ODE step within a single forward call. This
    saves an einsum per step (the coefficients are constant during a solve).

    Parameters
    ----------
    omega       : (6, 6, 6) tensor — precomputed connection (already scaled).
    coeffs      : (15,) tensor (deprecated, kept for backward compatibility).
    basis_stack : (15, 6, 6, 6) tensor (deprecated, kept for backward compatibility).
    scale       : float — kept for backward compatibility; ignored when omega is provided.

    Either pass `omega` directly (fast path), or pass `coeffs` + `basis_stack`
    (legacy path; rebuilds omega here).
    """

    def __init__(
        self,
        omega: torch.Tensor | None = None,
        coeffs: torch.Tensor | None = None,
        basis_stack: torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if omega is None:
            if coeffs is None or basis_stack is None:
                raise ValueError(
                    "ODEFunc requires either `omega`, or both `coeffs` and `basis_stack`."
                )
            omega = get_connection_tensor(coeffs, basis_stack)
            if scale != 1.0:
                omega = omega * scale
        self.omega = omega

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
        return -torch.einsum("mnl, bn, bl -> bm", self.omega, v, v)
