"""
activation.py
-------------
SO33Activation: parallel-transport activation layer in R^{3,3}.

Forward map
-----------
    F(x) = v(T)

where v(t) is the solution of the geodesic-like ODE:

    dv^mu / dt = -sum_{nu, lambda}  omega^mu_{nu lambda}  v^nu  v^lambda
    v(0) = x

Trainable parameters
--------------------
    coeffs in R^15  —  the 15 scalar coefficients of the so(3,3) connection basis
                        (minimal parameterisation, all other entries are derived)

Numerical stability (indefinite metric)
---------------------------------------
The split signature (+,+,+,-,-,-) allows hyperbolic trajectories analogous
to Lorentz boosts, which can diverge exponentially at random initialisation.
Mitigations applied:
  1. Small init:   coeffs ~ N(0, 0.01)  →  omega ≈ 0  at start
  2. Adaptive scale:  omega_eff = omega / (1 + ||omega||_F)  →  ||omega_eff|| < 1
  3. Frobenius regularisation:  L_reg = reg_coef * ||omega||_F^2  (add to loss)
  4. Gradient clipping recommended in outer training loop (max_norm = 1.0)

Backward pass
-------------
Approach A (adjoint=True, default):
    torchdiffeq.odeint_adjoint implements the continuous adjoint sensitivity
    method. Memory cost is O(1) w.r.t. trajectory length. The adjoint ODE is:

        dp^mu / dt = +2 * sum_{nu, lambda}  p^nu  omega^nu_{mu lambda}  v^lambda

    and the parameter gradient accumulates as:

        dL/dc_k = -integral_0^T  sum_{mu,nu,lambda}
                      p^mu  omega^{(k),mu}_{nu lambda}  v^nu  v^lambda  dt

Approach B (adjoint=False):
    Standard autograd through all ODE steps. Stores full trajectory in memory.
    Useful for debugging; use for short sequences only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .basis import get_basis_stack, get_connection_tensor, N_BASIS
from .ode_func import ODEFunc


class SO33Activation(nn.Module):
    """Parallel-Transport Activation in Pseudo-Euclidean Space R^{3,3}.

    Parameters
    ----------
    T        : float   integration horizon  (default 1.0)
    rtol     : float   ODE relative tolerance  (default 1e-4)
    atol     : float   ODE absolute tolerance  (default 1e-5)
    method   : str     ODE solver — 'dopri5' (default), 'rk4', 'euler'
    adjoint  : bool    True  → adjoint backprop (memory-efficient, recommended)
                       False → direct autograd (debugging)
    reg_coef : float   Frobenius regularisation coefficient  (default 1e-3)

    Input  : (B, 6) float32 or float64  — cast to float64 internally
    Output : (B, 6) float64
    """

    def __init__(
        self,
        T: float        = 1.0,
        rtol: float     = 1e-4,
        atol: float     = 1e-5,
        method: str     = "dopri5",
        adjoint: bool   = True,
        reg_coef: float = 1e-3,
    ) -> None:
        super().__init__()

        self.T        = T
        self.rtol     = rtol
        self.atol     = atol
        self.method   = method
        self.adjoint  = adjoint
        self.reg_coef = reg_coef

        # ── Trainable parameters: 15 so(3,3) connection coefficients ──────────
        # Small init keeps the ODE near-identity at the start of training.
        self.coeffs = nn.Parameter(
            torch.randn(N_BASIS, dtype=torch.float64) * 0.01
        )

        # ── Fixed basis (non-trainable, moves with .to(device)) ───────────────
        self.register_buffer("basis_stack", get_basis_stack())   # (15, 6, 6, 6)

        # ── Time integration interval ─────────────────────────────────────────
        self.register_buffer(
            "t_span",
            torch.tensor([0.0, T], dtype=torch.float64),
        )

    # ── Stability helpers ─────────────────────────────────────────────────────

    def _adaptive_scale(self) -> float:
        """Compute 1 / (1 + ||omega||_F) to bound the effective connection."""
        with torch.no_grad():
            omega = get_connection_tensor(self.coeffs, self.basis_stack)
            frob  = omega.norm(p="fro").item()
        return 1.0 / (1.0 + frob)

    def regularization_loss(self) -> torch.Tensor:
        """Frobenius regularisation: L_reg = reg_coef * ||omega||_F^2.

        Add this term to the training loss to prevent trajectories from
        diverging under the indefinite pseudo-Euclidean metric.
        """
        omega = get_connection_tensor(self.coeffs, self.basis_stack)
        return self.reg_coef * omega.pow(2).sum()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate the ODE from t=0 to t=T and return v(T).

        Parameters
        ----------
        x : (B, 6) tensor — interpreted as initial velocity v(0)

        Returns
        -------
        y : (B, 6) float64 tensor — terminal state v(T)
        """
        x     = x.double()           # ensure float64 for ODE solver
        scale = self._adaptive_scale()

        ode_func = ODEFunc(
            coeffs=self.coeffs,
            basis_stack=self.basis_stack,
            scale=scale,
        )

        if self.adjoint:
            # Approach A: memory-efficient continuous adjoint
            v_traj = odeint_adjoint(
                ode_func,
                x,
                self.t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
                adjoint_params=tuple(self.parameters()),
            )
        else:
            # Approach B: direct autograd (full trajectory in memory)
            v_traj = odeint(
                ode_func,
                x,
                self.t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )

        return v_traj[-1]   # v(T), shape (B, 6)

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, method={self.method}, "
            f"adjoint={self.adjoint}, n_params={N_BASIS}"
        )
