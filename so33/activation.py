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
    coeffs in R^K  —  scalar coefficients of the connection basis.
        K = 15 for full so(3,3), K = 6 for the so(3) ⊕ so(3) Euclidean
        ablation (signature_only=True).

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
    T              : float   integration horizon  (default 1.0)
    rtol           : float   ODE relative tolerance  (default 1e-4)
    atol           : float   ODE absolute tolerance  (default 1e-5)
    method         : str     ODE solver — 'dopri5' (default), 'rk4', 'euler'
    adjoint        : bool    True  → adjoint backprop (memory-efficient, recommended)
                             False → direct autograd (debugging)
    reg_coef       : float   Frobenius regularisation coefficient  (default 1e-3)
    dtype          : torch.dtype   parameter and basis dtype (default float64).
                                   Pass torch.float32 for faster benchmarking
                                   at the cost of numerical stability under the
                                   indefinite metric.
    signature_only : bool    When True, restrict the basis to same-signature
                             generators (so(3) ⊕ so(3), 6 generators, no
                             cross-signature Lorentz boosts). Used for the
                             Euclidean ablation. Default False.
    freeze_coeffs  : bool    When True, freeze the connection coefficients at
                             their initial random values so that gradients do
                             not update them. Used for the "fixed-Γ" ablation
                             which isolates whether *learning* the connection
                             matters versus simply having geometric structure.
                             Default False.
    bound_input    : bool    When True, normalise each input vector to a
                             bounded magnitude before integration:
                                 x' = x / (1 + ||x||_2)
                             so ||x'|| < 1. The geodesic ODE under an
                             indefinite metric can diverge exponentially,
                             which makes the adaptive solver fail with
                             ``underflow in dt`` on real-data inputs (e.g.
                             standardised tabular features through a Linear
                             projection). Bounding the initial state keeps
                             ||v||² in check throughout the integration.
                             Default False (preserves the analytical
                             behaviour expected by the existing tests; the
                             SO33Network convenience wrapper enables it by
                             default).

    Input  : (B, 6) tensor — cast to the layer dtype internally.
    Output : (B, 6) tensor in the layer dtype.
    """

    def __init__(
        self,
        T: float        = 1.0,
        rtol: float     = 1e-4,
        atol: float     = 1e-5,
        method: str     = "dopri5",
        adjoint: bool   = True,
        reg_coef: float = 1e-3,
        dtype: torch.dtype = torch.float64,
        signature_only: bool = False,
        freeze_coeffs: bool = False,
        bound_input: bool = False,
    ) -> None:
        super().__init__()

        self.T              = T
        self.rtol           = rtol
        self.atol           = atol
        self.method         = method
        self.adjoint        = adjoint
        self.reg_coef       = reg_coef
        self.dtype          = dtype
        self.signature_only = signature_only
        self.bound_input    = bound_input

        # ── Fixed basis (non-trainable, moves with .to(device)) ───────────────
        basis = get_basis_stack(dtype=dtype, signature_only=signature_only)
        self.register_buffer("basis_stack", basis)               # (K, 6, 6, 6)
        n_coeffs = basis.shape[0]

        # ── Trainable parameters: K connection coefficients ───────────────────
        # Small init keeps the ODE near-identity at the start of training.
        coeffs = torch.randn(n_coeffs, dtype=dtype) * 0.01
        self.coeffs = nn.Parameter(coeffs, requires_grad=not freeze_coeffs)

        # ── Time integration interval ─────────────────────────────────────────
        self.register_buffer(
            "t_span",
            torch.tensor([0.0, T], dtype=dtype),
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
        y : (B, 6) tensor — terminal state v(T), in self.dtype
        """
        x = x.to(self.dtype)
        if self.bound_input:
            # Bound ||x||_2 < 1 so ||v||² stays manageable through the ODE.
            # Without this the indefinite-metric trajectory can blow up
            # exponentially on inputs of moderate magnitude (e.g. standardised
            # tabular features through a Linear), causing the adaptive solver
            # to underflow dt to zero. Differentiable; trains end-to-end.
            x_norm = x.norm(dim=-1, keepdim=True)
            x = x / (1.0 + x_norm)

        scale = self._adaptive_scale()

        # Precompute omega once per forward — coeffs are constant during a
        # single ODE solve, so reusing the same tensor across solver steps
        # saves an einsum per step. Autograd still traces back to coeffs
        # via the non-leaf omega tensor.
        omega = get_connection_tensor(self.coeffs, self.basis_stack)
        if scale != 1.0:
            omega = omega * scale

        ode_func = ODEFunc(omega=omega)

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
        flags = []
        if self.signature_only:
            flags.append("signature_only=True")
        if not self.coeffs.requires_grad:
            flags.append("freeze_coeffs=True")
        if self.dtype != torch.float64:
            flags.append(f"dtype={self.dtype}")
        extras = (", " + ", ".join(flags)) if flags else ""
        return (
            f"T={self.T}, method={self.method}, "
            f"adjoint={self.adjoint}, n_params={self.coeffs.numel()}"
            f"{extras}"
        )
