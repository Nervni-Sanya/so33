"""
activation.py
-------------
SO33Activation: parallel-transport activation layer in R^{3,3}.

Forward map
-----------
    F(x) = v(T)

where v(t) is the solution of the geodesic-like ODE:

    dv^mu / dt = -sum_{nu, lambda}  omega_eff^mu_{nu lambda}  v^nu  v^lambda
    v(0) = x

with the effective (norm-capped) connection

    omega_eff = omega * kappa / (kappa + ||omega||_F),      ||omega_eff||_F < kappa

omega = sum_k c_k B^(k) is the learnable connection and kappa = connection_cap
(default 1.0, i.e. omega / (1 + ||omega||_F)). The scale is an ordinary
differentiable function of the coefficients: gradients flow through it.

Trainable parameters
--------------------
    coeffs in R^K  —  scalar coefficients of the connection basis.
        K = 15 for full so(3,3), K = 6 for the so(3) ⊕ so(3) Euclidean
        ablation (signature_only=True).

Structure of the flow
---------------------
Write the vector field as  f(v) = -M(v) v  with  M(v) = sum_k c_k v^{j_k} A^(k),
an element of so(3,3). Two consequences:

  * <v, v>_eta is conserved for every c (M(v) is eta-antisymmetric). A
    readout of <sigma(x), sigma(x)>_eta therefore only sees the input bound,
    never the connection: conservation is not equivariance.
  * The flow is SO(3,3)-equivariant only at omega_eff = 0 (there is no
    non-zero invariant rank-3 tensor on R^{3,3}); any learned c != 0 breaks
    equivariance at O(||c||).

Numerical stability (indefinite metric)
---------------------------------------
Rotations (same-signature generators) preserve the Euclidean norm; boosts
do not. Explicitly  d||v||^2/dt = -4 sum_{boost (i,j)} c_ij v_i v_j^2,  hence

    d||v|| / dt  <=  alpha ||v||^2,    alpha = 4 / (3 sqrt 6) * ||omega_eff||_F

(the constant is attained), so ||v(t)|| <= r0 / (1 - alpha r0 t). The flow
cannot blow up on [0, T] for inputs with ||x||_2 <= r0 once
alpha r0 T < 1, and ||v(T)|| <= r0 / (1 - alpha r0 T). Because
||omega_eff||_F < kappa, choosing kappa = safe_connection_cap(r0, T, margin)
guarantees ||v(T)|| <= r0 / (1 - margin) for EVERY value of the
coefficients. With the default kappa = 1 and T = 0.3 only inputs with
||x||_2 <~ 6 are covered; massless (null) inputs are not bounded by any
SO(3,3)-equivariant input map, so the cap is the stabiliser of choice for
unbounded (e.g. lifted four-momentum) inputs.

Mitigations applied:
  1. Small init:   coeffs ~ N(0, 0.01)  →  omega ≈ 0  at start
  2. Norm cap:     omega_eff = kappa * omega / (kappa + ||omega||_F)  (differentiable)
  3. Frobenius regularisation:  L_reg = reg_coef * ||omega||_F^2  (add to loss)
  4. Gradient clipping recommended in outer training loop (max_norm = 1.0)

Backward pass
-------------
Approach A (adjoint=True, default):
    torchdiffeq.odeint_adjoint implements the continuous adjoint sensitivity
    method. Memory cost is O(1) w.r.t. trajectory length. With a(T) = dL/dv(T)
    the adjoint ODE is (omega_eff is not symmetric in its lower indices, so
    both index orders appear):

        da_rho / dt = sum_{mu, lambda}  a_mu ( omega_eff^mu_{rho lambda}
                                             + omega_eff^mu_{lambda rho} ) v^lambda

    and the parameter gradient accumulates as:

        dL/dc_k = -integral_0^T  sum_{mu,nu,lambda}
                      a_mu  (d omega_eff^mu_{nu lambda} / d c_k)  v^nu  v^lambda  dt

    where d omega_eff / d c_k includes the derivative of the norm cap.

Approach B (adjoint=False):
    Standard autograd through all ODE steps (the exact gradient of the
    discretised map). Stores full trajectory in memory.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .basis import get_basis_stack, get_connection_tensor, N_BASIS, ETA
from .ode_func import ODEFunc

# d||v||/dt <= BOOST_GROWTH_CONST * ||omega_eff||_F * ||v||^2  (sharp constant).
BOOST_GROWTH_CONST = 4.0 / (3.0 * math.sqrt(6.0))

# torchdiffeq solvers that integrate on a fixed grid. Without a step_size
# they take a single step over [0, T].
_FIXED_GRID_METHODS = frozenset({
    "euler", "midpoint", "heun2", "heun3", "rk4",
    "explicit_adams", "implicit_adams", "fixed_adams",
})


def safe_connection_cap(input_norm: float, T: float, margin: float = 0.5) -> float:
    """Connection cap kappa that makes the flow provably blow-up free.

    For every value of the coefficients, ||omega_eff||_F < kappa, so with
    alpha = BOOST_GROWTH_CONST * kappa the bound d||v||/dt <= alpha ||v||^2
    gives, for any input with ||x||_2 <= input_norm,

        ||v(t)||_2 <= ||x||_2 / (1 - margin)   on [0, T]

    when kappa = margin / (BOOST_GROWTH_CONST * input_norm * T).
    margin = 0.5 therefore bounds the output by twice the input norm.

    Parameters
    ----------
    input_norm : largest Euclidean norm of the inputs the activation will see.
    T          : integration horizon.
    margin     : alpha * input_norm * T, in (0, 1).
    """
    if not (0.0 < input_norm < math.inf and 0.0 < T < math.inf):
        raise ValueError(f"input_norm and T must be positive and finite; "
                         f"got {input_norm}, {T}")
    if not 0.0 < margin < 1.0:
        raise ValueError(f"margin must lie in (0, 1); got {margin}")
    return margin / (BOOST_GROWTH_CONST * input_norm * T)


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
    bound_input    : bool or str
                             Input-bounding mode for ODE stability:
                               False / "none"      -- no bound (default).
                               True  / "euclidean" -- divide by 1 + ||x||_2.
                                                      Bounds ||x|| < 1.
                                                      NOT SO(3,3) invariant
                                                      (Euclidean norm scales
                                                      under boosts).
                               "eta"               -- divide by
                                                      1 + sqrt(|x . eta . x|+eps).
                                                      An SO(3,3)-equivariant
                                                      rescaling (x . eta . x is
                                                      the indefinite-metric
                                                      invariant), but NOT a
                                                      bound: it is the identity
                                                      on the null cone, and no
                                                      non-zero equivariant map
                                                      is bounded.
                             The geodesic ODE under the indefinite metric can
                             diverge, which makes the adaptive solver fail
                             with ``underflow in dt`` on real-data inputs.
                             ``"euclidean"`` was the original fix; it trades
                             exact equivariance for stability and is what
                             breaks the so33_equivariant OOD generalisation
                             (the non-invariant denominator propagates into the
                             eta-invariant readout). For unbounded inputs use
                             ``connection_cap`` instead.
    max_input_norm : float or None
                             Soft cap rescaling only inputs with ||x||_2 above
                             it (not SO(3,3) invariant). Default None.
    solver_options : dict or None
                             Forwarded to torchdiffeq as ``options=``. For a
                             fixed-grid method (e.g. 'rk4') without a
                             ``step_size`` the step defaults to T / 10.
    connection_cap : float   kappa > 0 in omega_eff = kappa * omega / (kappa + ||omega||_F).
                             Default 1.0. Use ``safe_connection_cap`` to pick a
                             value with a no-blow-up guarantee for a given
                             input-norm range.

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
        bound_input: bool | str = False,
        max_input_norm: float | None = None,
        solver_options: dict | None = None,
        connection_cap: float = 1.0,
    ) -> None:
        super().__init__()

        # Normalise bound_input to a string mode for forward-pass dispatch.
        # Accept legacy bool values for backwards compatibility.
        if bound_input is False:
            bound_mode = "none"
        elif bound_input is True:
            bound_mode = "euclidean"
        elif bound_input in ("none", "euclidean", "eta"):
            bound_mode = bound_input
        else:
            raise ValueError(
                f"bound_input must be bool or one of "
                f"'none'/'euclidean'/'eta'; got {bound_input!r}"
            )
        if not 0.0 < connection_cap < math.inf:
            raise ValueError(f"connection_cap must be positive and finite; got {connection_cap}")

        solver_options = dict(solver_options) if solver_options else None
        if method in _FIXED_GRID_METHODS and not (
                solver_options and ("step_size" in solver_options
                                    or "grid_constructor" in solver_options)):
            # Without a step the fixed-grid solver takes ONE step over [0, T].
            solver_options = dict(solver_options or {}, step_size=T / 10.0)

        self.T              = T
        self.rtol           = rtol
        self.atol           = atol
        self.method         = method
        self.adjoint        = adjoint
        self.reg_coef       = reg_coef
        self.dtype          = dtype
        self.signature_only = signature_only
        self.bound_input    = bound_mode
        self.max_input_norm = max_input_norm
        self.solver_options = solver_options
        self.connection_cap = float(connection_cap)

        # ── Fixed basis (non-trainable, moves with .to(device)) ───────────────
        basis = get_basis_stack(dtype=dtype, signature_only=signature_only)
        self.register_buffer("basis_stack", basis)               # (K, 6, 6, 6)
        # eta vector for the "eta" input-bound mode (SO(3,3) invariant).
        self.register_buffer("eta_diag", ETA.to(dtype))
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

    def _adaptive_scale(self, omega: torch.Tensor) -> torch.Tensor:
        """kappa / (kappa + ||omega||_F), differentiable in the coefficients.

        torch.linalg.vector_norm has a zero (sub)gradient at omega = 0, so
        the scale stays differentiable when the connection vanishes.
        """
        kappa = self.connection_cap
        return kappa / (kappa + torch.linalg.vector_norm(omega))

    def effective_connection(self) -> torch.Tensor:
        """The (6, 6, 6) connection the ODE integrates, with the norm cap applied."""
        omega = get_connection_tensor(self.coeffs, self.basis_stack)
        return omega * self._adaptive_scale(omega)

    def coeff_norm(self) -> float:
        """||c||_2 of the raw connection coefficients."""
        return float(self.coeffs.detach().norm().item())

    def effective_connection_norm(self) -> float:
        """||omega_eff||_F (always < connection_cap)."""
        with torch.no_grad():
            return float(torch.linalg.vector_norm(self.effective_connection()).item())

    def regularization_loss(self) -> torch.Tensor:
        """Frobenius regularisation: L_reg = reg_coef * ||omega||_F^2.

        Acts on the raw (uncapped) connection. Add this term to the training
        loss to keep the coefficients small.
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
        if self.bound_input == "euclidean":
            # Bound ||x||_2 < 1 so ||v||² stays manageable through the ODE.
            # Without this the indefinite-metric trajectory can blow up
            # exponentially on inputs of moderate magnitude (e.g. standardised
            # tabular features through a Linear), causing the adaptive solver
            # to underflow dt to zero. Differentiable; trains end-to-end.
            # NOTE: the Euclidean norm is NOT SO(3,3)-invariant, so this mode
            # breaks equivariance under non-compact boosts (see "eta" mode).
            x_norm = x.norm(dim=-1, keepdim=True)
            x = x / (1.0 + x_norm)
        elif self.bound_input == "eta":
            # Eta-norm rescaling. x . eta . x is the SO(3,3) invariant of x,
            # so dividing by 1 + sqrt(|x . eta . x|) commutes with the group:
            # bound(g x) = g bound(x) for any g in SO(3,3). It does NOT bound
            # the input: for null vectors (x . eta . x = 0, e.g. massless
            # four-momenta) the divisor is 1. The 1e-12 eps keeps the
            # gradient finite at the origin.
            inner = (x * self.eta_diag * x).sum(dim=-1, keepdim=True)
            eta_norm = (inner.abs() + 1e-12).sqrt()
            x = x / (1.0 + eta_norm)
        elif self.max_input_norm is not None:
            # Soft norm cap: leave typical inputs untouched but rescale only
            # the heavy-tailed outliers whose ||x|| exceeds the cap. This
            # prevents the rare-large-input geodesic blow-up (stochastic NaN
            # on real HIGGS) without the signal-squashing of bound_input,
            # which divides *every* input. Differentiable.
            x_norm = x.norm(dim=-1, keepdim=True)
            factor = (self.max_input_norm / x_norm.clamp_min(self.max_input_norm))
            x = x * factor

        # Compute omega_eff once per forward — coeffs are constant during a
        # single ODE solve, so reusing the same tensor across solver steps
        # saves an einsum per step. Autograd (and the adjoint's vjp) traces
        # back to coeffs through the non-leaf omega_eff, norm cap included.
        omega = self.effective_connection()

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
                options=self.solver_options,
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
                options=self.solver_options,
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
        if self.connection_cap != 1.0:
            flags.append(f"connection_cap={self.connection_cap:.4g}")
        extras = (", " + ", ".join(flags)) if flags else ""
        return (
            f"T={self.T}, method={self.method}, "
            f"adjoint={self.adjoint}, n_params={self.coeffs.numel()}"
            f"{extras}"
        )
