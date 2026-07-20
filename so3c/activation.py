"""
activation.py
-------------
SO3CActivation: geodesic-flow activation on the complexified space C^3.

Forward map
-----------
    F(x) = z(T),   dz/dt = -[a]_x z,   z(0) = beta(x)

where the connection a in C^3 ~= so(3, C) is either

    mode="static"  : 6 learnable scalars  a = rho + i beta      (the direct
                     complexified analogue of SO33Activation's 15 coefficients)
    mode="dynamic" : a = a(s(z)) predicted by HermitianMetric from the
                     SO(3, C) invariants s = (Re z.z, Im z.z)

Key structural property
-----------------------
[a]_x is complex antisymmetric, hence z^T [a]_x z = 0 and the flow conserves
the complex quadratic invariant z . z EXACTLY (both real and imaginary
parts). In dynamic mode a(s) is therefore constant along each trajectory, and
the flow admits the closed-form solution

    z(T) = exp(-T [a(s_0)]_x) z_0        (complex Rodrigues formula)

method="exact" evaluates this directly: no ODE solver, machine-precision
conservation, cheap exact gradients. method="dopri5"/"rk4" integrate the same
flow with torchdiffeq (recomputing a(s(z)) at every step in dynamic mode) —
used for cross-validation and as the template for the multi-particle case
(interaction.py), where no closed form exists.

Input bounding (carries over the central so33 lesson)
-----------------------------------------------------
    "none"      -- identity.
    "euclidean" -- x / (1 + ||x||_2). NOT invariant: breaks the group
                   structure under boosts (ablation mode).
    "bilinear"  -- x / (1 + |z . z|^(1/2)). SO(3, C)-invariant by
                   construction — the complexified analogue of "eta".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .algebra import (
    DIM_R,
    bilinear_invariant,
    complex_to_real,
    cross_matrix,
    expm_so3c,
    real_to_complex,
)
from .metric import HermitianMetric


class _GeodesicODE(nn.Module):
    """RHS f(t, v) = realify(-[a(v)]_x z(v)) for torchdiffeq integration."""

    def __init__(self, connection_fn) -> None:
        super().__init__()
        self._connection_fn = connection_fn

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        z = real_to_complex(v)
        a = self._connection_fn(v)
        dz = -(cross_matrix(a) @ z.unsqueeze(-1)).squeeze(-1)
        return complex_to_real(dz)


class SO3CActivation(nn.Module):
    """Geodesic-flow activation on C^3 with an so(3, C) connection.

    Parameters
    ----------
    T                : float  integration horizon (default 1.0)
    mode             : str    "dynamic" (HermitianMetric, default) | "static"
    method           : str    "exact" (closed form, default) | any torchdiffeq
                              method ("dopri5", "rk4", ...)
    rtol, atol       : float  solver tolerances (solver methods only)
    adjoint          : bool   adjoint backprop for solver methods
    reg_coef         : float  coefficient for regularization_loss()
    dtype            : torch.dtype  real dtype (complex derived from it)
    bound_input      : str    "none" | "euclidean" | "bilinear" (see module doc)
    hidden           : int    hidden width of the dynamic-metric MLP
    scale_connection : bool   soft-normalise a -> a / (1 + ||a||): bounds the
                              boost rapidity accumulated over the horizon
    solver_options   : dict | None   forwarded to torchdiffeq

    Input  : (B, 6) real tensor or (B, 3) complex tensor.
    Output : same layout as the input.
    """

    def __init__(
        self,
        T: float = 1.0,
        mode: str = "dynamic",
        method: str = "exact",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        adjoint: bool = False,
        reg_coef: float = 1e-3,
        dtype: torch.dtype = torch.float64,
        bound_input: str = "bilinear",
        hidden: int = 16,
        scale_connection: bool = True,
        solver_options: dict | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("static", "dynamic"):
            raise ValueError(f"mode must be 'static' or 'dynamic'; got {mode!r}")
        if bound_input not in ("none", "euclidean", "bilinear"):
            raise ValueError(
                f"bound_input must be 'none'/'euclidean'/'bilinear'; got {bound_input!r}"
            )
        self.T = T
        self.mode = mode
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.reg_coef = reg_coef
        self.dtype = dtype
        self.bound_input = bound_input
        self.scale_connection = scale_connection
        self.solver_options = dict(solver_options) if solver_options else None

        if mode == "static":
            # Small init keeps the flow near-identity at the start of training.
            self.coeffs = nn.Parameter(torch.randn(6, dtype=dtype) * 0.01)
            self.metric = None
        else:
            self.coeffs = None
            self.metric = HermitianMetric(hidden=hidden, dtype=dtype)

        self.register_buffer("t_span", torch.tensor([0.0, T], dtype=dtype))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bound(self, v: torch.Tensor) -> torch.Tensor:
        if self.bound_input == "euclidean":
            return v / (1.0 + v.norm(dim=-1, keepdim=True))
        if self.bound_input == "bilinear":
            q = bilinear_invariant(v)
            mod = (q.real.pow(2) + q.imag.pow(2) + 1e-12).sqrt()   # |z . z|
            return v / (1.0 + mod.sqrt().unsqueeze(-1))            # |z . z|^(1/2)
        return v

    def _connection(self, v: torch.Tensor) -> torch.Tensor:
        """(…, 6) real state -> (…, 3) complex connection a (scaled)."""
        if self.mode == "static":
            a = torch.complex(self.coeffs[:3], self.coeffs[3:])
            a = a.expand(v.shape[:-1] + (3,))
        else:
            a = self.metric(v)
        if self.scale_connection:
            norm = a.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
            a = a / (1.0 + norm)
        return a

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_complex = torch.is_complex(x)
        v = complex_to_real(x) if was_complex else x.to(self.dtype)
        if v.shape[-1] != DIM_R:
            raise ValueError(f"expected last dim {DIM_R} (real) or 3 (complex)")

        v = self._bound(v)

        if self.method == "exact":
            a = self._connection(v)
            Q = expm_so3c(a, t=-self.T)                 # exp(-T [a]_x)
            z = real_to_complex(v)
            z_out = (Q @ z.unsqueeze(-1)).squeeze(-1)
            out = complex_to_real(z_out)
        else:
            ode = _GeodesicODE(self._connection)
            solve = odeint_adjoint if self.adjoint else odeint
            kwargs = dict(
                rtol=self.rtol, atol=self.atol,
                method=self.method, options=self.solver_options,
            )
            if self.adjoint:
                kwargs["adjoint_params"] = tuple(self.parameters())
            traj = solve(ode, v, self.t_span.to(v.dtype), **kwargs)
            out = traj[-1]

        return real_to_complex(out) if was_complex else out

    def regularization_loss(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Penalty bounding the connection magnitude (add to training loss).

        static  : reg_coef * ||coeffs||^2
        dynamic : reg_coef * mean ||a(s(x))||^2 when a batch x is given,
                  else reg_coef * output-layer weight penalty (uniform bound).
        """
        if self.mode == "static":
            return self.reg_coef * self.coeffs.pow(2).sum()
        if x is not None:
            v = complex_to_real(x) if torch.is_complex(x) else x.to(self.dtype)
            a = self.metric(self._bound(v))
            return self.reg_coef * a.abs().pow(2).sum(dim=-1).mean()
        return self.reg_coef * self.metric.weight_penalty()

    @torch.no_grad()
    def invariant_drift(self, x: torch.Tensor) -> torch.Tensor:
        """|z(T).z(T) - z(0).z(0)| per sample — conservation diagnostic.

        Machine precision for method="exact"; solver-tolerance-limited
        otherwise. Compared against the bounded initial state, i.e. measures
        the flow itself, not the bounding map.
        """
        was_complex = torch.is_complex(x)
        v = complex_to_real(x) if was_complex else x.to(self.dtype)
        v = self._bound(v)
        q_in = bilinear_invariant(v)
        saved, self.bound_input = self.bound_input, "none"
        try:
            out = self.forward(v)
        finally:
            self.bound_input = saved
        q_out = bilinear_invariant(out)
        return (q_out - q_in).abs()

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, mode={self.mode}, method={self.method}, "
            f"bound_input={self.bound_input}"
        )
