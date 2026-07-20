"""
interaction.py
--------------
SO3CInteraction: an exactly SO(3, C)-equivariant multi-particle ODE layer.

This is the building block intended to replace / augment the nonlinearity in
LorentzNet's LGEB blocks (recall SO(3, C) ~= SO+(3, 1) as a real Lie group):
a learnable geodesic flow on the particle set instead of a pointwise
activation.

Vector field
------------
For particles z_1..z_N in C^3 the flow is driven by state-dependent algebra
elements built from bivectors of the particles themselves:

    dz_a/dt = -A_a z ,
    A_a = (1/n) sum_b phi_theta(s_aa, s_bb, s_ab) (z_a z_b^T - z_b z_a^T)

where s_ab = z_a . z_b is the complex bilinear invariant and phi_theta is a
small MLP with complex output (2 real outputs). Contracting the bivector:

    dz_a/dt = -(1/n) sum_b phi_ab (s_ab z_a - s_aa z_b)

so no matrices are materialised — the cost is O(N^2) pairwise products.

Structural guarantees (by construction, not empirically)
--------------------------------------------------------
1. Equivariance: bivectors transform in the adjoint, (Q z_a)(Q z_b)^T - ...
   = Q (z_a z_b^T - z_b z_a^T) Q^T with Q^T = Q^{-1} in SO(3, C), and the
   coefficients phi depend on invariants only. Hence f(Q z) = Q f(z) exactly
   for the continuous flow; discretisation error is solver-tolerance-limited.
2. Conservation: A_a is complex antisymmetric (an so(3, C) element), so each
   particle's own invariant z_a . z_a is conserved exactly. Pairwise products
   z_a . z_b are NOT conserved — that is the feature mixing.
3. Identity at init: the MLP output layer starts at zero, so the layer is a
   drop-in no-op before training.

Unlike the single-particle activation, A_a genuinely varies along the
trajectory (through the s_ab), so there is no closed form — this is where
the adaptive solver earns its keep.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .algebra import DIM_R, bilinear_invariant, complex_to_real, real_to_complex


class _InteractionODE(nn.Module):
    """RHS closure holding the coupling MLP and the padding mask."""

    def __init__(self, phi: nn.Module, mask: torch.Tensor) -> None:
        super().__init__()
        self.phi = phi
        self.mask = mask                     # (B, N) real, 1 = real particle

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        z = real_to_complex(v)                                   # (B, N, 3)
        S = z @ z.transpose(-1, -2)                              # (B, N, N)  s_ab
        s_diag = torch.diagonal(S, dim1=-2, dim2=-1)             # (B, N)     s_aa

        B, N = s_diag.shape
        feats = torch.stack(
            [
                s_diag.real.unsqueeze(-1).expand(B, N, N),       # s_aa
                s_diag.imag.unsqueeze(-1).expand(B, N, N),
                s_diag.real.unsqueeze(-2).expand(B, N, N),       # s_bb
                s_diag.imag.unsqueeze(-2).expand(B, N, N),
                S.real,                                          # s_ab
                S.imag,
            ],
            dim=-1,
        )
        phi_out = self.phi(torch.asinh(feats))                   # (B, N, N, 2)
        phi = torch.complex(phi_out[..., 0], phi_out[..., 1])    # (B, N, N)

        pair_mask = self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2)
        phi = phi * pair_mask
        n = self.mask.sum(dim=-1).clamp(min=1.0)                 # (B,)

        coef = (phi * S).sum(dim=-1)                             # (B, N)  sum_b phi_ab s_ab
        mixed = phi @ z                                          # (B, N, 3)  sum_b phi_ab z_b
        dz = -(coef.unsqueeze(-1) * z - s_diag.unsqueeze(-1) * mixed)
        dz = dz / n[:, None, None]
        dz = dz * self.mask.unsqueeze(-1)
        return complex_to_real(dz)


class SO3CInteraction(nn.Module):
    """Equivariant geodesic message-passing flow on a particle set.

    Parameters
    ----------
    hidden   : int    hidden width of the coupling MLP phi (default 16)
    T        : float  integration horizon (default 1.0)
    method   : str    torchdiffeq method (default "dopri5", adaptive)
    rtol     : float  relative tolerance (default 1e-6)
    atol     : float  absolute tolerance (default 1e-8)
    adjoint  : bool   adjoint backprop (default False: direct autograd)
    reg_coef : float  coefficient for regularization_loss()
    dtype    : torch.dtype  real parameter dtype (default float64)

    forward(x, mask=None)
        x    : (B, N, 6) real or (B, N, 3) complex particle states
        mask : (B, N) optional, 1 for real particles, 0 for padding;
               padded entries are returned unchanged and do not influence
               real particles.
    Output : same layout as the input.
    """

    def __init__(
        self,
        hidden: int = 16,
        T: float = 1.0,
        method: str = "dopri5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        adjoint: bool = False,
        reg_coef: float = 1e-3,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.T = T
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.reg_coef = reg_coef
        self.dtype = dtype

        self.phi = nn.Sequential(
            nn.Linear(6, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 2, dtype=dtype),
        )
        # Zero-init output layer: phi = 0 -> identity map at start.
        nn.init.zeros_(self.phi[-1].weight)
        nn.init.zeros_(self.phi[-1].bias)

        self.register_buffer("t_span", torch.tensor([0.0, T], dtype=dtype))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        was_complex = torch.is_complex(x)
        v = complex_to_real(x) if was_complex else x.to(self.dtype)
        if v.shape[-1] != DIM_R:
            raise ValueError(f"expected last dim {DIM_R} (real) or 3 (complex)")

        if mask is None:
            mask = torch.ones(v.shape[:-1], dtype=v.dtype, device=v.device)
        else:
            mask = mask.to(dtype=v.dtype, device=v.device)

        ode = _InteractionODE(self.phi, mask)
        solve = odeint_adjoint if self.adjoint else odeint
        kwargs = dict(rtol=self.rtol, atol=self.atol, method=self.method)
        if self.adjoint:
            kwargs["adjoint_params"] = tuple(self.parameters())
        traj = solve(ode, v, self.t_span.to(v.dtype), **kwargs)
        out = traj[-1]

        return real_to_complex(out) if was_complex else out

    def regularization_loss(self) -> torch.Tensor:
        """Output-layer weight penalty — uniformly bounds |phi| (tanh hidden)."""
        last: nn.Linear = self.phi[-1]
        return self.reg_coef * (last.weight.pow(2).sum() + last.bias.pow(2).sum())

    @torch.no_grad()
    def invariant_drift(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-particle |z_a(T).z_a(T) - z_a(0).z_a(0)| — conservation check."""
        was_complex = torch.is_complex(x)
        v = complex_to_real(x) if was_complex else x.to(self.dtype)
        q_in = bilinear_invariant(v)
        out = self.forward(v, mask)
        q_out = bilinear_invariant(out)
        return (q_out - q_in).abs()
