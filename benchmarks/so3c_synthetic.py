"""
benchmarks.so3c_synthetic
-------------------------
Synthetic SO(3, C)-structured dataset: "electromagnetic" configurations
z = E + i B in C^3, labelled by a field invariant and scrambled by random
SO(3, C) transformations (rotations + complexified boosts).

Physics of the task
-------------------
The complex bilinear invariant of z = E + i B is

    z . z = (|E|^2 - |B|^2)  +  2 i (E . B)

whose real and imaginary parts are exactly the two classical invariants of
the electromagnetic field. Under the realification C^3 ~= R^6:

    Re(z . z) = v^T eta v      -- the SO(3,3) eta-invariant of `so33`
    Im(z . z) = 2 x . y        -- invariant ONLY under the SO(3,C) subgroup

So a label carried by Im(z . z) (mode="im", default) is structurally
invisible to any architecture whose invariant readout is built on eta alone:
that is the experiment separating the complexified prior from the parent
SO(3,3) one. mode="re" is the control where both priors see the label.

Generation
----------
1. Draw the class label and the banded invariant (with a margin between
   class bands); the other invariant component is a nuisance ~ N(0, spread).
2. Build a "rest-frame" field  z0 = sqrt(q) * n  with a random real unit
   direction n  (then z0 . z0 = q exactly).
3. Scramble with a random SO(3, C) element  Q = exp([rho + i beta]_x),
   rho ~ N(0, 1) (rotations, always in-distribution), ||beta|| drawn from
   ``boost_range`` — the OOD knob, playing the role of rapidity.
"""

from __future__ import annotations

from typing import Tuple

import torch

from so3c.algebra import complex_to_real, expm_so3c


def generate_em_invariant_dataset(
    n_samples: int = 10_000,
    seed: int = 7,
    boost_range: Tuple[float, float] = (0.0, 0.6),
    invariant_mode: str = "im",
    band_lo: Tuple[float, float] = (0.5, 1.5),
    band_hi: Tuple[float, float] = (2.5, 4.5),
    nuisance_spread: float = 1.0,
    rot_scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complexified 3-vectors labelled by a field invariant, boost-scrambled.

    Parameters
    ----------
    boost_range    : (lo, hi) — ||beta|| of the scrambling SO(3,C) element is
                     drawn uniformly from this band (the "rapidity" knob).
    invariant_mode : "im" — label carried by Im(z.z) = 2 E.B (invisible to
                     eta-only models); "re" — label carried by Re(z.z)
                     (control: visible to both priors).
    band_lo/band_hi: class-0 / class-1 bands for the labelled invariant
                     (disjoint -> clean margin, like the mass bands in
                     benchmarks.synthetic).

    Returns
    -------
    X          : (n_samples, 6) real tensor (Re z, Im z).
    y          : (n_samples,) long tensor (0 = low band, 1 = high band).
    boost_norm : (n_samples,) float tensor of applied ||beta||.
    """
    if invariant_mode not in ("im", "re"):
        raise ValueError(f"invariant_mode must be 'im' or 're'; got {invariant_mode!r}")
    gen = torch.Generator().manual_seed(seed)

    y = (torch.rand(n_samples, generator=gen) > 0.5).long()
    u_band = torch.rand(n_samples, generator=gen, dtype=dtype)
    lo = torch.tensor(band_lo, dtype=dtype)
    hi = torch.tensor(band_hi, dtype=dtype)
    banded = torch.where(
        y.bool(),
        hi[0] + u_band * (hi[1] - hi[0]),
        lo[0] + u_band * (lo[1] - lo[0]),
    )
    nuisance = torch.randn(n_samples, generator=gen, dtype=dtype) * nuisance_spread

    if invariant_mode == "im":
        q = torch.complex(nuisance, banded)
    else:
        q = torch.complex(banded, nuisance)

    # Rest-frame field with the prescribed invariant: z0 = sqrt(q) * n,
    # n a random real unit vector (real => n . n = 1 => z0 . z0 = q).
    n_dir = torch.randn(n_samples, 3, generator=gen, dtype=dtype)
    n_dir = n_dir / n_dir.norm(dim=-1, keepdim=True)
    z0 = torch.sqrt(q).unsqueeze(-1) * n_dir.to(q.dtype)

    # Random SO(3,C) scrambling: full random rotation, banded boost norm.
    rho = torch.randn(n_samples, 3, generator=gen, dtype=dtype) * rot_scale
    b_dir = torch.randn(n_samples, 3, generator=gen, dtype=dtype)
    b_dir = b_dir / b_dir.norm(dim=-1, keepdim=True)
    b_lo, b_hi = boost_range
    boost_norm = torch.rand(n_samples, generator=gen, dtype=dtype) * (b_hi - b_lo) + b_lo
    beta = b_dir * boost_norm.unsqueeze(-1)

    Q = expm_so3c(torch.complex(rho, beta))              # (B, 3, 3)
    z = (Q @ z0.unsqueeze(-1)).squeeze(-1)               # (B, 3)
    return complex_to_real(z), y, boost_norm
