"""
so3c — Complexified SO(3): geodesic flows on C^3 with a dynamic Hermitian
metric.

Complexifying so(3) yields the Lorentz algebra as a real Lie algebra:

    so(3) (+) i so(3) = so(3, C) ~= so(3, 1) ~= sl(2, C)_R

so this package is simultaneously (a) the complexification of the rotation
group and (b) an exactly-Lorentz-structured flow layer. The realification
C^3 ~= R^6 embeds SO(3, C) into the SO(3, 3) of the parent `so33` package as
the subgroup commuting with the complex structure, and the eta-invariant of
`so33` reappears as Re(z . z).

Public API
----------
SO3CActivation  : single-state geodesic-flow activation; closed-form "exact"
                  mode (complex Rodrigues formula) or torchdiffeq solvers.
SO3CInteraction : exactly equivariant multi-particle ODE layer (adaptive
                  solver) — the intended LGEB drop-in for LorentzNet.
HermitianMetric : dynamic Hermitian metric g(s) = I + i eps(s) parameterised
                  by SO(3, C) invariants; supplies the connection a(s).
algebra         : generators, invariants, expm_so3c, realification utilities.
"""

from .algebra import (
    DIM_C,
    DIM_R,
    ETA,
    N_GEN,
    bilinear_invariant,
    complex_bilinear,
    complex_matrix_to_real,
    complex_structure,
    complex_to_real,
    cross_matrix,
    expm_so3c,
    invariant_features,
    random_group_element,
    real_to_complex,
    so3_generators,
    so3c_generator_stack,
)
from .metric import HermitianMetric
from .activation import SO3CActivation
from .interaction import SO3CInteraction

__all__ = [
    "DIM_C",
    "DIM_R",
    "ETA",
    "N_GEN",
    "bilinear_invariant",
    "complex_bilinear",
    "complex_matrix_to_real",
    "complex_structure",
    "complex_to_real",
    "cross_matrix",
    "expm_so3c",
    "invariant_features",
    "random_group_element",
    "real_to_complex",
    "so3_generators",
    "so3c_generator_stack",
    "HermitianMetric",
    "SO3CActivation",
    "SO3CInteraction",
]

__version__ = "0.1.0-alpha.1"
__author__ = "Panchenko Alexander"
