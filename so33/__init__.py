"""
so33 — Parallel Transport Activation in Pseudo-Euclidean Space R^{3,3}.

Public API
----------
SO33Activation        : ODE-based activation layer (15 trainable params)
SO33Network           : Linear -> SO33Activation -> Linear convenience wrapper
build_so33_basis      : construct the 15 canonical (6,6,6) basis tensors
get_connection_tensor : reconstruct omega from learnable coefficients
"""

from .basis import build_so33_basis, get_basis_stack, get_connection_tensor
from .ode_func import ODEFunc
from .activation import SO33Activation
from .network import SO33Network

__all__ = [
    "SO33Activation",
    "SO33Network",
    "build_so33_basis",
    "get_basis_stack",
    "get_connection_tensor",
    "ODEFunc",
]

__version__ = "1.0.0-beta.3"
__author__ = "Panchenko Alexander"
