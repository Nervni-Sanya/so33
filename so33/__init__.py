"""
so33 — Parallel Transport Activation in Pseudo-Euclidean Space R^{3,3}.

Public API
----------
SO33Activation        : ODE-based activation layer (15 trainable params)
SO33Network           : Linear -> SO33Activation -> Linear convenience wrapper
build_so33_basis      : construct the 15 canonical (6,6,6) basis tensors
get_connection_tensor : reconstruct omega from learnable coefficients
safe_connection_cap   : connection cap with a no-blow-up guarantee for a given input-norm range
"""

from .basis import build_so33_basis, get_basis_stack, get_connection_tensor
from .ode_func import ODEFunc
from .activation import SO33Activation, safe_connection_cap
from .network import SO33Network, BottleneckClassifier

__all__ = [
    "SO33Activation",
    "SO33Network",
    "BottleneckClassifier",
    "build_so33_basis",
    "get_basis_stack",
    "get_connection_tensor",
    "safe_connection_cap",
    "ODEFunc",
]

__version__ = "1.0.0-beta.3"
__author__ = "Panchenko Alexander"
