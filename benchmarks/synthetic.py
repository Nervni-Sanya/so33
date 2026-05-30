"""
benchmarks.synthetic
--------------------
Synthetic Lorentz-structured datasets and SO(3,3) transformations used
by the geometric test battery.

Building blocks
---------------
- ``generate_causal_dataset(n_samples)``: scale-up of the existing
  generator from ``tests/test_causal_classification.py``. Two classes:
  Lorentz boost (cross-signature) vs spatial rotation (same-signature).
- ``random_so33_element(rapidity, generator)``: sample a finite SO(3,3)
  group element by exponentiating a random linear combination of the
  15 so(3,3) generators. ``rapidity`` controls the magnitude of the
  random coefficients; small values give near-identity transforms.
- ``transform_inputs(X, g)``: apply an SO(3,3) element to a (B, 6)
  batch of inputs.
- ``boost_split(X, y, threshold)``: split a dataset by per-sample
  Lorentz norm into low-/high-rapidity subsets, used for the OOD
  generalization experiment.
"""

from __future__ import annotations

from typing import Tuple

import torch
import numpy as np
from torchdiffeq import odeint

from so33.basis import DIM, ETA, N_BASIS


# ─────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────

def _build_class_omegas(dtype: torch.dtype = torch.float64):
    """Return (omega_boost, omega_rot) ground-truth connection tensors."""
    eta = ETA.to(dtype)

    # Lorentz boost (0, 3): eta[0]=+1, eta[3]=-1 -> cross-signature
    A_boost = torch.zeros(DIM, DIM, dtype=dtype)
    A_boost[0, 3] =  1.0
    A_boost[3, 0] = -(eta[0] / eta[3])
    omega_boost = torch.zeros(DIM, DIM, DIM, dtype=dtype)
    omega_boost[:, :, 3] = A_boost

    # Spatial rotation (0, 1): eta[0]=eta[1]=+1 -> same-signature
    A_rot = torch.zeros(DIM, DIM, dtype=dtype)
    A_rot[0, 1] =  1.0
    A_rot[1, 0] = -(eta[0] / eta[1])
    omega_rot = torch.zeros(DIM, DIM, DIM, dtype=dtype)
    omega_rot[:, :, 1] = A_rot

    return omega_boost, omega_rot


def generate_causal_dataset(
    n_samples: int = 10_000,
    seed: int = 7,
    init_scale: float = 0.3,
    t_span_end: float = 1.0,
    step_size: float = 0.1,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate (X, y) for the synthetic causal classification task.

    Same physics as ``tests/test_causal_classification.py`` but parameterised
    so we can scale to 10k+ samples without copying the test code.

    Class 0 — Lorentz boost connection ω_{03} (cross-signature, timelike)
    Class 1 — Spatial rotation connection ω_{01} (same-signature, spacelike)

    Each sample's terminal state v(T) is the integrated geodesic from a
    random initial v(0) ~ N(0, init_scale) under the class-specific ω.

    Returns
    -------
    X : (n_samples, 6) tensor of terminal states, in dtype.
    y : (n_samples,) long tensor of class labels (0 or 1).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    half = n_samples // 2
    omega_boost, omega_rot = _build_class_omegas(dtype=dtype)
    t_span = torch.tensor([0.0, t_span_end], dtype=dtype)

    samples = []
    for cls_idx, omega_cls in enumerate([omega_boost, omega_rot]):
        for _ in range(half):
            v0 = torch.randn(1, DIM, dtype=dtype) * init_scale

            def rhs(t, v, w=omega_cls):
                return -torch.einsum("mnl, bn, bl -> bm", w, v, v)

            with torch.no_grad():
                traj = odeint(rhs, v0, t_span, method="rk4",
                              options={"step_size": step_size})

            samples.append((traj[-1].squeeze(0), cls_idx))

    perm = torch.randperm(len(samples))
    X = torch.stack([samples[i][0] for i in perm])
    y = torch.tensor([samples[i][1] for i in perm], dtype=torch.long)
    return X, y


def _boost_matrix(rapidity: torch.Tensor, space_axis: int = 0,
                  time_axis: int = 3, dtype: torch.dtype = torch.float64):
    """Batch of SO(3,3) boost matrices mixing one +axis and one -axis.

    For metric diag(+,+,+,-,-,-), a 'boost' between a spacelike axis i and
    a timelike axis j is the hyperbolic rotation
        [[cosh φ, sinh φ], [sinh φ, cosh φ]]
    embedded in the 6x6 identity. These are genuine SO(3,3) group elements
    (they preserve eta), so applying them is an exact symmetry of the model
    family — the property the OOD experiment probes.

    Parameters
    ----------
    rapidity : (B,) tensor of φ values.

    Returns
    -------
    (B, 6, 6) tensor of boost matrices.
    """
    B = rapidity.shape[0]
    M = torch.eye(DIM, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    ch, sh = torch.cosh(rapidity), torch.sinh(rapidity)
    M[:, space_axis, space_axis] = ch
    M[:, time_axis,  time_axis]  = ch
    M[:, space_axis, time_axis]  = sh
    M[:, time_axis,  space_axis] = sh
    return M


def generate_boost_invariant_dataset(
    n_samples: int = 10_000,
    seed: int = 7,
    rapidity_range: Tuple[float, float] = (0.0, 0.8),
    mass_lo: float = 1.0,
    mass_hi: float = 3.0,
    p_spread: float = 0.5,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """4-vectors labelled by a Lorentz invariant, scrambled by random boosts.

    Each sample is a particle 4-vector embedded in R^{3,3} as
    (px, py, pz, E, 0, 0): spatial momentum on the +axes 0..2, energy on
    the -axis 3 (matching the metric used elsewhere). The class label is
    the invariant mass band (low vs high), which is **boost-invariant** —
    but every sample is hit with a random SO(3,3) boost of rapidity drawn
    from ``rapidity_range``, so the raw components of the two classes
    overlap heavily. Only a model that can recover the invariant separates
    them cleanly, and a Lorentz-equivariant model should generalise across
    rapidity regimes it never saw in training (the OOD claim).

    Returns
    -------
    X        : (n_samples, 6) float tensor.
    y        : (n_samples,) long tensor (0 = low mass, 1 = high mass).
    rapidity : (n_samples,) float tensor of the applied boost rapidities.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Class by invariant mass.
    y = (torch.rand(n_samples) > 0.5).long()
    m = torch.where(y.bool(), torch.tensor(mass_hi), torch.tensor(mass_lo)).to(dtype)

    # Rest-frame-ish 4-vector: random 3-momentum, energy from mass shell.
    p = torch.randn(n_samples, 3, dtype=dtype) * p_spread
    E = torch.sqrt(m * m + p.pow(2).sum(-1))
    v6 = torch.zeros(n_samples, DIM, dtype=dtype)
    v6[:, 0:3] = p
    v6[:, 3]   = E

    # Random boost rapidity in range, random spatial axis to boost along.
    lo, hi = rapidity_range
    rapidity = torch.rand(n_samples, dtype=dtype) * (hi - lo) + lo
    axis = torch.randint(0, 3, (n_samples,))
    X = v6.clone()
    for a in range(3):
        sel = axis == a
        if sel.any():
            Mb = _boost_matrix(rapidity[sel], space_axis=a, time_axis=3, dtype=dtype)
            X[sel] = torch.einsum("bij,bj->bi", Mb, v6[sel])

    return X, y, rapidity


# ─────────────────────────────────────────────────────────────────────────
# SO(3,3) group elements (for equivariance tests and data augmentation)
# ─────────────────────────────────────────────────────────────────────────

def _so33_generator_matrix(coeffs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Linearly combine the 15 so(3,3) generators into a (6, 6) matrix.

    A_k = e_i e_j^T - (eta_i / eta_j) e_j e_i^T  (one per ordered pair i<j)
    A   = sum_k coeffs[k] * A_k
    """
    if coeffs.numel() != N_BASIS:
        raise ValueError(f"coeffs must have length {N_BASIS}, got {coeffs.numel()}")
    eta = ETA.to(dtype)
    A   = torch.zeros(DIM, DIM, dtype=dtype)
    k   = 0
    for i in range(DIM):
        for j in range(i + 1, DIM):
            c = coeffs[k].to(dtype)
            A[i, j] += c
            A[j, i] += c * (-(eta[i] / eta[j]))
            k += 1
    return A


def random_so33_element(
    rapidity: float = 0.3,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample a random SO(3,3) group element via matrix exponential.

    g = exp(A) where A = sum_k c_k * A_k  with  c_k ~ N(0, rapidity^2).

    The matrix-exp ensures g satisfies g^T eta g = eta exactly (up to
    floating-point error). ``rapidity`` controls how far from identity
    g is: small values give small boosts/rotations, large values push
    deep into the indefinite-metric regime.

    Returns
    -------
    g : (6, 6) tensor in SO(3,3).
    """
    coeffs = torch.empty(N_BASIS, dtype=dtype)
    if generator is None:
        coeffs.normal_(mean=0.0, std=rapidity)
    else:
        coeffs.normal_(mean=0.0, std=rapidity, generator=generator)
    A = _so33_generator_matrix(coeffs, dtype=dtype)
    return torch.linalg.matrix_exp(A)


def is_so33_element(g: torch.Tensor, atol: float = 1e-6) -> bool:
    """Check whether g satisfies the SO(3,3) condition g^T eta g = eta."""
    eta = torch.diag(ETA.to(g.dtype))
    lhs = g.T @ eta @ g
    return torch.allclose(lhs, eta, atol=atol)


def transform_inputs(X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Apply an SO(3,3) element g to a batch of inputs X: (B, 6) -> (B, 6).

    Acts as left-multiplication: X' = X @ g^T  (so each row x is mapped
    to g x).
    """
    return X @ g.T.to(X.dtype)


# ─────────────────────────────────────────────────────────────────────────
# Splits used for the OOD experiment
# ─────────────────────────────────────────────────────────────────────────

def lorentz_norm_squared(X: torch.Tensor) -> torch.Tensor:
    """Per-sample x^T eta x  for X of shape (B, 6).

    Negative values indicate timelike vectors; positive, spacelike. The
    magnitude correlates with how far the ODE was integrated under a
    Lorentz boost connection.
    """
    eta = ETA.to(X.dtype).unsqueeze(0)  # (1, 6)
    return (X * eta * X).sum(dim=-1)


def boost_split(
    X: torch.Tensor,
    y: torch.Tensor,
    threshold: float | None = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Split (X, y) into low-/high-rapidity subsets by |Lorentz norm|.

    The OOD experiment uses ``low`` for training and ``high`` for held-out
    evaluation: trained at small boosts, tested at large.

    If ``threshold`` is None, splits at the median absolute norm.

    Returns
    -------
    (X_low, y_low), (X_high, y_high)
    """
    norm = lorentz_norm_squared(X).abs()
    if threshold is None:
        threshold = norm.median().item()
    low_mask  = norm <= threshold
    high_mask = ~low_mask
    return (X[low_mask], y[low_mask]), (X[high_mask], y[high_mask])
