"""
benchmarks.equivariance
-----------------------
Empirical SO(3,3) equivariance probe shared by figure_equivariance,
diagnose_equivariant, run_boost_ood and the top-tagging runner.

For an activation sigma, inputs x and group elements g it measures

    eps_sigma = E_b || sigma(g x) - g sigma(x) ||_2 / || sigma(x) ||_2
    eps_m     = E_b | <sigma(x), sigma(x)>_eta - <sigma(g x), sigma(g x)>_eta |

averaged over the g's. The flow conserves <v, v>_eta for every connection,
so eps_m only tests whether the input bound is invariant; eps_sigma is
the quantity that measures equivariance of the activation itself.
"""

from __future__ import annotations

import math

import torch

from so33.basis import DIM, ETA
from benchmarks.synthetic import random_so33_element


def sample_group_elements(n: int, rapidity: float, seed: int,
                          dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """(n, 6, 6) random SO(3,3) elements exp(sum_k c_k A_k), c_k ~ N(0, rapidity^2)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.stack([
        random_so33_element(rapidity=rapidity, generator=gen, dtype=dtype)
        for _ in range(n)
    ])


@torch.no_grad()
def activation_equivariance(act, x: torch.Tensor, gs) -> dict[str, float]:
    """eps_sigma (mean and max over the batch) and eps_m of one activation.

    Each statistic is averaged over the group elements in ``gs``. Non-finite
    outputs (a diverging flow) make the corresponding entries NaN, and
    ``nonfinite_frac`` reports how many outputs were non-finite.
    """
    was_training = act.training
    act.eval()
    eta = ETA.to(act.dtype)
    x = x.to(act.dtype)
    y = act(x)
    m2_y = (y * eta * y).sum(-1)
    act_errs, act_maxes, feat_errs, nonfinite = [], [], [], []
    for g in gs:
        g = g.to(act.dtype)
        ygx = act(x @ g.T)
        g_y = y @ g.T
        num = (ygx - g_y).norm(dim=-1)
        den = y.norm(dim=-1).clamp_min(1e-12)
        rel = num / den
        act_errs.append(rel.mean().item())
        act_maxes.append(rel.max().item())
        m2_gy = (ygx * eta * ygx).sum(-1)
        feat_errs.append((m2_y - m2_gy).abs().mean().item())
        nonfinite.append(1.0 - torch.isfinite(ygx).all(-1).double().mean().item())
    act.train(was_training)
    n = len(act_errs)
    return {
        "eps_sigma": sum(act_errs) / n,
        "eps_sigma_max": sum(act_maxes) / n,
        "eps_m": sum(feat_errs) / n,
        "nonfinite_frac": max(nonfinite + [1.0 - torch.isfinite(y).all(-1).double().mean().item()]),
    }


def lifted_constituents(X_set: torch.Tensor, max_particles: int | None = None,
                        seed: int = 0) -> torch.Tensor:
    """(n, 6) lifted real constituents of a (B, K, 5) set tensor (optionally subsampled)."""
    from benchmarks.models import _lift_4to6
    real = X_set[..., 4] > 0
    p6 = _lift_4to6(X_set[..., :4])[real]
    if max_particles is not None and len(p6) > max_particles:
        gen = torch.Generator().manual_seed(seed)
        p6 = p6[torch.randperm(len(p6), generator=gen)[:max_particles]]
    return p6


def model_equivariance(model, X_set: torch.Tensor, *, n_groups: int = 8,
                       rapidity: float = 0.5, seed: int = 0,
                       max_particles: int = 2000) -> dict | None:
    """Per-channel equivariance of an Architecture-B model at its trained connection.

    Returns None for models without per-particle SO33 activations
    (``model.acts`` acting on lifted constituents).
    """
    acts = getattr(model, "acts", None)
    if not acts or not hasattr(acts[0], "coeffs") or not hasattr(model, "eta"):
        return None
    x = lifted_constituents(X_set, max_particles=max_particles, seed=seed)
    gs = sample_group_elements(n_groups, rapidity, seed + 1, dtype=acts[0].dtype)
    per_channel = [activation_equivariance(a, x, gs) for a in acts]
    keys = per_channel[0].keys()
    mean = {k: (sum(c[k] for c in per_channel) / len(per_channel)) for k in keys}
    mean = {k: (v if math.isfinite(v) else float("nan")) for k, v in mean.items()}
    return {
        "group_rapidity": rapidity,
        "n_group_elements": n_groups,
        "n_inputs": int(len(x)),
        "per_channel": per_channel,
        **mean,
    }
