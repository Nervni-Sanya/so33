"""
Test 7 — dtype param, signature-only ablation, frozen-Γ ablation,
         and BottleneckClassifier.

Verifies:
  • build_so33_basis(dtype=float32) returns float32 tensors and still
    satisfies the metric-connection antisymmetry (within tolerance).
  • build_so33_basis(signature_only=True) returns exactly 6 generators
    (3 from {0,1,2}^2 and 3 from {3,4,5}^2), all same-signature pairs.
  • SO33Activation(dtype=float32) trains for one Adam step without NaN
    and gradient flows to coeffs.
  • SO33Activation(signature_only=True) has 6 trainable coefficients and
    integrates correctly.
  • SO33Activation(freeze_coeffs=True): coeffs.requires_grad is False,
    its .grad stays None after backward, and the surrounding Linear
    layers still receive gradients (so the rest of the model still trains).
  • BottleneckClassifier with nn.ReLU runs end-to-end.
  • BottleneckClassifier with SO33Activation matches SO33Network output
    for the same weights and seed.

Run:
    python -m pytest tests/test_ablations_and_dtype.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from so33.basis import (
    DIM,
    ETA,
    N_BASIS,
    build_so33_basis,
    get_basis_stack,
    get_connection_tensor,
)
from so33.activation import SO33Activation
from so33.network import BottleneckClassifier, SO33Network


# ────────────────────────────────────────────────────────────────────────
# basis.py
# ────────────────────────────────────────────────────────────────────────

def test_basis_float32_metric_connection() -> None:
    """float32 basis still satisfies eta_mu omega[mu,nu,l] + eta_nu omega[nu,mu,l] = 0."""
    basis = build_so33_basis(dtype=torch.float32)
    eta   = ETA.to(torch.float32)

    assert len(basis) == N_BASIS
    for omega_k in basis:
        assert omega_k.dtype == torch.float32
        for mu in range(DIM):
            for nu in range(DIM):
                for lam in range(DIM):
                    val = (eta[mu] * omega_k[mu, nu, lam]
                         + eta[nu] * omega_k[nu, mu, lam]).item()
                    assert abs(val) < 1e-6, f"violation at {(mu,nu,lam)}: {val}"


def test_basis_signature_only_count_and_pairs() -> None:
    """signature_only=True returns exactly 6 same-signature generators."""
    basis = build_so33_basis(signature_only=True)
    assert len(basis) == 6

    # Each basis tensor was lifted as omega[:, :, j]; from indicator lift
    # construction the active j is the last index. Recover (i, j) pairs and
    # check they are all same-signature.
    eta = ETA
    spacelike_pairs = {(0, 1), (0, 2), (1, 2)}
    timelike_pairs  = {(3, 4), (3, 5), (4, 5)}
    expected_pairs  = spacelike_pairs | timelike_pairs

    found_pairs = set()
    for omega_k in basis:
        nonzero = (omega_k != 0).nonzero()
        # lambda index is the third coordinate; pick from any nonzero entry
        lam = nonzero[0, 2].item()
        # the matrix A = omega_k[:, :, lam] has +1 at (i, j) and ±1 at (j, i)
        A = omega_k[:, :, lam]
        i, j = sorted([int(x) for x in (A != 0).nonzero()[0].tolist()])
        found_pairs.add((i, j))
        assert eta[i] == eta[j], f"cross-signature pair leaked: ({i}, {j})"

    assert found_pairs == expected_pairs, f"unexpected pairs: {found_pairs}"


def test_get_basis_stack_caches_per_variant() -> None:
    """Cached stacks have correct shape for each (dtype, signature_only)."""
    full64 = get_basis_stack(dtype=torch.float64, signature_only=False)
    full32 = get_basis_stack(dtype=torch.float32, signature_only=False)
    sig64  = get_basis_stack(dtype=torch.float64, signature_only=True)

    assert full64.shape == (N_BASIS, DIM, DIM, DIM)
    assert full32.shape == (N_BASIS, DIM, DIM, DIM)
    assert sig64.shape  == (6,        DIM, DIM, DIM)

    assert full64.dtype == torch.float64
    assert full32.dtype == torch.float32

    # Same-call returns the same cached object.
    assert get_basis_stack(dtype=torch.float64) is full64


# ────────────────────────────────────────────────────────────────────────
# SO33Activation flags
# ────────────────────────────────────────────────────────────────────────

def test_so33_activation_float32_trains_one_step() -> None:
    """float32 activation produces finite output and finite gradients."""
    torch.manual_seed(0)
    act = SO33Activation(T=0.3, rtol=1e-3, atol=1e-4, method="rk4",
                         adjoint=True, dtype=torch.float32)
    assert act.coeffs.dtype == torch.float32
    assert act.basis_stack.dtype == torch.float32

    x = torch.randn(4, DIM, dtype=torch.float32) * 0.2
    y = act(x)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()

    loss = (y ** 2).sum()
    loss.backward()
    assert act.coeffs.grad is not None
    assert torch.isfinite(act.coeffs.grad).all()
    assert act.coeffs.grad.norm().item() > 0


def test_so33_activation_signature_only_has_six_params() -> None:
    """Signature-only activation has 6 trainable connection coefficients."""
    act = SO33Activation(signature_only=True, T=0.3, rtol=1e-3, atol=1e-4,
                         method="rk4", adjoint=True)
    assert act.coeffs.numel() == 6
    assert act.basis_stack.shape == (6, DIM, DIM, DIM)

    x = torch.randn(2, DIM, dtype=torch.float64) * 0.1
    y = act(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert act.coeffs.grad is not None
    assert torch.isfinite(act.coeffs.grad).all()


def test_so33_activation_freeze_coeffs_stops_grad_but_lets_outer_train() -> None:
    """freeze_coeffs=True: coeffs.requires_grad is False; outer Linear layers still train."""
    torch.manual_seed(1)
    act = SO33Activation(freeze_coeffs=True, T=0.3, rtol=1e-3, atol=1e-4,
                         method="rk4", adjoint=True)
    assert act.coeffs.requires_grad is False

    model = nn.Sequential(
        nn.Linear(DIM, DIM).double(),
        act,
        nn.Linear(DIM, 1).double(),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(8, DIM, dtype=torch.float64) * 0.2
    y_true = torch.randn(8, 1, dtype=torch.float64)

    coeffs_before = act.coeffs.detach().clone()
    linear_w_before = model[0].weight.detach().clone()

    opt.zero_grad()
    loss = ((model(x) - y_true) ** 2).mean()
    loss.backward()

    # Frozen coeffs receive no grad.
    assert act.coeffs.grad is None

    # Outer Linear layers do receive grads.
    assert model[0].weight.grad is not None
    assert model[0].weight.grad.norm().item() > 0

    opt.step()

    # After step: coeffs unchanged, outer Linear updated.
    assert torch.equal(act.coeffs, coeffs_before)
    assert not torch.equal(model[0].weight, linear_w_before)


# ────────────────────────────────────────────────────────────────────────
# BottleneckClassifier
# ────────────────────────────────────────────────────────────────────────

def test_bottleneck_classifier_relu_runs() -> None:
    """BottleneckClassifier with ReLU activation runs end-to-end."""
    torch.manual_seed(2)
    model = BottleneckClassifier(
        in_features=28, out_features=2, activation=nn.ReLU(), hidden=6,
    )
    x = torch.randn(16, 28, dtype=torch.float64)
    y = model(x)
    assert y.shape == (16, 2)
    assert torch.isfinite(y).all()

    # regularization_loss returns 0 for activations without one.
    assert model.regularization_loss().item() == 0.0


def test_bottleneck_classifier_so33_matches_so33network() -> None:
    """Wrapping SO33Activation in BottleneckClassifier matches SO33Network."""
    torch.manual_seed(3)
    so33_net = SO33Network(in_features=10, out_features=3, T=0.3, adjoint=True)

    # Build a BottleneckClassifier with the same components and copy weights.
    activation = so33_net.activation  # share the same module instance
    bn = BottleneckClassifier(
        in_features=10, out_features=3, activation=activation, hidden=DIM,
    )
    bn.input_proj.load_state_dict(so33_net.input_proj.state_dict())
    bn.output_proj.load_state_dict(so33_net.output_proj.state_dict())

    x = torch.randn(4, 10, dtype=torch.float64) * 0.2
    y_so33 = so33_net(x)
    y_bn   = bn(x)

    assert torch.allclose(y_so33, y_bn, atol=1e-10), \
        f"BottleneckClassifier output diverged from SO33Network"

    # regularization_loss forwards correctly when activation has one.
    assert torch.allclose(bn.regularization_loss(), so33_net.regularization_loss())


if __name__ == "__main__":
    print("\n╔" + "─" * 61 + "╗")
    print("║  TEST 7 · dtype + ablation flags + BottleneckClassifier    ║")
    print("╚" + "─" * 61 + "╝")
    test_basis_float32_metric_connection()
    test_basis_signature_only_count_and_pairs()
    test_get_basis_stack_caches_per_variant()
    test_so33_activation_float32_trains_one_step()
    test_so33_activation_signature_only_has_six_params()
    test_so33_activation_freeze_coeffs_stops_grad_but_lets_outer_train()
    test_bottleneck_classifier_relu_runs()
    test_bottleneck_classifier_so33_matches_so33network()
    print("  ✓ All checks passed")
