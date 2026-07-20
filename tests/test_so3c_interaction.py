"""
Test — SO3CInteraction: exact equivariance, conservation, padding masks.

Verifies:
  • Identity map at init (zero-init coupling MLP).
  • Map equivariance f(g z) = g f(z) under rotations AND boosts — this layer
    is equivariant by construction (bivector connection + invariant
    couplings), unlike the single-state activation which only conserves
    invariants.
  • Per-particle conservation of z_a . z_a along the interacting flow.
  • Padding-mask correctness: padded particles pass through unchanged and do
    not influence real ones.
  • Gradients reach the coupling MLP.

Run:
    python tests/test_so3c_interaction.py
    python -m pytest tests/test_so3c_interaction.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from so3c.algebra import bilinear_invariant, random_group_element
from so3c.interaction import SO3CInteraction


def _randomised(layer: SO3CInteraction, seed: int = 0) -> SO3CInteraction:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        last = layer.phi[-1]
        last.weight.copy_(
            torch.randn(last.weight.shape, dtype=torch.float64, generator=gen) * 0.3
        )
        last.bias.copy_(
            torch.randn(last.bias.shape, dtype=torch.float64, generator=gen) * 0.3
        )
    return layer


def test_identity_at_init() -> None:
    torch.manual_seed(0)
    layer = SO3CInteraction()
    x = torch.randn(4, 5, 6, dtype=torch.float64)
    err = (layer(x) - x).abs().max().item()
    assert err < 1e-14, f"not identity at init: {err:.2e}"
    print(f"  ✓ identity at init ({err:.1e})")


def test_equivariance() -> None:
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(3, 6, 6, dtype=torch.float64, generator=gen) * 0.5
    layer = _randomised(SO3CInteraction(rtol=1e-9, atol=1e-11))

    for boost in (0.0, 0.5, 1.0):
        g = random_group_element(rot_scale=1.0, boost_scale=boost, generator=gen)
        lhs = layer(x @ g.T)               # f(g z)
        rhs = layer(x) @ g.T               # g f(z)
        rel = ((lhs - rhs).norm() / rhs.norm()).item()
        assert rel < 1e-6, f"equivariance broken (boost={boost}): {rel:.2e}"
        print(f"  ✓ equivariance at boost_scale={boost}: rel. error {rel:.1e}")


def test_conservation() -> None:
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(3, 6, 6, dtype=torch.float64, generator=gen) * 0.5
    layer = _randomised(SO3CInteraction(rtol=1e-9, atol=1e-11))
    drift = layer.invariant_drift(x).max().item()
    assert drift < 1e-7, f"per-particle z.z drift {drift:.2e}"

    # The layer must actually move the state (otherwise conservation is vacuous).
    displacement = (layer(x) - x).abs().max().item()
    assert displacement > 1e-3, f"flow suspiciously static: {displacement:.2e}"
    print(f"  ✓ conservation: drift {drift:.1e} at displacement {displacement:.1e}")


def test_padding_mask() -> None:
    gen = torch.Generator().manual_seed(3)
    x_full = torch.randn(1, 5, 6, dtype=torch.float64, generator=gen) * 0.5
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    layer = _randomised(SO3CInteraction(rtol=1e-9, atol=1e-11))

    out_masked = layer(x_full, mask=mask)
    out_trunc = layer(x_full[:, :3])

    err_real = (out_masked[:, :3] - out_trunc).abs().max().item()
    assert err_real < 1e-7, f"padding influenced real particles: {err_real:.2e}"

    err_pad = (out_masked[:, 3:] - x_full[:, 3:]).abs().max().item()
    assert err_pad < 1e-14, f"padded particles changed: {err_pad:.2e}"
    print(f"  ✓ mask: real-particle match {err_real:.1e}, padded untouched {err_pad:.1e}")


def test_gradients() -> None:
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(2, 4, 6, dtype=torch.float64, generator=gen) * 0.5
    layer = _randomised(SO3CInteraction())
    loss = layer(x).pow(2).sum() + layer.regularization_loss()
    loss.backward()
    grads = [p.grad for p in layer.parameters()]
    assert all(g is not None for g in grads), "missing grads"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grads"
    total = sum(g.abs().sum().item() for g in grads)
    assert total > 0, "zero gradient signal"
    print(f"  ✓ gradients flow (|grad| sum = {total:.3e})")


if __name__ == "__main__":
    print("\n── SO3CInteraction tests ──")
    test_identity_at_init()
    test_equivariance()
    test_conservation()
    test_padding_mask()
    test_gradients()
    print("All SO3CInteraction tests passed.\n")
