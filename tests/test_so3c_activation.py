"""
Test — SO3CActivation: conservation, exact-vs-solver agreement, the
invariant-bound lesson, and gradient flow.

Verifies:
  • Identity map at init (dynamic mode, zero-init metric MLP).
  • Exact conservation of z.z in method="exact" (machine precision) and
    solver-tolerance-limited conservation for dopri5.
  • Closed form agrees with the adaptive solver.
  • Invariants of the output are exactly invariant under SO(3,C) input
    transformations with bound_input="bilinear" — and are NOT with
    "euclidean" (the complexified analogue of the so33 eta-bound result).
  • Gradients reach the connection parameters in both modes.

Run:
    python tests/test_so3c_activation.py
    python -m pytest tests/test_so3c_activation.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from so3c.activation import SO3CActivation
from so3c.algebra import bilinear_invariant, random_group_element, real_to_complex


def _randomised(act: SO3CActivation, seed: int = 0) -> SO3CActivation:
    """Give the layer a non-trivial connection (as if partially trained)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if act.mode == "static":
            act.coeffs.copy_(torch.randn(6, dtype=torch.float64, generator=gen) * 0.3)
        else:
            last = act.metric.net[-1]
            last.weight.copy_(
                torch.randn(last.weight.shape, dtype=torch.float64, generator=gen) * 0.3
            )
            last.bias.copy_(
                torch.randn(last.bias.shape, dtype=torch.float64, generator=gen) * 0.3
            )
    return act


def test_identity_at_init() -> None:
    torch.manual_seed(0)
    act = SO3CActivation(mode="dynamic", method="exact", bound_input="none")
    x = torch.randn(16, 6, dtype=torch.float64)
    err = (act(x) - x).abs().max().item()
    assert err < 1e-14, f"not identity at init: {err:.2e}"
    print(f"  ✓ identity at init ({err:.1e})")


def test_conservation() -> None:
    x = torch.randn(32, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    for mode in ("static", "dynamic"):
        act = _randomised(SO3CActivation(mode=mode, method="exact"))
        drift = act.invariant_drift(x).max().item()
        assert drift < 1e-12, f"{mode}/exact drift {drift:.2e}"

        act_ode = _randomised(SO3CActivation(mode=mode, method="dopri5"))
        drift_ode = act_ode.invariant_drift(x).max().item()
        assert drift_ode < 1e-6, f"{mode}/dopri5 drift {drift_ode:.2e}"
        print(f"  ✓ conservation [{mode}]: exact {drift:.1e}, dopri5 {drift_ode:.1e}")


def test_exact_matches_solver() -> None:
    x = torch.randn(16, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    for mode in ("static", "dynamic"):
        # Seed the global RNG identically before each construction: the
        # dynamic-metric hidden layer draws its init from the global RNG.
        torch.manual_seed(10)
        exact = _randomised(SO3CActivation(mode=mode, method="exact"))
        torch.manual_seed(10)
        solver = _randomised(SO3CActivation(mode=mode, method="dopri5",
                                            rtol=1e-10, atol=1e-12))
        err = (exact(x) - solver(x)).abs().max().item()
        assert err < 1e-8, f"{mode}: exact vs dopri5 {err:.2e}"
        print(f"  ✓ exact vs dopri5 [{mode}]: {err:.1e}")


def test_invariant_bound_preserves_group_structure() -> None:
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(64, 6, dtype=torch.float64, generator=gen)
    g = random_group_element(rot_scale=1.0, boost_scale=1.0, generator=gen)

    act = _randomised(SO3CActivation(mode="dynamic", method="exact",
                                     bound_input="bilinear"))
    q_base = bilinear_invariant(act(x))
    q_transformed = bilinear_invariant(act(x @ g.T))
    err_inv = (q_transformed - q_base).abs().max().item()
    assert err_inv < 1e-10, f"bilinear bound broke invariance: {err_inv:.2e}"

    act_euc = _randomised(SO3CActivation(mode="dynamic", method="exact",
                                         bound_input="euclidean"))
    q_base_e = bilinear_invariant(act_euc(x))
    q_transformed_e = bilinear_invariant(act_euc(x @ g.T))
    err_euc = (q_transformed_e - q_base_e).abs().max().item()
    assert err_euc > 1e-3, (
        f"euclidean bound unexpectedly invariant ({err_euc:.2e}) — "
        f"the ablation should demonstrate breakage"
    )
    print(f"  ✓ invariance: bilinear bound {err_inv:.1e}  vs  euclidean bound {err_euc:.1e}")


def test_complex_input_path() -> None:
    x = torch.randn(8, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    act = _randomised(SO3CActivation(mode="dynamic", method="exact"))
    out_real = act(x)
    out_complex = act(real_to_complex(x))
    assert torch.is_complex(out_complex)
    err = (out_complex - real_to_complex(out_real)).abs().max().item()
    assert err < 1e-14, f"complex path mismatch: {err:.2e}"
    print(f"  ✓ complex input path consistent ({err:.1e})")


def test_gradients() -> None:
    x = torch.randn(8, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    for mode in ("static", "dynamic"):
        act = _randomised(SO3CActivation(mode=mode, method="exact"))
        loss = act(x).pow(2).sum() + act.regularization_loss(x)
        loss.backward()
        grads = [p.grad for p in act.parameters()]
        assert all(g is not None for g in grads), f"{mode}: missing grads"
        assert all(torch.isfinite(g).all() for g in grads), f"{mode}: non-finite grads"
        total = sum(g.abs().sum().item() for g in grads)
        assert total > 0, f"{mode}: zero gradient signal"
        print(f"  ✓ gradients flow [{mode}] (|grad| sum = {total:.3e})")


if __name__ == "__main__":
    print("\n── SO3CActivation tests ──")
    test_identity_at_init()
    test_conservation()
    test_exact_matches_solver()
    test_invariant_bound_preserves_group_structure()
    test_complex_input_path()
    test_gradients()
    print("All SO3CActivation tests passed.\n")
