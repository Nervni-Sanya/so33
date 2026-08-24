"""
Test — float32 viability, the precision the GPU runs will actually use.

T4 GPUs have no fp64 tensor cores (fp64 runs at 1/32 the fp32 rate), so the
Kaggle scaling study runs in float32. Everything the architecture claims
must therefore survive single precision: the closed-form exponential must
still land in SO(3,C), the flow must still conserve the complex invariant,
and the classifiers' logits must still be Lorentz-invariant.

Tolerances here are ~1e-5, not the ~1e-12 of the float64 suite. That is not
a weaker claim about the mathematics — the float64 tests still assert the
exact statements — it is the honest float32 round-off floor.

Run:
    python -m pytest tests/test_so3c_float32.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from so3c.activation import SO3CActivation
from so3c.algebra import bilinear_invariant, cross_matrix, expm_so3c
from so3c.lift import bivector_lift, random_lorentz_pair

F32_TOL = 1e-5


def _random_p4(n: int, gen: torch.Generator, dtype=torch.float64) -> torch.Tensor:
    p = torch.randn(n, 3, dtype=dtype, generator=gen)
    m = torch.rand(n, dtype=dtype, generator=gen) + 0.5
    E = torch.sqrt(m * m + p.pow(2).sum(-1))
    return torch.cat([E.unsqueeze(-1), p], dim=-1)


def test_expm_stays_in_group_float32() -> None:
    """exp must remain complex-orthogonal with unit determinant in float32."""
    gen = torch.Generator().manual_seed(0)
    a = torch.complex(torch.randn(2048, 3, generator=gen),
                      torch.randn(2048, 3, generator=gen) * 0.5)
    Q = expm_so3c(a)
    assert Q.dtype == torch.complex64, "float32 input must give complex64"
    eye = torch.eye(3, dtype=Q.dtype)
    ortho = (Q.transpose(-1, -2) @ Q - eye).abs().max().item()
    det = (torch.linalg.det(Q) - 1.0).abs().max().item()
    assert ortho < F32_TOL, f"Q^T Q != I in float32: {ortho:.2e}"
    assert det < F32_TOL, f"det Q != 1 in float32: {det:.2e}"

    ref = expm_so3c(a.to(torch.complex128))
    err = (Q.to(torch.complex128) - ref).abs().max().item()
    assert err < F32_TOL, f"float32 vs float64 exp: {err:.2e}"


def test_flow_conserves_invariant_float32() -> None:
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(512, 6, generator=gen)
    act = SO3CActivation(mode="dynamic", method="exact", dtype=torch.float32)
    with torch.no_grad():
        last = act.metric.net[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=gen) * 0.3)
        last.bias.copy_(torch.randn(last.bias.shape, generator=gen) * 0.3)
    drift = act.invariant_drift(x).max().item()
    assert drift < F32_TOL, f"z.z drift in float32: {drift:.2e}"


def test_bivector_intertwiner_float32() -> None:
    """bivec(Lp, Lq) = Q bivec(p, q) must hold to single precision."""
    gen = torch.Generator().manual_seed(2)
    p = _random_p4(256, gen)
    q = _random_p4(256, gen)
    L, a = random_lorentz_pair(rot_scale=1.0, boost_scale=0.5, generator=gen)

    p32, q32, L32 = p.float(), q.float(), L.float()
    Q = expm_so3c(a.to(torch.complex64))
    lhs = bivector_lift(p32 @ L32.T, q32 @ L32.T)
    rhs = (Q @ bivector_lift(p32, q32).unsqueeze(-1)).squeeze(-1)
    rel = ((lhs - rhs).abs().max() / rhs.abs().max()).item()
    assert rel < F32_TOL, f"intertwiner broken in float32: {rel:.2e}"


@pytest.mark.parametrize("name", ["so3c_invariant_set", "so3c_equivariant_set"])
def test_set_model_invariance_float32(name: str) -> None:
    """Logits must not move under a Lorentz transform of the jet, float32."""
    from benchmarks.models import build_model

    gen = torch.Generator().manual_seed(3)
    p4 = _random_p4(8 * 12, gen).reshape(8, 12, 4)
    mask = torch.ones(8, 12, dtype=torch.float64)
    x = torch.cat([p4, mask.unsqueeze(-1)], dim=-1)
    L, _ = random_lorentz_pair(rot_scale=1.0, boost_scale=0.5, generator=gen)
    x_t = torch.cat([p4 @ L.T, mask.unsqueeze(-1)], dim=-1)

    torch.manual_seed(4)
    model = build_model(name, in_features=4, out_features=2,
                        representation="constituents", dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        err = (model(x_t.float()) - model(x.float())).abs().max().item()
    assert err < F32_TOL, f"{name} logits moved in float32: {err:.2e}"


def test_float32_model_trains() -> None:
    """A float32 model must produce finite gradients end to end."""
    from benchmarks.models import build_model

    gen = torch.Generator().manual_seed(5)
    p4 = _random_p4(16 * 10, gen).reshape(16, 10, 4).float()
    x = torch.cat([p4, torch.ones(16, 10, 1)], dim=-1)
    y = torch.randint(0, 2, (16,))
    torch.manual_seed(6)
    model = build_model("so3c_equivariant_set", in_features=4, out_features=2,
                        representation="constituents", dtype=torch.float32)
    loss = torch.nn.functional.cross_entropy(model(x), y) + model.regularization_loss()
    loss.backward()
    for pn, prm in model.named_parameters():
        assert prm.grad is not None, f"missing grad: {pn}"
        assert torch.isfinite(prm.grad).all(), f"non-finite grad in float32: {pn}"
