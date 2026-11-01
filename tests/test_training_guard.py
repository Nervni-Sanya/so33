"""
Test 11 — divergence guards, early stopping and per-epoch records.

Verifies:
  • a step with a non-finite loss is skipped: the parameters and the
    optimiser state never see the NaN, and the run continues;
  • a non-finite GRADIENT with a finite loss (how a diverging adjoint pass
    shows up) is caught as well;
  • a run that keeps producing NaN stops after max_nonfinite_steps and is
    reported as diverged, with its parameters untouched;
  • evaluate_test reports non-finite logits as a divergence (no AUC)
    instead of an argmax accuracy near 0.5;
  • aggregate renders diverged seeds as such, including old records whose
    only trace is a NaN test AUC;
  • early stopping restores the best-validation-loss weights;
  • models with SO33 activations log their coefficient norms per epoch.

Run:
    python -m pytest tests/test_training_guard.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import pytest
import torch
import torch.nn as nn

from benchmarks.train import TrainConfig, predict_logits, train_classifier
from benchmarks.tabular_runner import evaluate_test
from benchmarks.aggregate import group_by_experiment_model, is_diverged, render_tabular
from so33.network import SO33Network


class _NaNEvery(nn.Module):
    """Linear classifier whose output is NaN on every ``period``-th call."""

    def __init__(self, period: int | None):
        super().__init__()
        self.lin = nn.Linear(4, 2).double()
        self.period = period
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        out = self.lin(x.double())
        if self.training and self.period and self.calls % self.period == 0:
            out = out * float("nan")
        return out

    def regularization_loss(self):
        return torch.zeros((), dtype=torch.float64)


class _NaNGrad(torch.autograd.Function):
    """Identity in the forward pass, NaN gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return g * float("nan")


class _BadBackward(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2).double()

    def forward(self, x):
        return _NaNGrad.apply(self.lin(x.double()))

    def regularization_loss(self):
        return torch.zeros((), dtype=torch.float64)


def _data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 4, dtype=torch.float64, generator=g)
    y = (X[:, 0] > 0).long()
    return X, y


def _cfg(**kw):
    base = dict(epochs=3, batch_size=16, lr=1e-2, seed=0, cosine_schedule=False)
    return TrainConfig(**{**base, **kw})


def test_nonfinite_loss_is_skipped_and_parameters_stay_finite():
    X, y = _data()
    model = _NaNEvery(period=3)
    res = train_classifier(model, X, y, X, y, _cfg())
    assert res.nonfinite_steps > 0 and not res.diverged
    assert res.first_nonfinite["reason"] == "non-finite loss"
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert res.epochs_run == 3


def test_nonfinite_gradient_with_finite_loss_is_caught():
    X, y = _data()
    model = _BadBackward()
    before = [p.detach().clone() for p in model.parameters()]
    res = train_classifier(model, X, y, X, y, _cfg(max_nonfinite_steps=2))
    assert res.diverged
    assert res.first_nonfinite["reason"] == "non-finite gradient"
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p, q)                  # no step was ever taken


def test_persistent_nan_is_reported_as_divergence():
    X, y = _data()
    model = _NaNEvery(period=1)                   # every training call is NaN
    before = [p.detach().clone() for p in model.parameters()]
    res = train_classifier(model, X, y, X, y, _cfg(max_nonfinite_steps=5))
    assert res.diverged and res.nonfinite_steps == 6
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p, q)


def test_evaluate_test_reports_nonfinite_logits_as_divergence():
    X, y = _data()

    class AllNaN(nn.Module):
        def forward(self, x):
            return torch.full((len(x), 2), float("nan"), dtype=torch.float64)

    with pytest.warns(RuntimeWarning, match="non-finite"):
        out = evaluate_test(AllNaN(), X, y, n_classes=2)
    assert out["diverged"] and out["nonfinite_frac"] == 1.0
    assert "test_auc" not in out and "bg_rej_30" not in out
    assert out["test_acc"] == 0.0                 # non-finite rows count as wrong


def _rec(model, seed, auc, diverged=None):
    tm = {"test_acc": 0.9, "test_auc": auc}
    r = {"experiment": "top_tagging_canonical", "dataset": "x", "model": model,
         "family": "equivariant_set", "seed": seed, "n_params": 542,
         "train_metrics": {"final_val_acc": 0.9}, "test_metrics": tm}
    if diverged is not None:
        r["diverged"] = diverged
    return r


def test_aggregate_reports_diverged_seeds():
    recs = [
        _rec("so33_equivariant_eta_bounded", 0, 0.94, diverged=False),
        _rec("so33_equivariant_eta_bounded", 1, None, diverged=True),
        _rec("so33_equivariant_unbounded", 0, float("nan")),   # pre-guard record
        _rec("so33_equivariant_unbounded", 1, float("nan")),
    ]
    assert is_diverged(recs[2]) and not is_diverged(recs[0])
    md = render_tabular(group_by_experiment_model(recs))
    assert "so33_equivariant_eta_bounded (1/2 diverged)" in md
    row = next(l for l in md.splitlines() if l.startswith("| so33_equivariant_unbounded"))
    assert "(2/2 diverged)" in row and "| diverged |" in row
    assert "nan" not in md


def test_early_stopping_restores_the_best_weights():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(64, 10, dtype=torch.float64, generator=g)
    y = torch.randint(0, 2, (64,), generator=g)                  # pure noise:
    Xv = torch.randn(64, 10, dtype=torch.float64, generator=g)    # validation loss
    yv = torch.randint(0, 2, (64,), generator=g)                  # rises as it overfits

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2)).double()

        def forward(self, x):
            return self.net(x)

        def regularization_loss(self):
            return torch.zeros((), dtype=torch.float64)

    torch.manual_seed(0)
    model = MLP()
    res = train_classifier(model, X, y, Xv, yv,
                           _cfg(epochs=40, lr=3e-2, early_stop_patience=3))
    assert res.epochs_run < 40 and res.restored_best
    with torch.no_grad():
        val_loss = nn.CrossEntropyLoss()(model(Xv), yv).item()
    assert val_loss == pytest.approx(res.best_val_loss, rel=1e-12)
    assert res.best_epoch < res.epochs_run


def test_history_records_so33_coefficient_norms():
    X = torch.randn(32, 6, dtype=torch.float64)
    y = (X[:, 0] > 0).long()
    model = SO33Network(in_features=6, out_features=2, T=0.3, method="rk4")
    res = train_classifier(model, X, y, X, y, _cfg(epochs=2))
    assert all(len(h["coeff_norms"]) == 1 for h in res.history)
    assert res.final_coeff_norms == [pytest.approx(model.activation.coeff_norm())]
    assert math.isfinite(res.best_val_loss)


def test_chunked_prediction_equals_a_full_batch_forward():
    torch.manual_seed(5)
    model = SO33Network(in_features=6, out_features=2, T=0.3, method="rk4")
    X = torch.randn(50, 6, dtype=torch.float64)
    with torch.no_grad():
        full = model(X)
    assert torch.allclose(predict_logits(model, X, chunk_size=7), full, rtol=0, atol=1e-12)
