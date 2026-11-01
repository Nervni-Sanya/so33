"""
Test 15 — contracts of the training loop and evaluate_test.

Pins what the divergence guard, early stopping and evaluation promise:
BatchNorm buffers are restored on a skipped step, an epoch with non-finite
validation rows is never chosen as the best epoch, a run that never took an
optimiser step is diverged, validation and evaluation run in eval mode, the
loop seeds itself, the cosine schedule anneals to zero, early stopping waits
exactly ``patience`` epochs and keeps the earliest of tied epochs, final_*
metrics describe the returned (restored) model, and first_nonfinite records
epoch, step, reason and coefficients.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import pytest
import torch
import torch.nn as nn

from benchmarks.train import TrainConfig, train_classifier
from benchmarks.tabular_runner import evaluate_test
from so33.network import SO33Network


def _data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 4, dtype=torch.float64, generator=g)
    y = (X[:, 0] > 0).long()
    return X, y


def _cfg(**kw):
    base = dict(epochs=3, batch_size=16, lr=1e-2, seed=0, cosine_schedule=False)
    return TrainConfig(**{**base, **kw})


def _with_zero_reg(model):
    model.regularization_loss = lambda: torch.zeros((), dtype=torch.float64)
    return model


class _NaNEvery(nn.Module):
    """Linear classifier whose output is NaN on every ``period``-th call (training only)."""

    def __init__(self, period):
        super().__init__()
        self.lin = nn.Linear(4, 2).double()
        self.period, self.calls = period, 0

    def forward(self, x):
        self.calls += 1
        out = self.lin(x.double())
        if self.training and self.period and self.calls % self.period == 0:
            out = out * float("nan")
        return out

    def regularization_loss(self):
        return torch.zeros((), dtype=torch.float64)


class _NaNAtTrainCall(nn.Module):
    """Output NaN on training forward call number ``at`` (1-based); never if ``at`` is 0."""

    def __init__(self, at, net=None):
        super().__init__()
        self.net = net if net is not None else nn.Linear(4, 2).double()
        self.at, self.n = at, 0

    def forward(self, x):
        out = self.net(x)
        if self.training:
            self.n += 1
            if self.n == self.at:
                out = out * float("nan")
        return out

    def regularization_loss(self):
        reg = getattr(self.net, "regularization_loss", None)
        return reg() if reg else torch.zeros((), dtype=torch.float64)


# ── the divergence record ────────────────────────────────────────────────────

def test_first_nonfinite_record_gives_epoch_step_reason_and_coefficients():
    X, y = _data(n=64)                            # batch 16 -> 4 steps per epoch
    res = train_classifier(_NaNAtTrainCall(at=6), X, y, X, y, _cfg(epochs=3))
    assert res.nonfinite_steps == 1 and not res.diverged
    fb = res.first_nonfinite
    assert (fb["epoch"], fb["step"], fb["reason"]) == (2, 6, "non-finite loss")   # 6th batch = 2nd of epoch 2
    assert fb["coeff_norms"] == []                # no SO33 activation in this model
    clean = train_classifier(_NaNAtTrainCall(at=0), X, y, X, y, _cfg(epochs=1))
    assert clean.first_nonfinite is None and clean.nonfinite_steps == 0

    torch.manual_seed(0)                          # with SO33 activations the record carries ||c||
    net = SO33Network(in_features=4, out_features=2, T=0.3, method="rk4")
    initial = net.activation.coeff_norm()
    res = train_classifier(_NaNAtTrainCall(at=1, net=net), X, y, X, y, _cfg(epochs=1))
    assert res.first_nonfinite["coeff_norms"] == [pytest.approx(initial)]   # no step taken before it


# ── early stopping ───────────────────────────────────────────────────────────

def _noise_task():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(64, 10, dtype=torch.float64, generator=g)
    y = torch.randint(0, 2, (64,), generator=g)
    Xv = torch.randn(64, 10, dtype=torch.float64, generator=g)
    yv = torch.randint(0, 2, (64,), generator=g)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2)).double()
    return _with_zero_reg(net), X, y, Xv, yv


def test_early_stopping_waits_exactly_patience_epochs_after_the_best_epoch():
    model, X, y, Xv, yv = _noise_task()
    res = train_classifier(model, X, y, Xv, yv, _cfg(epochs=40, lr=3e-2, early_stop_patience=3))
    losses = [h["val_loss"] for h in res.history]
    assert len(res.history) == res.epochs_run < 40
    assert res.best_epoch == 1 + losses.index(min(losses))      # history and best_epoch agree
    assert res.best_val_loss == min(losses)
    assert res.epochs_run == res.best_epoch + 3                  # patience = 3 epochs without improvement


def test_no_restore_when_the_last_epoch_is_the_best():
    X, y = _data(n=128, seed=3)
    model = _with_zero_reg(nn.Linear(4, 2).double())
    res = train_classifier(model, X, y, X, y, _cfg(epochs=3, lr=1e-3, early_stop_patience=5))
    losses = [h["val_loss"] for h in res.history]
    assert losses == sorted(losses, reverse=True)                # still improving at the last epoch
    assert res.best_epoch == res.epochs_run == 3 and not res.restored_best


def test_a_tie_in_validation_loss_keeps_the_earliest_epoch():
    X, y = _data()
    model = _with_zero_reg(nn.Linear(4, 2).double())
    res = train_classifier(model, X, y, X, y, _cfg(epochs=10, lr=0.0, early_stop_patience=3))
    assert len({h["val_loss"] for h in res.history}) == 1       # lr = 0: every epoch is identical
    assert res.best_epoch == 1 and res.epochs_run == 4


class _NaNValRows(nn.Module):
    """Linear classifier; on evaluation call number ``call`` the first ``frac`` of the rows are NaN."""

    def __init__(self, call, frac=0.5):
        super().__init__()
        self.lin = nn.Linear(4, 2).double()
        self.evals, self.call, self.frac = 0, call, frac

    def forward(self, x):
        out = self.lin(x.double())
        if not self.training:
            self.evals += 1
            if self.evals == self.call:
                out = out.clone()
                out[: int(self.frac * len(out))] = float("nan")
        return out

    def regularization_loss(self):
        return torch.zeros((), dtype=torch.float64)


def _val_task():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, 4, dtype=torch.float64, generator=g)
    y = (X[:, 0] > 0).long()
    Xv = torch.randn(200, 4, dtype=torch.float64, generator=g)
    yv = (Xv[:, 0] > 0).long()
    yv[:100] = torch.randint(0, 2, (100,), generator=g)          # first half: label noise (hard rows)
    return X, y, Xv, yv


def test_nonfinite_validation_rows_count_as_wrong_and_are_logged():
    X, y, Xv, yv = _val_task()
    yv[:100] = 0                                  # the rows that will be NaN carry label 0 (argmax(NaN) = 0)
    torch.manual_seed(0)
    res = train_classifier(_NaNValRows(call=1), X, y, Xv, yv, _cfg(epochs=1, batch_size=20))
    h = res.history[0]
    assert h["val_nonfinite_frac"] == pytest.approx(0.5)
    assert math.isfinite(h["val_loss"])           # computed on the finite rows only
    assert h["val_acc"] <= 0.5                    # NaN rows are wrong, not "class 0"


def test_an_epoch_with_nonfinite_validation_rows_is_not_selected_as_best():     # BUG on HEAD
    """Epoch 3 has the lowest loss only because its hard half is NaN and left out of the loss."""
    X, y, Xv, yv = _val_task()
    torch.manual_seed(0)
    res = train_classifier(_NaNValRows(call=3), X, y, Xv, yv,
                           _cfg(epochs=6, batch_size=20, early_stop_patience=3))
    assert res.history[2]["val_nonfinite_frac"] == pytest.approx(0.5)
    assert res.best_epoch == 6 and not res.restored_best


def test_final_validation_metrics_describe_the_returned_model():                 # BUG on HEAD (minor)
    model, X, y, Xv, yv = _noise_task()
    res = train_classifier(model, X, y, Xv, yv, _cfg(epochs=40, lr=3e-2, early_stop_patience=3))
    assert res.restored_best
    model.eval()
    with torch.no_grad():
        returned = nn.CrossEntropyLoss()(model(Xv), yv).item()
    assert res.final_val_loss == pytest.approx(returned, rel=1e-12)


# ── what the loop does around the model ──────────────────────────────────────

def test_validation_runs_in_eval_mode():
    torch.manual_seed(0)
    model = _with_zero_reg(nn.Sequential(nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(),
                                         nn.Dropout(0.5), nn.Linear(16, 2)).double())
    X, y = _data(n=64)
    res = train_classifier(model, X, y, X, y, _cfg(epochs=1))
    model.eval()
    with torch.no_grad():
        again = nn.CrossEntropyLoss()(model(X), y).item()
    assert res.final_val_loss == pytest.approx(again, rel=1e-12)


def test_train_classifier_seeds_itself():
    """Batch order and dropout depend on cfg.seed, not on the global RNG state at the call."""
    X, y = _data()

    def run(prior):
        torch.manual_seed(0)
        model = _with_zero_reg(nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.3),
                                             nn.Linear(16, 2)).double())
        torch.manual_seed(prior)
        torch.rand(prior)                          # a different global RNG state when training starts
        train_classifier(model, X, y, X, y, _cfg(epochs=2))
        return [p.detach().clone() for p in model.parameters()]

    assert all(torch.equal(p, q) for p, q in zip(run(1), run(2)))


def test_cosine_schedule_anneals_the_learning_rate_to_zero(monkeypatch):
    captured = {}
    orig = torch.optim.Adam

    class SpyAdam(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["opt"] = self

    monkeypatch.setattr(torch.optim, "Adam", SpyAdam)
    X, y = _data()
    train_classifier(_NaNEvery(period=None), X, y, X, y, _cfg(epochs=4, lr=1e-2, cosine_schedule=True))
    assert captured["opt"].param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


class _NaNBeforeBN(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.lin = nn.Linear(4, 8).double()
        self.bn = nn.BatchNorm1d(8).double()
        self.out = nn.Linear(8, 2).double()
        self.period, self.calls = period, 0

    def forward(self, x):
        h = self.lin(x.double())
        if self.training:
            self.calls += 1
            if self.calls % self.period == 0:
                h = h * float("nan")
        return self.out(torch.relu(self.bn(h)))

    def regularization_loss(self):
        return torch.zeros((), dtype=torch.float64)


def test_a_skipped_nan_batch_does_not_poison_the_batchnorm_statistics():         # BUG on HEAD
    X, y = _data(n=64)
    model = _NaNBeforeBN(period=3)
    res = train_classifier(model, X, y, X, y, _cfg(epochs=3))
    assert res.nonfinite_steps > 0 and not res.diverged
    assert all(torch.isfinite(b).all() for b in model.buffers())
    assert math.isfinite(res.final_val_loss) and "val_nonfinite_frac" not in res.history[-1]


def test_a_run_that_skips_every_step_is_reported_as_diverged():                  # BUG on HEAD
    X, y = _data(n=64)                            # 12 steps in all, all skipped: below max_nonfinite_steps
    res = train_classifier(_NaNEvery(period=1), X, y, X, y, _cfg(max_nonfinite_steps=20))
    assert res.diverged


def test_evaluate_test_uses_eval_mode_and_leaves_batchnorm_statistics_alone():
    pytest.importorskip("sklearn")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Dropout(0.5), nn.Linear(8, 2)).double()
    X, y = _data(n=64)
    model.train()
    before = [b.clone() for b in model.buffers()]
    first = evaluate_test(model, X, y, n_classes=2)
    second = evaluate_test(model, X, y, n_classes=2)
    assert first == second                         # dropout off: deterministic
    assert all(torch.equal(p, q) for p, q in zip(before, model.buffers()))


class _NaNOnCalls(nn.Module):
    """Linear classifier whose training output is NaN on the calls listed in ``bad``."""

    def __init__(self, bad):
        super().__init__()
        self.lin = nn.Linear(4, 2).double()
        self.bad, self.calls = set(bad), 0

    def forward(self, x):
        out = self.lin(x.double())
        if self.training:
            self.calls += 1
            if self.calls in self.bad:
                out = out * float("nan")
        return out

    def regularization_loss(self):
        return torch.zeros((), dtype=torch.float64)


def test_rare_nonfinite_batches_in_a_long_run_do_not_end_it():
    """25 isolated bad batches in 2,500 steps (1%): skipped, the run goes on."""
    X, y = _data(n=250)
    model = _NaNOnCalls(range(100, 2501, 100))
    res = train_classifier(model, X, y, X, y, _cfg(epochs=10, batch_size=1, lr=1e-3))
    assert res.nonfinite_steps == 25 and not res.diverged and res.epochs_run == 10
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_more_than_max_consecutive_nonfinite_batches_end_the_run():
    X, y = _data(n=64)                                     # 4 steps per epoch
    model = _NaNOnCalls(range(9, 10_000))                  # healthy for 8 steps, then NaN
    res = train_classifier(model, X, y, X, y, _cfg(epochs=20, max_nonfinite_steps=5))
    assert res.diverged and res.nonfinite_steps == 6
    assert res.first_nonfinite["step"] == 9
