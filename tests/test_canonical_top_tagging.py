"""
Test 8 — canonical Kasieczka split loader, background rejection, aggregate.

Verifies, on tiny synthetic ``top_tagging_{train,val,test}.npz`` files:
  • use_canonical_splits=True honours the per-file split sizes (not a
    random 70/15/15 re-split) and tags the dataset name "canonical".
  • max_train_samples caps the train split only; val/test load in full.
  • A missing val file falls back to a seeded 5% carve-out of train
    (deterministic), and the test split stays the full test file.
  • The global-RMS normalisation scale is fitted on train only.
  • The default (internal) path is unchanged.
  • evaluate_test reports 1/eps_B at eps_S in {0.3, 0.5} correctly
    (hand-computed ROC), inf for a perfect separator, and warns, rather
    than silently dropping the metrics, when scikit-learn is missing.
  • aggregate renders the equivariant_set family with the 1/eB column,
    skips JSONs that are not per-model records, and writes UTF-8.

Run:
    python -m pytest tests/test_canonical_top_tagging.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import math

import numpy as np
import pytest
import torch

from benchmarks.datasets import load_top_tagging_constituents


def _make_npz(path, n, seed, scale=1.0):
    """(n, 200, 4) constituents, 5 real particles per jet, class-correlated."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) > 0.5).astype(np.int64)
    cons = np.zeros((n, 200, 4), dtype=np.float32)
    for i in range(n):
        base = rng.normal(size=(5, 4)).astype(np.float32) * 10.0 * scale
        base[:, 0] = (np.abs(base[:, 0]) + 50.0 * scale
                      + labels[i] * 15.0 * scale)
        cons[i, :5] = base
    np.savez(path, constituents=cons, labels=labels)


@pytest.fixture
def full_cache(tmp_path):
    _make_npz(tmp_path / "top_tagging_train.npz", 400, 0)
    _make_npz(tmp_path / "top_tagging_val.npz",   120, 1)
    _make_npz(tmp_path / "top_tagging_test.npz",  200, 2)
    return tmp_path


@pytest.fixture
def noval_cache(tmp_path):
    _make_npz(tmp_path / "top_tagging_train.npz", 400, 0)
    _make_npz(tmp_path / "top_tagging_test.npz",  200, 2)
    return tmp_path


def test_canonical_sizes_follow_files(full_cache):
    s = load_top_tagging_constituents(
        full_cache, n_constituents=16, use_canonical_splits=True)
    assert (len(s.X_train), len(s.X_val), len(s.X_test)) == (400, 120, 200)
    assert s.X_train.shape[1:] == (16, 5)          # 4 momenta + mask channel
    assert "canonical" in s.name


def test_max_train_samples_caps_train_only(full_cache):
    s = load_top_tagging_constituents(
        full_cache, n_constituents=16, use_canonical_splits=True,
        max_train_samples=150)
    assert len(s.X_train) == 150
    assert len(s.X_val) == 120 and len(s.X_test) == 200


def test_missing_val_file_carves_five_percent_of_train(noval_cache, capsys):
    kw = dict(n_constituents=16, use_canonical_splits=True, seed=0)
    s = load_top_tagging_constituents(noval_cache, **kw)
    assert len(s.X_val) == 20 and len(s.X_train) == 380
    assert len(s.X_test) == 200                    # test stays the full file
    assert "no val file found" in capsys.readouterr().out
    s2 = load_top_tagging_constituents(noval_cache, **kw)
    assert torch.equal(s.y_val, s2.y_val)          # deterministic carve


def test_normalisation_scale_is_fitted_on_train_only(tmp_path):
    # Test jets 10x larger than train: if the scale leaked from test, the
    # train RMS would not come out at exactly 1.
    _make_npz(tmp_path / "top_tagging_train.npz", 300, 0, scale=1.0)
    _make_npz(tmp_path / "top_tagging_val.npz",   100, 1, scale=1.0)
    _make_npz(tmp_path / "top_tagging_test.npz",  100, 2, scale=10.0)
    s = load_top_tagging_constituents(
        tmp_path, n_constituents=16, use_canonical_splits=True,
        normalize="global")

    def rms(X):
        real = X[..., :4][X[..., 4].bool()]
        return real.pow(2).mean().sqrt().item()

    assert rms(s.X_train) == pytest.approx(1.0, abs=1e-4)
    assert rms(s.X_test) > 5.0


def test_default_path_is_still_internal_split(full_cache):
    s = load_top_tagging_constituents(full_cache, n_constituents=16)
    total = 400 + 120 + 200
    assert len(s.X_train) == int(0.7 * total)
    assert "internal" in s.name


def test_canonical_rejects_one_class_split(tmp_path):
    _make_npz(tmp_path / "top_tagging_train.npz", 100, 0)
    _make_npz(tmp_path / "top_tagging_val.npz",    50, 1)
    # a test file whose labels are all one class
    rng = np.random.default_rng(3)
    cons = rng.normal(size=(40, 200, 4)).astype(np.float32)
    np.savez(tmp_path / "top_tagging_test.npz",
             constituents=cons, labels=np.zeros(40, dtype=np.int64))
    with pytest.raises(ValueError, match="constant/empty labels"):
        load_top_tagging_constituents(
            tmp_path, n_constituents=16, use_canonical_splits=True)


class _ScoreModel(torch.nn.Module):
    """Logits [-s, s] with s = X[:, 0]; softmax([-s, s])[1] is monotone in s."""
    def forward(self, X):
        return torch.stack([-X[:, 0], X[:, 0]], dim=-1)


def _scores_and_labels():
    """10 positives, 100 negatives with a hand-computable ROC.

    Descending order near the top:
        pos 3.0, neg 2.95, pos 2.9, neg 2.85, pos 2.8, pos 2.7,
        neg 2.65, pos 2.6, ...
    eps_S = 0.3 needs 3 positives (threshold 2.8): 2 negatives above -> 1/eB = 50.
    eps_S = 0.5 needs 5 positives (threshold 2.6): 3 negatives above -> 1/eB = 100/3.
    The other 97 negatives sit well below the lowest positive (2.1).
    """
    pos = [3.0 - 0.1 * i for i in range(10)]
    neg = [2.95, 2.85, 2.65] + [-3.0 + 0.05 * i for i in range(97)]
    s = torch.tensor(pos + neg, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor([1] * 10 + [0] * 100)
    return s, y


def test_background_rejection_matches_hand_computed_roc():
    pytest.importorskip("sklearn")
    from benchmarks.tabular_runner import evaluate_test
    X, y = _scores_and_labels()
    out = evaluate_test(_ScoreModel(), X, y, n_classes=2)
    assert out["bg_rej_30"] == pytest.approx(50.0)
    assert out["bg_rej_50"] == pytest.approx(100.0 / 3.0)
    assert 0.5 < out["test_auc"] < 1.0


def test_background_rejection_is_inf_for_perfect_separation():
    pytest.importorskip("sklearn")
    from benchmarks.tabular_runner import evaluate_test
    X = torch.tensor([[2.0]] * 10 + [[-2.0]] * 10)
    y = torch.tensor([1] * 10 + [0] * 10)
    out = evaluate_test(_ScoreModel(), X, y, n_classes=2)
    assert out["test_auc"] == pytest.approx(1.0)
    assert math.isinf(out["bg_rej_30"]) and math.isinf(out["bg_rej_50"])


def test_missing_sklearn_warns_instead_of_silently_dropping(monkeypatch):
    from benchmarks.tabular_runner import evaluate_test
    # A None entry in sys.modules makes `from sklearn.metrics import ...`
    # raise ImportError, as on a machine without scikit-learn.
    monkeypatch.setitem(sys.modules, "sklearn.metrics", None)
    X, y = _scores_and_labels()
    with pytest.warns(RuntimeWarning, match="scikit-learn"):
        out = evaluate_test(_ScoreModel(), X, y, n_classes=2)
    assert "test_acc" in out and "test_auc" not in out


def _record(model, family, auc, rej, seed=0):
    return {
        "experiment": "top_tagging_canonical", "dataset": "x",
        "model": model, "family": family, "seed": seed, "n_params": 4802,
        "train_metrics": {"final_val_acc": 0.9},
        "test_metrics": {"test_acc": 0.9, "test_auc": auc, "bg_rej_30": rej},
    }


def test_aggregate_renders_equivariant_family_with_rejection_column():
    from benchmarks.aggregate import group_by_experiment_model, render_tabular
    recs = [
        _record("eta_invariants", "equivariant_set", 0.948, 50.0, seed=0),
        _record("eta_invariants", "equivariant_set", 0.948, 48.0, seed=1),
        _record("eta_invariants", "equivariant_set", 0.948, float("inf"), seed=2),
        _record("relu_bottleneck", "matched_bottleneck", 0.755, 8.0),
    ]
    md = render_tabular(group_by_experiment_model(recs))
    assert "equivariant / invariant set models" in md
    assert "1/eB@0.3" in md
    row = next(l for l in md.splitlines() if l.startswith("| eta_invariants"))
    assert "| 3 |" in row                      # all three seeds counted
    assert "49±1" in row                      # mean/std over the finite seeds
    assert "inf" not in md and "nan" not in md


def test_aggregate_skips_records_without_a_model_field(tmp_path):
    from benchmarks.aggregate import group_by_experiment_model
    diag = {"experiment": "diagnose_equivariant", "seed": 0}   # no "model"
    groups = group_by_experiment_model([diag, _record("m", "natural_width", 0.8, 5.0)])
    assert list(groups) == [("top_tagging_canonical", "m")]


def test_aggregate_cli_writes_utf8(tmp_path):
    from benchmarks.aggregate import main as aggregate_main
    res = tmp_path / "results"
    res.mkdir()
    (res / "a.json").write_text(
        json.dumps(_record("eta_invariants", "equivariant_set", 0.948, 50.0)))
    (res / "diag.json").write_text(
        json.dumps({"experiment": "diagnose_equivariant", "seed": 0}))
    out = tmp_path / "tables.md"
    assert aggregate_main(["--results-dir", str(res), "--out", str(out)]) == 0
    assert "±" in out.read_bytes().decode("utf-8")
