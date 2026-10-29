"""
Test 13 — benchmarks.stats (seed-level comparisons).

Verifies against hand-computed values that the sample standard deviation
uses ddof = 1, that the paired per-seed difference and its 95% t-interval
are computed on the common seeds, that Welch's test matches scipy, that
diverged runs are excluded and counted, and that top-level metrics
(boost_ood's ood_auc) are found.

Run:
    python -m pytest tests/test_stats.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import math

import pytest

pytest.importorskip("scipy")

from benchmarks.stats import compare, main, render


def _rec(model, seed, auc, diverged=False, experiment="higgs"):
    return {"experiment": experiment, "model": model, "seed": seed,
            "family": "matched_bottleneck", "n_params": 1, "diverged": diverged,
            "train_metrics": {"final_val_acc": 0.7},
            "test_metrics": {"test_acc": 0.7, "test_auc": auc}}


RECORDS = (
    [_rec("relu", s, a) for s, a in enumerate([0.75, 0.76, 0.77])]
    + [_rec("so33", s, a) for s, a in enumerate([0.76, 0.78, 0.77])]   # diffs .01 .02 .00
    + [_rec("so33", 3, None, diverged=True)]
)


def test_summary_and_paired_difference_match_hand_values():
    rows = {r["model"]: r for r in compare(RECORDS, "higgs", "relu")}
    s = rows["so33"]
    assert s["n"] == 3 and s["n_diverged"] == 1
    assert s["mean"] == pytest.approx(0.77)
    assert s["sd"] == pytest.approx(0.01)                     # ddof = 1
    # paired diffs (0.01, 0.02, 0.00): mean 0.01, sd 0.01, t_{0.975, 2} = 4.3027
    half = 4.302652729911275 * 0.01 / math.sqrt(3)
    assert s["diff_mean"] == pytest.approx(0.01)
    assert s["diff_ci_low"] == pytest.approx(0.01 - half)
    assert s["diff_ci_high"] == pytest.approx(0.01 + half)
    assert s["paired_p"] == pytest.approx(0.2254, abs=1e-3)   # t = sqrt(3), df = 2
    from scipy import stats
    assert s["welch_p"] == pytest.approx(
        stats.ttest_ind([0.76, 0.78, 0.77], [0.75, 0.76, 0.77], equal_var=False).pvalue)
    assert "welch_p" not in rows["relu"]


def test_top_level_metric_and_markdown(tmp_path):
    recs = [{"experiment": "boost_ood", "model": m, "seed": s, "ood_auc": v}
            for m, vals in (("relu_mlp", [0.94, 0.95]), ("so33", [0.88, 0.89]))
            for s, v in enumerate(vals)]
    rows = compare(recs, "boost_ood", "relu_mlp", metric="ood_auc")
    md = render(rows, "boost_ood", "relu_mlp", "ood_auc")
    assert "| so33 | 2 |" in md and "-0.0600" in md
    res = tmp_path / "results"
    res.mkdir()
    for i, r in enumerate(recs):
        (res / f"r{i}.json").write_text(json.dumps(r))
    out = tmp_path / "stats.md"
    assert main(["--results-dir", str(res), "--experiment", "boost_ood",
                 "--baseline", "relu_mlp", "--metric", "ood_auc", "--out", str(out)]) == 0
    assert "paired diff" in out.read_text(encoding="utf-8")


def test_unknown_baseline_is_an_error():
    with pytest.raises(ValueError, match="baseline"):
        compare(RECORDS, "higgs", "gelu")
