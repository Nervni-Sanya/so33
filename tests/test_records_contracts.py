"""
Test 16 — contracts of the result records, aggregate, stats and provenance.

Pins: a diverged run is flagged in its record and leaves no score file;
aggregate keeps experiments and families in separate tables, uses the sample
standard deviation, prints no spread for a single seed, says when a column
uses fewer seeds than its row and reports diverged / old NaN boost-OOD seeds;
stats uses Welch's test, pairs only common seeds, excludes guard-diverged
runs and degrades gracefully; run_metadata reports its own repository (not an
enclosing one) and survives a missing git; parse_seeds and experiment_tag
edge cases; the equivariance probe against a hand computation.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import math
import shutil
import subprocess

import pytest
import torch
import torch.nn as nn

import benchmarks.tabular_runner as tabular_runner
from benchmarks.aggregate import group_by_experiment_model, render_boost_ood, render_tabular
from benchmarks.datasets import synthetic_tabular
from benchmarks.provenance import experiment_tag, parse_seeds, run_metadata


# ── run_tabular_experiment: what a diverged run leaves behind ────────────────

class _Flaky(nn.Module):
    """mode "train": every training forward is NaN; mode "eval": NaN logits at evaluation only."""

    def __init__(self, mode, n_in):
        super().__init__()
        self.lin = nn.Linear(n_in, 2).double()
        self.mode = mode

    def forward(self, x):
        out = self.lin(x.double())
        if (self.mode == "train") == self.training:
            out = out * float("nan")
        return out

    def regularization_loss(self):
        return torch.zeros((), dtype=torch.float64)


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.filterwarnings("ignore:.*non-finite.*:RuntimeWarning")
def test_diverged_run_is_flagged_in_its_record_and_leaves_no_scores(mode, tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(tabular_runner, "build_model",
                        lambda name, in_features, **kw: _Flaky(mode, in_features))
    split = synthetic_tabular(n_samples=300, n_features=6, n_classes=2, name="toy_flat", seed=0)
    tabular_runner.run_tabular_experiment(
        "toy", split, models=["relu_bottleneck"], epochs=2, batch_size=8, results_dir=tmp_path,
        save_scores=True, run_meta={"git_commit": None})
    rec = json.loads((tmp_path / "toy__relu_bottleneck__seed0.json").read_text())
    assert rec["diverged"] is True and rec["family"] == "matched_bottleneck"
    assert rec["train_guard"]["diverged"] is (mode == "train")       # stopped by the guard ...
    assert rec["test_metrics"].get("diverged", False) is (mode == "eval")   # ... or seen at test time
    assert "equivariance" not in rec
    if mode == "eval":
        assert "test_auc" not in rec["test_metrics"]
        assert not (tmp_path / "toy__relu_bottleneck__seed0__scores.npz").exists()
    md = render_tabular(group_by_experiment_model([rec]))
    assert "relu_bottleneck (1/1 diverged)" in md


# ── aggregate ────────────────────────────────────────────────────────────────

def _boost(model, seed, id_auc, ood_auc, gap, **kw):
    r = {"experiment": "boost_ood", "model": model, "seed": seed, "n_params": 56,
         "id_auc": id_auc, "ood_auc": ood_auc, "auc_gap": gap}
    r.update(kw)
    return r


def test_aggregate_boost_ood_reports_diverged_seeds_and_old_nan_records():
    nan = float("nan")
    recs = [_boost("relu", 0, 1.0, 0.90, 0.10, diverged=False),
            _boost("relu", 1, nan, nan, nan, diverged=True),     # new record
            _boost("so33", 0, 1.0, 0.80, 0.20),
            _boost("so33", 1, 1.0, nan, nan),                    # old record: NaN OOD AUC, no flag
            _boost("dead", 0, nan, nan, nan, diverged=True)]
    md = render_boost_ood(group_by_experiment_model(recs))
    assert "| relu (1/2 diverged) | 2 | 56 | 1.000" in md and "0.900" in md
    assert "so33 (1/2 diverged)" in md
    row = next(l for l in md.splitlines() if l.startswith("| dead"))
    assert "(1/1 diverged)" in row and "| diverged |" in row
    assert "nan" not in md.lower()


def _tab(experiment, model, seed, auc, family="matched_bottleneck", **tm):
    return {"experiment": experiment, "dataset": "x", "model": model, "family": family, "seed": seed,
            "n_params": 10, "train_metrics": {"final_val_acc": auc},
            "test_metrics": {"test_acc": auc, "test_auc": auc, **tm}}


def test_aggregate_keeps_experiments_and_families_in_separate_tables():
    recs = [_tab("top_tagging_canonical", "m", 0, 0.90), _tab("top_tagging_canonical", "m", 1, 0.92),
            _tab("top_tagging_canonical-ntrain_1000", "m", 0, 0.70),
            _tab("top_tagging_canonical", "w", 0, 0.80, family="natural_width")]
    md = render_tabular(group_by_experiment_model(recs))
    assert md.count("## top_tagging_canonical\n") == 1 and "## top_tagging_canonical-ntrain_1000" in md
    assert md.count("### matched bottleneck") == 2 and md.count("### natural width MLPs") == 1
    big, small = md.split("## top_tagging_canonical-ntrain_1000")
    assert "0.910±0.014" in big and "0.700" in small and "0.910" not in small


def test_aggregate_spread_is_the_sample_standard_deviation():
    recs = [_tab("e", "m", s, a) for s, a in enumerate((0.90, 0.92, 0.94))]
    assert "0.920±0.020" in render_tabular(group_by_experiment_model(recs))   # ddof = 1 (ddof = 0: 0.016)


def test_aggregate_does_not_print_a_zero_spread_for_a_single_seed():             # BUG on HEAD
    md = render_tabular(group_by_experiment_model([_tab("e", "m", 0, 0.93)]))
    assert "±0.000" not in md and "0.930" in md


def test_aggregate_says_when_a_column_uses_fewer_seeds_than_the_row():           # BUG on HEAD
    recs = [_tab("e", "m", 0, 0.90, bg_rej_30=100.0), _tab("e", "m", 1, 0.92, bg_rej_30=120.0),
            _tab("e", "m", 2, 0.94, bg_rej_30=float("inf"))]       # perfect separation in seed 2
    row = next(l for l in render_tabular(group_by_experiment_model(recs)).splitlines()
               if l.startswith("| m "))
    assert "| 3 |" in row and "110±14 (2/3)" in row                # not a bare "110±14" over 3 seeds


# ── stats ────────────────────────────────────────────────────────────────────

def _srec(model, seed, auc, experiment="higgs", **kw):
    r = {"experiment": experiment, "model": model, "seed": seed, "family": "matched_bottleneck",
         "n_params": 1, "train_metrics": {"final_val_acc": 0.7},
         "test_metrics": {"test_acc": 0.7, "test_auc": auc}}
    r.update(kw)
    return r


def test_welch_differs_from_the_pooled_test_when_variances_and_sizes_differ():
    scipy_stats = pytest.importorskip("scipy").stats
    from benchmarks.stats import compare
    base = [0.70, 0.71, 0.69, 0.70, 0.72, 0.68]
    model = [0.75, 0.90, 0.60]
    recs = ([_srec("b", s, v) for s, v in enumerate(base)]
            + [_srec("m", s, v) for s, v in enumerate(model)])
    row = next(r for r in compare(recs, "higgs", "b") if r["model"] == "m")
    welch = scipy_stats.ttest_ind(model, base, equal_var=False).pvalue
    pooled = scipy_stats.ttest_ind(model, base, equal_var=True).pvalue
    assert abs(welch - pooled) > 0.02                # these data tell the two tests apart
    assert row["welch_p"] == pytest.approx(welch, rel=1e-12)


def test_paired_difference_uses_only_the_common_seeds():
    scipy_stats = pytest.importorskip("scipy").stats
    from benchmarks.stats import compare
    base = {0: 0.70, 1: 0.72, 2: 0.71, 3: 0.69}
    model = {2: 0.75, 3: 0.74, 4: 0.90, 5: 0.60}      # common seeds: 2, 3
    recs = ([_srec("b", s, v) for s, v in base.items()]
            + [_srec("m", s, v) for s, v in model.items()])
    row = next(r for r in compare(recs, "higgs", "b") if r["model"] == "m")
    assert row["n"] == 4 and row["paired_n"] == 2
    assert row["diff_mean"] == pytest.approx(((0.75 - 0.71) + (0.74 - 0.69)) / 2)
    assert row["paired_p"] == pytest.approx(scipy_stats.ttest_rel([0.75, 0.74], [0.71, 0.69]).pvalue)


def test_stats_with_a_single_seed_or_a_missing_metric_degrade_gracefully():
    pytest.importorskip("scipy")
    from benchmarks.stats import compare, render
    recs = [_srec("b", 0, 0.70), _srec("m", 0, 0.80),
            {"experiment": "higgs", "model": "n", "seed": 0, "test_metrics": {"test_acc": 0.5}}]
    rows = {r["model"]: r for r in compare(recs, "higgs", "b")}
    assert rows["m"]["n"] == 1 and math.isnan(rows["m"]["sd"]) and "welch_p" not in rows["m"]
    assert rows["n"]["n"] == 0                        # no test_auc in that record
    assert "—" in render(list(rows.values()), "higgs", "b", "test_auc")


def test_zero_variance_paired_differences_are_reported_not_nan():
    pytest.importorskip("scipy")
    from benchmarks.stats import compare
    base = [_srec("b", s, 0.5 + 0.25 * s) for s in range(3)]
    same = [_srec("m", s, 0.5 + 0.25 * s) for s in range(3)]               # identical to the baseline
    shifted = [_srec("m", s, 0.5 + 0.25 * s + 0.125) for s in range(3)]    # constant +0.125 (exact in binary)
    row = next(r for r in compare(base + same, "higgs", "b") if r["model"] == "m")
    assert row["paired_p"] == 1.0 and row["diff_mean"] == 0.0
    row = next(r for r in compare(base + shifted, "higgs", "b") if r["model"] == "m")
    assert row["paired_p"] == 0.0 and row["diff_ci_low"] == row["diff_ci_high"] == 0.125


def test_a_guard_diverged_run_with_finite_metrics_is_still_excluded():
    """The training guard can stop a run whose (finite) parameters still give a finite AUC."""
    pytest.importorskip("scipy")
    from benchmarks.stats import compare
    recs = [_srec("b", s, v) for s, v in enumerate((0.70, 0.71, 0.72))]
    recs += [_srec("m", 0, 0.80), _srec("m", 1, 0.81), _srec("m", 2, 0.99, diverged=True)]
    row = next(r for r in compare(recs, "higgs", "b") if r["model"] == "m")
    assert row["n"] == 2 and row["n_diverged"] == 1 and row["mean"] == pytest.approx(0.805)


# ── provenance ───────────────────────────────────────────────────────────────

def test_run_metadata_degrades_when_git_is_unavailable(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", no_git)
    meta = run_metadata(["x"])
    assert meta["git_commit"] is None and meta["git_dirty"] is None
    json.dumps(meta)                                  # still serialisable


def _init_repo(path):
    run = lambda *a: subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t",
                                     "-c", "commit.gpgsign=false", *a], check=True, capture_output=True)
    run("init", "-q")
    (path / "f.txt").write_text("a")
    run("add", "f.txt")
    run("commit", "-q", "-m", "c")
    return run


def test_run_metadata_reports_commit_and_dirty_flag_of_its_own_repository(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    from benchmarks import provenance
    run = _init_repo(tmp_path)
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    clean = provenance.run_metadata(["x"])
    assert len(clean["git_commit"]) == 40 and clean["git_dirty"] is False
    (tmp_path / "f.txt").write_text("b")
    assert provenance.run_metadata(["x"])["git_dirty"] is True


def test_run_metadata_does_not_report_an_enclosing_unrelated_repository(tmp_path, monkeypatch):   # BUG on HEAD
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    from benchmarks import provenance
    _init_repo(tmp_path)
    vendored = tmp_path / "vendor"
    vendored.mkdir()
    monkeypatch.setattr(provenance, "REPO_ROOT", vendored)     # code lives inside someone else's repo
    meta = provenance.run_metadata(["x"])
    assert meta["git_commit"] is None and meta["git_dirty"] is None


def test_parse_seeds_edge_cases():
    assert parse_seeds(" 1 , 2 ") == [1, 2]
    assert parse_seeds("0 - 3") == [0, 1, 2, 3]
    assert parse_seeds("0, ,2") == [0, 2]
    assert parse_seeds(None, default=4) == [4]
    for bad in ("", ",,", "a", "1.5", "1-2-3"):
        if bad:
            with pytest.raises(ValueError):
                parse_seeds(bad)
    with pytest.raises(ValueError):
        parse_seeds(",,")


def test_parse_seeds_rejects_reversed_ranges_and_trains_no_seed_twice():          # BUG on HEAD
    for bad in ("7-0", "0,7-3"):
        with pytest.raises(ValueError):
            parse_seeds(bad)
    assert parse_seeds("0-2,1-3") == [0, 1, 2, 3] and parse_seeds("3,3") == [3]


def test_experiment_tag_never_contains_a_double_underscore_or_path_characters():  # BUG on HEAD
    for value in ("a___b", "_x", "a/b", "C:\\data\\x", "a b"):
        tag = experiment_tag("x", {"opt": value}, {"opt": None})
        assert "__" not in tag and not set("/\\: ") & set(tag), tag
    assert experiment_tag("x", {"wd": 1e-05}, {"wd": 0.0}) == "x-wd_1e-05"


# ── equivariance probe ──────────────────────────────────────────────────────

def test_activation_equivariance_matches_a_hand_computation():
    from benchmarks.equivariance import activation_equivariance, sample_group_elements
    from so33.basis import ETA

    class Shift(nn.Module):                           # sigma(x) = x + c e: not equivariant
        dtype = torch.float64

        def __init__(self, c, e):
            super().__init__()
            self.c, self.e = c, e

        def forward(self, x):
            return x + self.c * self.e

    e = torch.tensor([1.0, 0, 0, 0.5, 0, 0], dtype=torch.float64)
    torch.manual_seed(0)
    x = torch.randn(40, 6, dtype=torch.float64)
    gs = sample_group_elements(4, 0.5, seed=1)
    out = activation_equivariance(Shift(0.7, e), x, gs)
    eta = ETA.to(torch.float64)
    y = x + 0.7 * e
    rels, maxes, ems = [], [], []
    for g in gs:
        lhs = x @ g.T + 0.7 * e                       # sigma(g x)
        rel = (lhs - y @ g.T).norm(dim=1) / y.norm(dim=1)   # ||sigma(g x) - g sigma(x)|| / ||sigma(x)||
        rels.append(rel.mean())
        maxes.append(rel.max())
        ems.append(((y * eta * y).sum(-1) - (lhs * eta * lhs).sum(-1)).abs().mean())
    assert out["eps_sigma"] == pytest.approx(float(sum(rels) / 4), rel=1e-12)
    assert out["eps_sigma_max"] == pytest.approx(float(sum(maxes) / 4), rel=1e-12)
    assert out["eps_m"] == pytest.approx(float(sum(ems) / 4), rel=1e-12)
    assert out["nonfinite_frac"] == 0.0
    res = activation_equivariance(Shift(float("nan"), e), x, gs)
    assert math.isnan(res["eps_sigma"]) and res["nonfinite_frac"] == 1.0


def test_rewritten_result_records_do_not_make_the_code_dirty(tmp_path, monkeypatch):
    """results/*.json are committed and rewritten by re-runs; that must not flag
    the code as modified in every later record."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    from benchmarks import provenance
    run = _init_repo(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text("{}")
    run("add", "results/r.json")
    run("commit", "-q", "-m", "results")
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    (tmp_path / "results" / "r.json").write_text('{"rerun": 1}')
    assert provenance.run_metadata(["x"])["git_dirty"] is False
    (tmp_path / "f.txt").write_text("code change")
    assert provenance.run_metadata(["x"])["git_dirty"] is True


def test_a_diverged_rerun_removes_the_stale_scores_of_the_previous_run(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    split = synthetic_tabular(n_samples=300, n_features=6, n_classes=2, name="toy_flat", seed=0)
    kw = dict(models=["relu_bottleneck"], epochs=2, batch_size=8, results_dir=tmp_path,
              save_scores=True, run_meta={"git_commit": None})
    tabular_runner.run_tabular_experiment("toy", split, **kw)
    scores = tmp_path / "toy__relu_bottleneck__seed0__scores.npz"
    assert scores.exists()
    monkeypatch.setattr(tabular_runner, "build_model",
                        lambda name, in_features, **k: _Flaky("eval", in_features))
    tabular_runner.run_tabular_experiment("toy", split, **kw)
    rec = json.loads((tmp_path / "toy__relu_bottleneck__seed0.json").read_text())
    assert rec["diverged"] is True and not scores.exists()
    assert not list(tmp_path.glob("*.tmp"))                # atomic write left nothing behind


def test_stats_reports_the_number_of_paired_seeds_and_explains_an_unknown_baseline(tmp_path, capsys):
    pytest.importorskip("scipy")
    from benchmarks.stats import compare, main, render
    recs = [{"experiment": "e", "model": m, "seed": s, "test_metrics": {"test_auc": a}}
            for m, vals in (("base", [0.70, 0.71, 0.72]), ("m", [0.71, 0.73]))
            for s, a in enumerate(vals)]
    md = render(compare(recs, "e", "base"), "e", "base", "test_auc")
    assert "(n=2)" in md
    res = tmp_path / "res"
    res.mkdir()
    for i, r in enumerate(recs):
        (res / f"r{i}.json").write_text(json.dumps(r))
    assert main(["--results-dir", str(res), "--experiment", "e", "--baseline", "nope"]) == 2
    assert "base, m" in capsys.readouterr().err
