"""
Test 12 — what the benchmark harness records.

Verifies, on toy massless jets and tiny canonical npz files:
  • every result JSON carries provenance (git commit, versions, argv);
  • experiment names include every non-default protocol option, so runs
    with different normalisations / K / epochs never overwrite each other
    (the per-component run used to overwrite the global-scaling JSONs);
  • --seeds runs several seeds in one process;
  • an Architecture-B record carries the stability block (largest train
    constituent norm, connection caps < 1 under "auto"), the trained
    coefficients and the equivariance errors at the trained connection;
    "off" keeps the legacy cap of 1;
  • --save-scores writes per-sample scores aligned with the test labels;
  • <act>_mlp_w<H> models are natural-width MLPs of width H.

Run:
    python -m pytest tests/test_harness_records.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest
import torch

from benchmarks.models import build_model, count_parameters, model_family
from benchmarks.provenance import experiment_tag, parse_seeds, run_metadata
from benchmarks.synthetic import massless_jet_split
from benchmarks.tabular_runner import run_tabular_experiment


def _make_npz(path, n, seed):
    """(n, 200, 4) constituents, 5 real particles per jet, class-correlated."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) > 0.5).astype(np.int64)
    cons = np.zeros((n, 200, 4), dtype=np.float32)
    for i in range(n):
        base = rng.normal(size=(5, 4)).astype(np.float32) * 10.0
        base[:, 0] = np.abs(base[:, 0]) + 50.0 + labels[i] * 15.0
        cons[i, :5] = base
    np.savez(path, constituents=cons, labels=labels)


@pytest.fixture
def canonical_cache(tmp_path):
    for split, n, seed in (("train", 200, 0), ("val", 60, 1), ("test", 80, 2)):
        _make_npz(tmp_path / f"top_tagging_{split}.npz", n, seed)
    return tmp_path


def test_run_metadata_records_code_version_and_command_line():
    meta = run_metadata(["prog", "--flag"])
    assert meta["argv"] == ["prog", "--flag"]
    assert meta["git_commit"] is None or len(meta["git_commit"]) == 40
    for key in ("git_dirty", "python", "torch", "numpy", "platform", "utc_time"):
        assert key in meta


def test_experiment_tag_and_seed_parsing():
    assert experiment_tag("x", {"norm": "global", "k": 32}, {"norm": "global", "k": 32}) == "x"
    assert (experiment_tag("x", {"norm": "per_component", "k": 16},
                           {"norm": "global", "k": 32})
            == "x-norm_per_component-k_16")
    assert parse_seeds("0-2") == [0, 1, 2]
    assert parse_seeds("0,3,5") == [0, 3, 5]
    assert parse_seeds(None, default=4) == [4]


def test_width_mlp_names():
    assert model_family("tanh_mlp_w24") == "natural_width"
    m = build_model("tanh_mlp_w24", in_features=105, out_features=2)
    assert count_parameters(m) == 105 * 24 + 24 + 24 * 2 + 2
    m = build_model("relu_mlp_w8", in_features=4, out_features=2,
                    representation="constituents")
    assert count_parameters(m) == 4 * 8 + 8 + 8 * 2 + 2


@pytest.fixture(scope="module")
def toy_split():
    return massless_jet_split(n_jets=300, n_constituents=8, seed=0)


def _run(toy_split, tmp_path, models, **kw):
    run_tabular_experiment("toy", toy_split, models=models, epochs=1,
                           representation="constituents", results_dir=tmp_path,
                           run_meta=run_metadata(["test"]), **kw)
    return {m: json.loads((tmp_path / f"toy__{m}__seed0.json").read_text())
            for m in models}


def test_architecture_b_record_has_stability_and_equivariance(toy_split, tmp_path):
    recs = _run(toy_split, tmp_path, ["so33_equivariant_eta_bounded", "eta_invariants"],
                save_scores=True)
    b = recs["so33_equivariant_eta_bounded"]
    assert b["diverged"] is False
    assert b["stability"]["connection_cap_mode"] == "auto"
    r_max = b["stability"]["max_train_constituent_norm"]
    assert r_max == pytest.approx(
        toy_split.X_train[..., :4][toy_split.X_train[..., 4] > 0].norm(dim=-1).max().item())
    caps = b["so33"]["connection_caps"]
    assert len(caps) == 4 and all(c < 1.0 for c in caps)
    assert len(b["so33"]["coeffs"][0]) == 15
    eq = b["equivariance"]
    assert eq["eps_sigma"] > 0 and eq["eps_m"] < 1e-6 and len(eq["per_channel"]) == 4
    assert b["provenance"]["argv"] == ["test"]
    assert all("coeff_norms" in h for h in b["history"])
    a = recs["eta_invariants"]                    # no flow: no SO33 blocks
    assert "so33" not in a and "equivariance" not in a
    scores = np.load(tmp_path / "toy__so33_equivariant_eta_bounded__seed0__scores.npz")
    assert len(scores["scores"]) == len(toy_split.y_test)
    assert np.array_equal(scores["labels"], toy_split.y_test.numpy())
    from sklearn.metrics import roc_auc_score     # scores[i] belongs to test jet i
    assert roc_auc_score(scores["labels"], scores["scores"]) == pytest.approx(
        b["test_metrics"]["test_auc"], abs=1e-6)


def test_connection_cap_off_keeps_the_legacy_cap(toy_split, tmp_path):
    recs = _run(toy_split, tmp_path, ["so33_equivariant_unbounded"], connection_cap="off")
    assert recs["so33_equivariant_unbounded"]["so33"]["connection_caps"] == [1.0] * 4


def test_runs_with_different_protocols_do_not_overwrite_each_other(canonical_cache, tmp_path):
    from benchmarks.run_top_tagging import main
    common = ["--cache-dir", str(canonical_cache), "--representation", "constituents",
              "--canonical-splits", "--models", "relu_bottleneck", "--epochs", "1",
              "--n-constituents", "16", "--results-dir", str(tmp_path)]
    assert main(common) == 0
    assert main(common + ["--normalize", "per_component", "--seeds", "0-1"]) == 0
    names = sorted(p.name for p in tmp_path.glob("*.json"))
    assert names == [
        "top_tagging_canonical-k_16-epochs_1__relu_bottleneck__seed0.json",
        "top_tagging_canonical-norm_per_component-k_16-epochs_1__relu_bottleneck__seed0.json",
        "top_tagging_canonical-norm_per_component-k_16-epochs_1__relu_bottleneck__seed1.json",
    ]
    rec = json.loads((tmp_path / names[1]).read_text())
    assert "norm=per_component" in rec["dataset"]


def test_results_do_not_depend_on_the_global_rng_state(tmp_path):
    """Each model is seeded right before it is built, so its record does not
    depend on whatever consumed random numbers earlier in the process (other
    models, other seeds, data loading)."""
    from benchmarks.datasets import synthetic_tabular
    split = synthetic_tabular(n_samples=300, n_features=6, n_classes=2,
                              name="toy_flat", seed=0)
    meta = run_metadata(["test"])
    recs = []
    for prior_seed, prior_draws, out in ((1, 5, tmp_path / "a"), (2, 9, tmp_path / "b")):
        torch.manual_seed(prior_seed)
        torch.rand(prior_draws)                    # a different global RNG state
        run_tabular_experiment("toy_flat", split, models=["so33"], epochs=1,
                               results_dir=out, run_meta=meta)
        recs.append(json.loads((out / "toy_flat__so33__seed0.json").read_text()))
    assert recs[0]["test_metrics"] == recs[1]["test_metrics"]
    assert recs[0]["so33"]["coeffs"] == recs[1]["so33"]["coeffs"]


def test_canonical_seed_rerun_reproduces_the_multi_seed_record(tmp_path):
    """Without a canonical val file a fixed 5% of train is held out, so
    `--seed 1` alone gives the same record as seed 1 of `--seeds 0-1`."""
    data = tmp_path / "data"
    data.mkdir()
    _make_npz(data / "top_tagging_train.npz", 200, 0)      # no val file
    _make_npz(data / "top_tagging_test.npz", 80, 2)
    from benchmarks.run_top_tagging import main
    common = ["--cache-dir", str(data), "--representation", "constituents",
              "--canonical-splits", "--models", "relu_bottleneck", "--epochs", "1",
              "--n-constituents", "8"]
    assert main(common + ["--seeds", "0-1", "--results-dir", str(tmp_path / "multi")]) == 0
    assert main(common + ["--seed", "1", "--results-dir", str(tmp_path / "single")]) == 0
    name = "top_tagging_canonical-k_8-epochs_1__relu_bottleneck__seed1.json"
    multi = json.loads((tmp_path / "multi" / name).read_text())
    single = json.loads((tmp_path / "single" / name).read_text())
    assert multi["test_metrics"] == single["test_metrics"]
    assert multi["n_train"] == single["n_train"] == 190


def test_aggregate_renders_tagged_boost_ood_experiments():
    from benchmarks.aggregate import group_by_experiment_model, render_boost_ood
    recs = [{"experiment": exp, "model": "relu_mlp", "seed": 0, "n_params": 2306,
             "id_auc": 1.0, "ood_auc": 0.95, "auc_gap": 0.05}
            for exp in ("boost_ood", "boost_ood-cap_off")]
    md = render_boost_ood(group_by_experiment_model(recs))
    assert "## boost_ood\n" in md and "## boost_ood-cap_off" in md


def test_adult_fallback_is_not_filed_as_adult(tmp_path, monkeypatch):
    """If Adult cannot be fetched the loader falls back to breast cancer;
    those runs must not land in the Adult ('neutral') experiment."""
    import sklearn.datasets

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", offline)
    from benchmarks.run_neutral import main
    assert main(["--models", "relu_bottleneck", "--epochs", "1",
                 "--results-dir", str(tmp_path)]) == 0
    names = [q.name for q in tmp_path.glob("*.json")]
    assert names == ["neutral_breast_cancer-epochs_1__relu_bottleneck__seed0.json"]


def test_boost_ood_names_runs_with_non_default_bins(tmp_path):
    from benchmarks.run_boost_ood import main
    assert main(["--quick", "--models", "relu_mlp", "--rapidity-bin", "0.5",
                 "--results-dir", str(tmp_path)]) == 0
    names = sorted(q.name for q in tmp_path.glob("*.json"))
    assert names == ["boost_ood-n_1000-epochs_5-bin_0.5__relu_mlp__seed0.json"]
