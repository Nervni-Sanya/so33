"""
Linear energy-flow-polynomial baseline (benchmarks/run_efp_baseline.py).

Verifies, against energyflow (the EFP tests skip without it):
  • The (E, px, py, pz) -> (pT, y, phi) conversion equals energyflow's
    ptyphims_from_p4s, and stays finite where float32 rounding puts a massless
    constituent's E below |pz| (energyflow's arctanh(pz/E) is NaN there).
  • The d = 1 EFP (one edge) is sum_{i,j} z_i z_j theta_ij over ALL ordered
    pairs (twice the i < j sum), with z_i = pT_i / sum pT and
    theta_ij = (Delta y^2 + Delta phi^2)^(beta/2), Delta phi wrapped across
    phi = +-pi; the d = 2 multigraphs (double edge, path, product of two
    edges) follow the same definition.
  • The EFPs are invariant under longitudinal boosts, rotations about the beam
    and a global rescaling of the momenta (so the loader's normalisation does
    not matter), and do change under a transverse boost.
  • Only real constituents (mask > 0, pT > 0) enter; an empty jet gives
    all-zero EFPs; batch_compute with a process pool and in chunks equals the
    serial result.
  • The accuracy threshold maximises val accuracy and never splits tied
    scores.
  • Experiment / model names follow run_top_tagging's rule for the same data
    options (so the rows land in the same aggregate table).
  • End to end on --toy-jets 300 and on tiny canonical npz files (dmax 3):
    JSON keys of the score-based schema, n_params = #EFPs + 1, a finite AUC
    well above 0.5, test accuracy at the val-chosen threshold, and per-jet
    scores aligned with the test labels.
  • --cache-efps serves a smaller dmax bit for bit and refuses caches made for
    other data, another beta or a smaller dmax.
  • Without energyflow the runner exits with the "pip install energyflow" hint.

Run:
    python -m pytest tests/test_efp_baseline.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import math

import numpy as np
import pytest

from benchmarks import run_efp_baseline as efp
from benchmarks.provenance import experiment_tag
from benchmarks.synthetic import generate_massless_jets

N_COL, D_COL, P_COL = (efp.SPEC_COLS.index(c) for c in ("n", "d", "p"))

SCHEMA_KEYS = {"experiment", "dataset", "model", "family", "seed", "n_params",
               "model_size", "n_train", "n_val", "n_test", "n_features",
               "walltime_sec", "diverged", "train_metrics", "test_metrics",
               "config", "provenance"}


@pytest.fixture
def ef():
    """The optional energyflow package; the EFP tests skip without it."""
    return pytest.importorskip("energyflow")


# A five-constituent jet straddling phi = +-pi, in (pT, y, phi).
PT = np.array([300.0, 150.0, 80.0, 40.0, 10.0])
RAP = np.array([0.10, -0.25, 0.32, 0.05, -0.40])
PHI = np.array([3.05, -3.10, 2.80, -2.95, 3.12])


def _packed_jet(pt, rap, phi, k=8):
    """One packed jet (1, k, 5) of massless constituents, zero-padded."""
    X = np.zeros((1, k, 5))
    m = len(pt)
    X[0, :m, 0] = pt * np.cosh(rap)
    X[0, :m, 1] = pt * np.cos(phi)
    X[0, :m, 2] = pt * np.sin(phi)
    X[0, :m, 3] = pt * np.sinh(rap)
    X[0, :m, 4] = 1.0
    return X


def _theta(i, j, beta, wrap=True):
    dphi = abs(PHI[i] - PHI[j])
    if wrap:
        dphi = min(dphi, 2.0 * math.pi - dphi)
    return ((RAP[i] - RAP[j]) ** 2 + dphi ** 2) ** (beta / 2.0)


def test_ptyphi_conversion_matches_energyflow(ef):
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 3)) * np.array([40.0, 40.0, 120.0])
    mass = np.where(np.arange(50) < 25, 0.0, rng.uniform(0.1, 5.0, 50))
    p4 = np.column_stack([np.sqrt((p ** 2).sum(axis=1) + mass ** 2), p])
    np.testing.assert_allclose(efp.p4_to_ptyphi(p4),
                               ef.ptyphims_from_p4s(p4, mass=False),
                               rtol=1e-10, atol=1e-10)
    # E a hair below |pz| (float32 rounding of a massless particle):
    tachyon = np.array([[10.0, 1e-3, 0.0, 10.0 + 1e-6]])
    with np.errstate(invalid="ignore"):
        assert np.isnan(ef.ptyphims_from_p4s(tachyon, mass=False)[0, 1])
    y = efp.p4_to_ptyphi(tachyon)[0, 1]
    assert y == pytest.approx(np.arcsinh((10.0 + 1e-6) / 1e-3))   # = eta


@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_d1_efp_is_the_sum_over_all_ordered_pairs(ef, beta):
    z = PT / PT.sum()
    m = len(PT)
    ordered = sum(z[i] * z[j] * _theta(i, j, beta)
                  for i in range(m) for j in range(m))
    unordered = sum(z[i] * z[j] * _theta(i, j, beta)
                    for i in range(m) for j in range(i + 1, m))
    unwrapped = sum(z[i] * z[j] * _theta(i, j, beta, wrap=False)
                    for i in range(m) for j in range(m))

    F = efp.compute_efps(_packed_jet(PT, RAP, PHI), efp.make_efpset(1, beta))
    assert F.shape == (1, 1)                      # d <= 1: the one-edge graph
    assert F[0, 0] == pytest.approx(ordered, rel=1e-12)
    assert F[0, 0] == pytest.approx(2.0 * unordered, rel=1e-12)
    assert abs(F[0, 0] - unwrapped) > 0.5 * F[0, 0]   # Delta phi is wrapped
    single = ef.EFP([(0, 1)], measure="hadr", beta=beta, normed=True,
                    coords="ptyphim")
    assert single.compute(np.column_stack([PT, RAP, PHI])) == pytest.approx(
        ordered, rel=1e-12)


def test_d2_efps_follow_the_multigraph_definition(ef):
    beta = 0.5
    z = PT / PT.sum()
    T = np.array([[_theta(i, j, beta) for j in range(5)] for i in range(5)])
    efpset = efp.make_efpset(2, beta)
    F = efp.compute_efps(_packed_jet(PT, RAP, PHI), efpset)[0]
    s = np.asarray(efpset.specs)
    d, n, p = s[:, D_COL], s[:, N_COL], s[:, P_COL]
    expected = {
        "edge": ((d == 1), z @ T @ z),
        "double edge": ((d == 2) & (n == 2), z @ (T ** 2) @ z),
        "path": ((d == 2) & (n == 3), ((T @ z) ** 2) @ z),
        "two edges": ((d == 2) & (p == 2), (z @ T @ z) ** 2),
    }
    assert len(F) == 4
    for name, (col, value) in expected.items():
        assert col.sum() == 1, name
        assert F[col][0] == pytest.approx(value, rel=1e-12), name


def _boost_z(P, rapidity):
    out = P.copy()
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    out[..., 0] = ch * P[..., 0] + sh * P[..., 3]
    out[..., 3] = sh * P[..., 0] + ch * P[..., 3]
    return out


def _boost_x(P, rapidity):
    out = P.copy()
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    out[..., 0] = ch * P[..., 0] + sh * P[..., 1]
    out[..., 1] = sh * P[..., 0] + ch * P[..., 1]
    return out


def _rotate_z(P, angle):
    out = P.copy()
    c, s = math.cos(angle), math.sin(angle)
    out[..., 1] = c * P[..., 1] - s * P[..., 2]
    out[..., 2] = s * P[..., 1] + c * P[..., 2]
    return out


def test_efps_are_invariant_under_longitudinal_boosts_and_rotations(ef):
    X = efp.toy_jet_split(20, 32, seed=3).X_train.numpy()      # 14 jets, GeV
    efpset = efp.make_efpset(4, 0.5)
    F0 = efp.compute_efps(X, efpset)
    e1 = F0[:, np.asarray(efpset.specs)[:, D_COL] == 1][:, 0]
    assert F0.shape == (14, 35) and np.isfinite(F0).all()
    assert (e1 > 0).all() and e1.std() > 0.1 * e1.mean()      # not trivial

    def efps_after(transform):
        Xt = X.copy()
        Xt[..., :4] = transform(X[..., :4])                   # padding stays 0
        return efp.compute_efps(Xt, efpset)

    for transform in (lambda P: _boost_z(P, 1.3),
                      lambda P: _rotate_z(P, 2.5),             # across phi = 0
                      lambda P: 3.7 * P,
                      lambda P: _boost_z(_rotate_z(P, -1.9), -0.8)):
        np.testing.assert_allclose(efps_after(transform), F0, rtol=1e-9)
    moved = efps_after(lambda P: _boost_x(P, 0.6))
    assert np.max(np.abs(moved - F0) / np.abs(F0)) > 1e-2


def test_jets_to_events_keeps_real_constituents_only():
    X = np.zeros((2, 4, 5))
    X[0, 0] = [100.0, 60.0, 80.0, 0.0, 1.0]
    X[0, 1] = [50.0, -30.0, 40.0, 0.0, 1.0]
    X[0, 2] = [5.0, 0.0, 0.0, 5.0, 1.0]          # pT = 0: z = 0, y infinite
    events = efp.jets_to_events(X)                # jet 1 has no constituents
    assert [e.shape for e in events] == [(2, 3), (0, 3)]
    np.testing.assert_allclose(events[0][:, 0], [100.0, 50.0])
    assert efp.jets_to_events(np.zeros((0, 4, 5))) == []


def test_parallel_and_chunked_batch_compute_match_serial(ef):
    X = efp.toy_jet_split(40, 32, seed=5).X_train.numpy()      # 28 jets
    X = np.concatenate([X, np.zeros((1, 32, 5))])              # + an empty jet
    efpset = efp.make_efpset(3, 0.5)
    serial = efp.compute_efps(X, efpset)
    assert np.isfinite(serial).all() and not serial[-1].any()  # empty -> zeros
    # chunks of 7 leave a one-jet chunk, which batch_compute returns 1-D
    np.testing.assert_array_equal(efp.compute_efps(X, efpset, chunk=7), serial)
    np.testing.assert_array_equal(
        efp.compute_efps(X, efpset, n_jobs=2, chunk=7), serial)


def test_best_threshold_maximises_validation_accuracy():
    t, acc = efp.best_threshold(np.array([0.1, 0.4, 0.35, 0.8]),
                                np.array([0, 0, 1, 1]))
    assert t == pytest.approx(0.225) and acc == 0.75
    t, acc = efp.best_threshold(np.array([-2.0, -1.0, 1.0, 3.0]),
                                np.array([0, 0, 1, 1]))
    assert t == 0.0 and acc == 1.0
    # Tied scores are never split: the only cuts are below 1.0, between 1.0
    # and 2.0, and above 2.0; the first two tie at 3/4, the lower one wins.
    t, acc = efp.best_threshold(np.array([1.0, 1.0, 2.0, 2.0]),
                                np.array([0, 1, 1, 1]))
    assert t == -math.inf and acc == 0.75


def _args(**kw):
    base = dict(toy_jets=None, canonical_splits=False, max_samples=100_000,
                max_train_samples=None, n_constituents=32)
    return argparse.Namespace(**{**base, **kw})


def test_experiment_and_model_names_follow_run_top_tagging():
    # run_top_tagging's options at their training defaults, plus the data options.
    train_opts = {"norm": "global", "pool": "mean", "cap": "auto",
                  "epochs": 30, "wd": 0.0, "es": None}

    def top_tagging_name(base, k, extra, extra_default):
        key = "ntrain" if base == "top_tagging_canonical" else "n"
        return experiment_tag(base, {**train_opts, "k": k, key: extra},
                              {**train_opts, "k": 32, key: extra_default})

    for k, ntrain in ((32, None), (16, None), (32, 300_000), (16, 300_000)):
        assert efp.experiment_name(_args(canonical_splits=True, n_constituents=k,
                                         max_train_samples=ntrain)) == \
            top_tagging_name("top_tagging_canonical", k, ntrain, None)
    for k, n in ((32, 100_000), (32, 50_000), (24, 50_000)):
        assert efp.experiment_name(_args(n_constituents=k, max_samples=n)) == \
            top_tagging_name("top_tagging_constituents", k, n, 100_000)
    assert efp.experiment_name(_args(canonical_splits=True)) == "top_tagging_canonical"
    assert efp.experiment_name(_args(canonical_splits=True, n_constituents=16,
                                     max_train_samples=150)) == \
        "top_tagging_canonical-k_16-ntrain_150"
    assert efp.experiment_name(_args(max_samples=300)) == "top_tagging_constituents-n_300"
    assert efp.experiment_name(_args(toy_jets=300)) == "toy_jets"
    assert efp.experiment_name(_args(toy_jets=300, n_constituents=16)) == "toy_jets-k_16"

    assert efp.model_name(5, 0.5, "svd", None) == "efp_linear_d5"
    assert efp.model_name(3, 1.0, "svd", None) == "efp_linear_d3-beta_1.0"
    assert efp.model_name(7, 0.5, "lsqr", "auto") == "efp_linear_d7-lda_lsqr-shrink_auto"


def _record(results_dir, stem):
    return json.loads((results_dir / f"{stem}.json").read_text())


def _check_record(r, dmax, n_efps):
    assert SCHEMA_KEYS <= set(r)
    assert r["family"] == "jet_observable" and r["diverged"] is False
    assert r["model"] == f"efp_linear_d{dmax}"
    assert r["n_features"] == n_efps and r["n_params"] == n_efps + 1
    assert set(r["train_metrics"]) == {"final_val_acc", "val_auc"}
    assert set(r["test_metrics"]) == {"test_acc", "test_auc", "bg_rej_30", "bg_rej_50"}
    tm = r["test_metrics"]
    assert math.isfinite(tm["test_auc"]) and tm["test_auc"] > 0.8
    assert r["train_metrics"]["val_auc"] > 0.8
    assert 0.5 < tm["test_acc"] <= 1.0 and 0.5 < r["train_metrics"]["final_val_acc"] <= 1.0
    assert tm["bg_rej_30"] >= tm["bg_rej_50"] > 1.0
    assert r["config"]["dmax"] == dmax and r["config"]["normalize"] == "none"
    assert r["provenance"]["argv"]


def _check_scores(results_dir, stem, r, y_test):
    from sklearn.metrics import roc_auc_score
    z = np.load(results_dir / f"{stem}__scores.npz")
    assert z["scores"].dtype == np.float32 and z["labels"].dtype == np.int8
    assert np.array_equal(z["labels"], np.asarray(y_test))       # same jets, same order
    scores = z["scores"].astype(np.float64)
    assert roc_auc_score(z["labels"], scores) == pytest.approx(
        r["test_metrics"]["test_auc"], abs=1e-9)
    # test accuracy at the threshold chosen on val, not re-optimised on test
    acc = ((scores > r["efp"]["val_threshold"]) == z["labels"]).mean()
    assert acc == pytest.approx(r["test_metrics"]["test_acc"])


def test_runner_end_to_end_on_toy_jets(ef, tmp_path):
    assert efp.main(["--toy-jets", "300", "--dmax", "3", "--save-scores",
                     "--results-dir", str(tmp_path)]) == 0
    stem = "toy_jets__efp_linear_d3__seed0"
    r = _record(tmp_path, stem)
    _check_record(r, dmax=3, n_efps=12)                   # 1 + 3 + 8 EFPs, + bias
    assert r["experiment"] == "toy_jets"
    assert (r["n_train"], r["n_val"], r["n_test"]) == (210, 45, 45)
    assert r["efp"]["per_degree"] == {"1": 1, "2": 3, "3": 8}
    _check_scores(tmp_path, stem, r, efp.toy_jet_split(300, 32, seed=0).y_test)


def _write_kasieczka_file(path, n, seed):
    """Toy massless jets in the layout of top_tagging_<split>.npz."""
    X, y = generate_massless_jets(n, n_constituents=64, seed=seed)
    cons = np.zeros((n, 200, 4), dtype=np.float32)
    cons[:, :64] = X.numpy()
    np.savez(path, constituents=cons, labels=y.numpy())


def test_runner_on_tiny_canonical_files(ef, tmp_path):
    data, res = tmp_path / "data", tmp_path / "results"
    data.mkdir()
    for name, n, seed in (("train", 240, 10), ("val", 80, 11), ("test", 120, 12)):
        _write_kasieczka_file(data / f"top_tagging_{name}.npz", n, seed)
    common = ["--cache-dir", str(data), "--results-dir", str(res), "--dmax", "3"]

    assert efp.main(common + ["--canonical-splits", "--save-scores"]) == 0
    stem = "top_tagging_canonical__efp_linear_d3__seed0"
    r = _record(res, stem)
    _check_record(r, dmax=3, n_efps=12)
    assert (r["n_train"], r["n_val"], r["n_test"]) == (240, 80, 120)
    assert r["dataset"] == "top_tagging_constituents(k=32,norm=none,split=canonical)"
    _check_scores(res, stem, r, np.load(data / "top_tagging_test.npz")["labels"])

    assert efp.main(common + ["--canonical-splits", "--max-train-samples", "100",
                              "--n-constituents", "16"]) == 0
    r = _record(res, "top_tagging_canonical-k_16-ntrain_100__efp_linear_d3__seed0")
    assert (r["n_train"], r["n_val"], r["n_test"]) == (100, 80, 120)

    assert efp.main(common + ["--max-samples", "300"]) == 0       # internal split
    r = _record(res, "top_tagging_constituents-n_300__efp_linear_d3__seed0")
    assert (r["n_train"], r["n_val"], r["n_test"]) == (210, 45, 45)


def test_efp_cache_serves_smaller_dmax_and_rejects_mismatches(ef, tmp_path):
    cache = tmp_path / "efps.npz"
    res, direct = tmp_path / "res", tmp_path / "direct"
    toy = ["--toy-jets", "200"]
    assert efp.main(toy + ["--dmax", "4", "--cache-efps", str(cache),
                           "--results-dir", str(res)]) == 0
    assert cache.is_file()
    assert _record(res, "toy_jets__efp_linear_d4__seed0")["efp"]["cache_hit"] is False

    assert efp.main(toy + ["--dmax", "3", "--cache-efps", str(cache),
                           "--results-dir", str(res)]) == 0
    assert efp.main(toy + ["--dmax", "3", "--results-dir", str(direct)]) == 0
    hit = _record(res, "toy_jets__efp_linear_d3__seed0")
    ref = _record(direct, "toy_jets__efp_linear_d3__seed0")
    assert hit["efp"]["cache_hit"] is True and ref["efp"]["cache_hit"] is False
    assert hit["test_metrics"] == ref["test_metrics"]           # bit-identical features
    assert hit["train_metrics"] == ref["train_metrics"]
    assert hit["n_params"] == ref["n_params"] == 13

    before = cache.read_bytes()
    with pytest.raises(ValueError, match="different setting.*beta"):
        efp.main(toy + ["--dmax", "3", "--beta", "1.0", "--cache-efps", str(cache),
                        "--results-dir", str(res)])
    with pytest.raises(ValueError, match="different setting.*fingerprint"):
        efp.main(toy + ["--dmax", "3", "--seed", "1", "--cache-efps", str(cache),
                        "--results-dir", str(res)])
    with pytest.raises(ValueError, match="d <= 4 only"):
        efp.main(toy + ["--dmax", "5", "--cache-efps", str(cache),
                        "--results-dir", str(res)])
    assert cache.read_bytes() == before                          # left alone


def test_missing_energyflow_fails_with_install_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "energyflow", None)        # import fails
    rc = efp.main(["--toy-jets", "50", "--results-dir", str(tmp_path / "res")])
    assert rc == 2
    assert "pip install energyflow" in capsys.readouterr().err
    assert not (tmp_path / "res").exists()
