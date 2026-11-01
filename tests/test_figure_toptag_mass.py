"""
Test 14 — benchmarks.figure_toptag_mass (how much of a tagger is the jet mass).

Verifies the jet-mass computation on a hand-checkable configuration, and
the script end to end on tiny canonical npz files: the per-jet score file
must line up with the canonical test split (it refuses a mismatch), and
the JSON summary carries inclusive AUC / 1/eps_B and per-mass-bin AUCs for
the score file, the jet mass and the mass window.

Run:
    python -m pytest tests/test_figure_toptag_mass.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from benchmarks.figure_toptag_mass import jet_mass, main


def test_jet_mass_of_back_to_back_massless_pair():
    p4 = np.array([[[50.0, 0.0, 0.0, 50.0], [50.0, 0.0, 0.0, -50.0], [0.0] * 4]])
    mask = np.array([[1.0, 1.0, 0.0]])
    assert jet_mass(p4, mask)[0] == pytest.approx(100.0)


def _make_npz(path, n, seed):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) > 0.5).astype(np.int64)
    cons = np.zeros((n, 200, 4), dtype=np.float32)
    for i in range(n):
        base = rng.normal(size=(5, 4)).astype(np.float32) * 10.0
        base[:, 0] = np.abs(base[:, 0]) + 50.0 + labels[i] * 15.0
        cons[i, :5] = base
    np.savez(path, constituents=cons, labels=labels)
    return labels


def test_script_end_to_end(tmp_path):
    _make_npz(tmp_path / "top_tagging_train.npz", 50, 0)
    _make_npz(tmp_path / "top_tagging_val.npz", 30, 1)
    labels = _make_npz(tmp_path / "top_tagging_test.npz", 400, 2)
    rng = np.random.default_rng(3)
    scores = labels + rng.normal(size=len(labels))
    good = tmp_path / "good__scores.npz"
    np.savez(good, scores=scores.astype(np.float32), labels=labels.astype(np.int8))
    out = tmp_path / "fig"
    args = ["--cache-dir", str(tmp_path), "--canonical-splits", "--n-constituents", "16",
            "--out-dir", str(out), "--published", "Some tagger:1000"]
    assert main(args + ["--scores", f"model={good}"]) == 0
    assert (out / "fig_toptag_mass_dominance.pdf").exists()
    summary = json.loads((out / "toptag_mass_dominance.json").read_text())
    assert summary["n_test"] == 400
    assert set(summary["curves"]) == {"model", "jet mass $M$", r"mass window $-|M-172.5|$"}
    model = summary["curves"]["model"]
    assert 0.5 < model["test_auc"] <= 1.0 and len(model["binned_auc"]) == 6
    bad = tmp_path / "bad__scores.npz"
    np.savez(bad, scores=scores[::-1].astype(np.float32), labels=labels[::-1].astype(np.int8))
    assert main(args + ["--scores", f"model={bad}"]) == 1
