"""
Test — harness plumbing that had no coverage at all.

Everything added during the GPU-port phase (checkpoint/resume, capacity
kwargs, device handling, the figure scripts) was verified by hand from the
CLI and then left untested. These are the regressions for it, plus a
regression for the seeding bug that made every run irreproducible.

Run:
    python -m pytest tests/test_harness.py -v
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn

from benchmarks.datasets import DatasetSplit
from benchmarks.models import build_model
from benchmarks.tabular_runner import evaluate_test, run_tabular_experiment
from benchmarks.train import TrainConfig, forward_in_chunks, train_classifier


# ── helpers ───────────────────────────────────────────────────────────────

def _tiny_split(n: int = 240, k: int = 6, seed: int = 0) -> DatasetSplit:
    """A small separable constituent dataset, so training actually moves."""
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=g) > 0.5).long()
    p = torch.randn(n, k, 3, generator=g) * (1.0 + 0.6 * y[:, None, None])
    m = torch.rand(n, k, generator=g) * 0.1
    E = torch.sqrt(m * m + p.pow(2).sum(-1))
    X = torch.cat([E.unsqueeze(-1), p, torch.ones(n, k, 1)], dim=-1).float()
    a, b = int(0.6 * n), int(0.8 * n)
    return DatasetSplit(X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:],
                        name="tiny", n_features=4, n_classes=2)


def _small_model(dtype=torch.float64) -> nn.Module:
    torch.manual_seed(0)
    return build_model("so3c_invariant_set", in_features=4, out_features=2,
                       representation="constituents", dtype=dtype)


# ── capacity plumbing (so3c_kwargs) ───────────────────────────────────────

@pytest.mark.parametrize("kwargs,expected", [
    (None, 9056),
    ({"channels": 8, "hidden": 128, "act_hidden": 32}, 38584),
    ({"channels": 32, "hidden": 128, "act_hidden": 32}, 198376),
])
def test_so3c_kwargs_sets_capacity(kwargs, expected) -> None:
    """--channels/--hidden must actually reach the model, not be dropped."""
    m = build_model("so3c_equivariant_set", in_features=4, out_features=2,
                    representation="constituents", so3c_kwargs=kwargs)
    assert sum(p.numel() for p in m.parameters()) == expected


def test_so3c_kwargs_pruned_for_models_without_those_knobs() -> None:
    """The no-flow and interaction models have no channel axis; passing one
    must be dropped silently rather than raising TypeError."""
    for name in ("so3c_invariant_set", "so3c_interaction_set"):
        m = build_model(name, in_features=4, out_features=2,
                        representation="constituents",
                        so3c_kwargs={"channels": 16, "hidden": 32,
                                     "act_hidden": 8, "T": 2.0})
        assert m(_tiny_split().X_train[:4]).shape == (4, 2)


# ── device handling ───────────────────────────────────────────────────────

def test_evaluate_test_moves_data_to_model_device() -> None:
    """evaluate_test must follow the model's device. Before the fix it fed
    CPU tensors to a CUDA model and died on the first Linear."""
    split = _tiny_split()
    model = _small_model()
    res = evaluate_test(model, split.X_test, split.y_test, 2)
    assert "test_acc" in res and 0.0 <= res["test_acc"] <= 1.0
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA here")
def test_evaluate_test_cuda_roundtrip() -> None:
    split = _tiny_split()
    model = _small_model().cuda()
    res = evaluate_test(model, split.X_test, split.y_test, 2)   # data on CPU
    assert 0.0 <= res["test_acc"] <= 1.0


def test_forward_in_chunks_matches_full_batch() -> None:
    split = _tiny_split(n=200)
    model = _small_model()
    model.eval()
    with torch.no_grad():
        full = model(split.X_train)
        chunked = forward_in_chunks(model, split.X_train, chunk_size=16)
    assert torch.allclose(full, chunked, atol=1e-12)


# ── checkpoint / resume ───────────────────────────────────────────────────

def test_resume_reproduces_continuous_training(tmp_path) -> None:
    """An interrupted+resumed run must equal an uninterrupted one exactly.

    This is the property the Kaggle 12h cap depends on. It only holds
    because the RNG state is part of the checkpoint: the batch order comes
    from torch.randperm, so a resume without it replays a different shuffle.
    """
    split = _tiny_split()
    epochs = 6

    torch.manual_seed(0)
    cont = train_classifier(_small_model(), split.X_train, split.y_train,
                            split.X_val, split.y_val,
                            TrainConfig(epochs=epochs, batch_size=32, seed=0))

    ckpt = tmp_path / "run.pt"
    torch.manual_seed(0)
    part = train_classifier(
        _small_model(), split.X_train, split.y_train, split.X_val, split.y_val,
        TrainConfig(epochs=epochs, batch_size=32, seed=0,
                    ckpt_path=str(ckpt), max_seconds=0.0))
    assert ckpt.is_file(), "checkpoint was not written"
    assert part.epochs_run < epochs, "max_seconds did not interrupt training"

    torch.manual_seed(0)
    resumed = train_classifier(
        _small_model(), split.X_train, split.y_train, split.X_val, split.y_val,
        TrainConfig(epochs=epochs, batch_size=32, seed=0,
                    ckpt_path=str(ckpt), resume=True))

    assert resumed.epochs_run == cont.epochs_run == epochs
    assert len(resumed.history) == len(cont.history) == epochs
    for a, b in zip(cont.history, resumed.history):
        assert a["val_acc"] == b["val_acc"], f"diverged at epoch {a['epoch']}"
        assert a["train_loss"] == b["train_loss"]
    assert resumed.final_val_acc == cont.final_val_acc


def test_resume_accumulates_walltime(tmp_path) -> None:
    """walltime must carry across a resume, not restart from zero."""
    split = _tiny_split()
    ckpt = tmp_path / "wt.pt"
    part = train_classifier(
        _small_model(), split.X_train, split.y_train, split.X_val, split.y_val,
        TrainConfig(epochs=6, batch_size=32, seed=0,
                    ckpt_path=str(ckpt), max_seconds=0.0))
    resumed = train_classifier(
        _small_model(), split.X_train, split.y_train, split.X_val, split.y_val,
        TrainConfig(epochs=6, batch_size=32, seed=0,
                    ckpt_path=str(ckpt), resume=True))
    assert resumed.walltime_sec > part.walltime_sec


# ── reproducibility ───────────────────────────────────────────────────────

def test_identical_runs_are_identical(tmp_path) -> None:
    """Regression: the model is built before train_classifier seeds, so for
    a long time nn.Linear drew from an unseeded RNG and no two runs agreed.
    run_tabular_experiment now seeds before build_model."""
    split = _tiny_split()
    out = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        run_tabular_experiment("t", split, models=["so3c_invariant_set"],
                               seed=0, epochs=3, batch_size=32,
                               representation="constituents", results_dir=d)
        rec = json.loads(
            (d / "t__so3c_invariant_set__seed0.json").read_text())
        out.append(rec["test_metrics"]["test_acc"])
    assert out[0] == out[1], f"non-deterministic: {out}"


# ── figure scripts ────────────────────────────────────────────────────────

def _fake_results(d: pathlib.Path, experiment: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(("so3c_equivariant_set", "eta_invariants")):
        (d / f"{experiment}__{model}__seed0.json").write_text(json.dumps({
            "experiment": experiment, "model": model, "seed": 0,
            "n_params": 9056 - i * 4000, "family": "equivariant_set",
            "walltime_sec": 10.0, "epochs_run": 2,
            "history": [{"epoch": e, "train_loss": 0.5 / e, "train_acc": 0.8,
                         "val_loss": 0.5, "val_acc": 0.8 + 0.01 * e}
                        for e in (1, 2)],
            "train_metrics": {"final_val_acc": 0.82, "best_val_acc": 0.82,
                              "final_train_acc": 0.8},
            "test_metrics": {"test_acc": 0.8, "test_auc": 0.95 - 0.02 * i,
                             "bg_rej_30": 100 - 40 * i, "bg_rej_50": 50},
        }))


def test_figure_pareto_runs(tmp_path) -> None:
    import benchmarks.figure_pareto as fp
    _fake_results(tmp_path / "res", "top_tagging_canonical")
    rc = fp.main(["--results-dir", str(tmp_path / "res"),
                  "--out-dir", str(tmp_path / "fig")])
    assert rc == 0
    assert (tmp_path / "fig" / "pareto_data.csv").is_file()


def test_figure_convergence_runs(tmp_path) -> None:
    import benchmarks.figure_convergence as fc
    _fake_results(tmp_path / "res", "top_tagging_constituents")
    rc = fc.main(["--results-dir", str(tmp_path / "res"),
                  "--out-dir", str(tmp_path / "fig")])
    assert rc == 0
    assert (tmp_path / "fig" / "convergence_val_acc.csv").is_file()


def test_figure_scaling_runs(tmp_path) -> None:
    import benchmarks.figure_scaling as fs
    root = tmp_path / "scal"
    for axis, knobs in (("channels", (4, 8)), ("width", (64, 128))):
        for k in knobs:
            _fake_results(root / f"{axis}_{k}", "top_tagging_canonical")
    rc = fs.main(["--results-dir", str(root), "--out-dir", str(tmp_path / "fig")])
    assert rc == 0
    assert (tmp_path / "fig" / "scaling.csv").is_file()


def test_max_seconds_caps_the_session_not_the_run(tmp_path) -> None:
    """A run resumed after it passed max_seconds must still make progress.

    max_seconds exists for Kaggle's per-session cap, so it has to be compared
    with the time spent in THIS session. It was compared with the cumulative
    clock restored from the checkpoint instead, and then a run already past
    the cap trained one epoch per session and stopped again: the two K=64
    message-passing runs that halted at 26026 s and 26068 s against a
    26000 s cap could never have reached epoch 30.
    """
    import torch
    import torch.nn as nn
    from benchmarks.train import TrainConfig, train_classifier

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2))

        def forward(self, x):
            return self.net(x)

        def regularization_loss(self):
            return torch.zeros(())

    torch.manual_seed(0)
    X = torch.randn(256, 6)
    y = (X[:, 0] > 0).long()
    ckpt = tmp_path / "session.pt"

    part = train_classifier(Tiny(), X, y, X, y, TrainConfig(
        epochs=4, batch_size=64, ckpt_path=str(ckpt), max_seconds=0.0))
    assert part.epochs_run == 1

    state = torch.load(ckpt, weights_only=False)
    state["walltime_sec"] = 1e6          # far past the cap, as after a long session
    torch.save(state, ckpt)

    resumed = train_classifier(Tiny(), X, y, X, y, TrainConfig(
        epochs=4, batch_size=64, ckpt_path=str(ckpt), resume=True,
        max_seconds=1e5))
    assert resumed.epochs_run == 4, (
        "resumed session stopped at epoch %d: max_seconds was compared with "
        "the cumulative clock, not this session's" % resumed.epochs_run)
    assert resumed.walltime_sec > 1e6    # the recorded clock still accumulates


def test_cuda_resume_restores_rng_state(tmp_path) -> None:
    """Resuming on a GPU must get past restoring the saved RNG state.

    train_classifier loaded checkpoints with map_location=device, which put
    the saved CPU and CUDA RNG states on the GPU as well. torch.set_rng_state
    and torch.cuda.set_rng_state both require a CPU ByteTensor, so every GPU
    resume died with "RNG state must be a torch.ByteTensor" before a single
    step -- the K=64 finish kernel lost both seeds that way. On CPU,
    map_location is already the CPU, so only a CUDA test can see this.
    """
    import pytest
    import torch
    import torch.nn as nn
    from benchmarks.train import TrainConfig, train_classifier

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: the failure only exists when the RNG state lands on the GPU")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2))

        def forward(self, x):
            return self.net(x)

        def regularization_loss(self):
            return torch.zeros((), device=self.net[0].weight.device)

    torch.manual_seed(0)
    X = torch.randn(256, 6)
    y = (X[:, 0] > 0).long()
    ckpt = tmp_path / "cuda_resume.pt"

    part = train_classifier(Tiny(), X, y, X, y, TrainConfig(
        epochs=3, batch_size=64, device="cuda", ckpt_path=str(ckpt),
        max_seconds=0.0))
    assert part.epochs_run == 1

    resumed = train_classifier(Tiny(), X, y, X, y, TrainConfig(
        epochs=3, batch_size=64, device="cuda", ckpt_path=str(ckpt),
        resume=True))
    assert resumed.epochs_run == 3
    assert len(resumed.history) == 3
