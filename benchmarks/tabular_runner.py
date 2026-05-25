"""
benchmarks.tabular_runner
-------------------------
Shared driver for tabular real-data experiments.

Trains every model variant on a DatasetSplit, evaluates on the test
split, and writes one JSON per (model, seed) into ``results_dir``.
Reports two tables side-by-side:

- ``matched_bottleneck``: every model uses Linear -> hidden=6 ->
  activation -> Linear. Apples-to-apples comparison; isolates whether
  the geometry helps at equal compression.
- ``natural_width``: MLPs use a wide hidden layer (default 256) while
  SO33 stays at 6. Apples-to-oranges; tests whether SO33 is competitive
  in practice given its hard 6-dim bottleneck.

Used by the per-dataset runners (run_higgs, run_top_tagging, run_neutral)
which only differ in their dataset loader.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from typing import Iterable

import torch
import torch.nn as nn

from .datasets import DatasetSplit
from .models import build_model, MATCHED_MODELS, NATURAL_MODELS
from .train import train_classifier, TrainConfig


def evaluate_test(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
) -> dict[str, float]:
    """Return test accuracy and binary AUC (when applicable)."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        pred   = logits.argmax(dim=-1)
        acc    = (pred == y_test).float().mean().item()

    out: dict[str, float] = {"test_acc": acc}
    if n_classes == 2:
        try:
            from sklearn.metrics import roc_auc_score
            scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            out["test_auc"] = float(roc_auc_score(y_test.cpu().numpy(), scores))
        except Exception:
            pass
    return out


def run_tabular_experiment(
    experiment: str,
    split: DatasetSplit,
    *,
    models: Iterable[str] = MATCHED_MODELS + NATURAL_MODELS,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-3,
    natural_hidden: int = 256,
    T: float = 0.3,
    representation: str = "flat",
    results_dir: pathlib.Path | str = "results",
) -> list[dict]:
    """Train a set of model variants on ``split`` and write one JSON per run.

    Returns
    -------
    summary : list of {model, family, n_params, val_acc, test_acc, test_auc?, walltime}
              records, one per (model, seed). Also printed as a table.
    """
    out_dir = pathlib.Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{experiment}] {split.summary()}")

    summary: list[dict] = []
    for name in models:
        family = "natural_width" if name in NATURAL_MODELS else "matched_bottleneck"
        print(f"[{experiment}] {name} ({family}) | train ...")

        model = build_model(
            name,
            in_features  = split.n_features,
            out_features = split.n_classes,
            T = T,
            natural_hidden = natural_hidden,
            representation = representation,
        )
        cfg = TrainConfig(
            epochs=epochs, batch_size=batch_size, lr=lr,
            seed=seed, cosine_schedule=True, grad_clip=1.0,
        )
        train_res = train_classifier(
            model, split.X_train, split.y_train, split.X_val, split.y_val, cfg,
        )
        test_res = evaluate_test(model, split.X_test, split.y_test, split.n_classes)

        record = {
            "experiment":    experiment,
            "dataset":       split.name,
            "model":         name,
            "family":        family,
            "seed":          seed,
            "n_params":      train_res.n_params,
            "n_train":       len(split.X_train),
            "n_val":         len(split.X_val),
            "n_test":        len(split.X_test),
            "n_features":    split.n_features,
            "walltime_sec":  train_res.walltime_sec,
            "epochs_run":    train_res.epochs_run,
            "train_metrics": {
                "final_train_acc": train_res.final_train_acc,
                "final_val_acc":   train_res.final_val_acc,
                "best_val_acc":    train_res.best_val_acc,
            },
            "test_metrics":  test_res,
            "config":        asdict(cfg),
        }
        out_path = out_dir / f"{experiment}__{name}__seed{seed}.json"
        out_path.write_text(json.dumps(record, indent=2))

        line = {
            "model":     name,
            "family":    family,
            "n_params":  train_res.n_params,
            "val_acc":   train_res.final_val_acc,
            "test_acc":  test_res["test_acc"],
            "test_auc":  test_res.get("test_auc"),
            "walltime":  train_res.walltime_sec,
        }
        summary.append(line)

        auc_s = f" auc={line['test_auc']:.3f}" if line["test_auc"] is not None else ""
        print(f"   val_acc={line['val_acc']:.3f}  test_acc={line['test_acc']:.3f}"
              f"{auc_s}  ({line['walltime']:.1f}s)")

    _print_summary(experiment, summary)
    return summary


def _print_summary(experiment: str, summary: list[dict]) -> None:
    has_auc = any(r["test_auc"] is not None for r in summary)

    def fmt_table(rows: list[dict], heading: str) -> None:
        if not rows:
            return
        print(f"\n[{experiment}] {heading}")
        cols = "params  val_acc  test_acc"
        if has_auc:
            cols += "  test_auc"
        print(f"{'model':<24} {cols}")
        print("-" * (25 + len(cols)))
        for r in rows:
            line = (f"{r['model']:<24} {r['n_params']:>6}  "
                    f"{r['val_acc']:>7.3f}  {r['test_acc']:>8.3f}")
            if has_auc:
                line += f"  {r['test_auc']:>8.3f}" if r["test_auc"] is not None else "       —"
            print(line)

    matched = [r for r in summary if r["family"] == "matched_bottleneck"]
    natural = [r for r in summary if r["family"] == "natural_width"]
    fmt_table(matched, "matched bottleneck (hidden=6)")
    fmt_table(natural, "natural width MLPs")
