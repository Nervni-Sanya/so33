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
from .models import build_model, MATCHED_MODELS, NATURAL_MODELS, SET_MODELS, SO3C_MODELS
from .train import train_classifier, TrainConfig, forward_in_chunks


def evaluate_test(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Return test accuracy, binary AUC, and background rejection.

    For binary tasks we also report background rejection ``1/eps_B`` at
    signal efficiencies ``eps_S in {0.3, 0.5}``. Background rejection at
    fixed signal efficiency is the field-standard headline metric for
    jet tagging (LorentzNet, PELICAN, LGN all report ``1/eps_B`` at
    ``eps_S = 0.3``), so reporting it makes our top-tagging numbers
    directly comparable to the published literature. It is a no-op extra
    for non-tagging tabular tasks (still cheap to compute).
    """
    model.eval()
    # The model may live on CUDA after training while the split is still on
    # CPU: move the test tensors to wherever the parameters are.
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    with torch.no_grad():
        logits = forward_in_chunks(model, X_test, chunk_size)
        pred   = logits.argmax(dim=-1)
        acc    = (pred == y_test).float().mean().item()

    out: dict[str, float] = {"test_acc": acc}
    if n_classes == 2:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            y_true = y_test.cpu().numpy()
            scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            # Stash the raw scores so callers can persist them: every
            # post-hoc analysis (binned performance, ROC, rejection curves)
            # is then free, instead of costing a full retrain per figure.
            out["_scores"] = scores
            out["_labels"] = y_true
            out["test_auc"] = float(roc_auc_score(y_true, scores))
            # Background rejection 1/eps_B at fixed signal efficiency eps_S.
            fpr, tpr, _ = roc_curve(y_true, scores)   # tpr ascending
            for eff, key in ((0.3, "bg_rej_30"), (0.5, "bg_rej_50")):
                # First operating point that reaches the target signal
                # efficiency; its false-positive rate is eps_B.
                idx = int((tpr >= eff).argmax())
                eps_b = float(fpr[idx])
                out[key] = float("inf") if eps_b <= 0.0 else 1.0 / eps_b
        except Exception as e:
            # Do NOT swallow silently: a missing test_auc in the results
            # JSON has historically hidden real failures (NaN logits in the
            # so33_equivariant runs went unnoticed for weeks).
            print(f"[evaluate_test] WARNING: AUC/rejection failed: {e!r}")
    return out


def run_tabular_experiment(
    experiment: str,
    split: DatasetSplit,
    *,
    models: Iterable[str] | None = None,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-3,
    natural_hidden: int = 256,
    T: float = 0.3,
    representation: str = "flat",
    pool: str = "mean",
    results_dir: pathlib.Path | str = "results",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    so3c_kwargs: dict | None = None,
    eval_chunk_size: int = 4096,
    ckpt_dir: pathlib.Path | str | None = None,
    resume: bool = False,
    max_seconds: float | None = None,
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

    if models is None:
        # SET_MODELS need per-particle (B, K, 5) input — only valid for the
        # constituents path. Skip them for flat tabular experiments
        # (HIGGS, Adult, top_tagging aggregated).
        models = MATCHED_MODELS + NATURAL_MODELS
        if representation == "constituents":
            models = models + SET_MODELS

    summary: list[dict] = []
    for name in models:
        if name in SET_MODELS and representation != "constituents":
            print(f"[{experiment}] SKIP {name}: requires representation=constituents")
            continue
        if name in SO3C_MODELS and representation != "flat":
            print(f"[{experiment}] SKIP {name}: so3c models support representation=flat only")
            continue
        if name in SET_MODELS:
            family = "equivariant_set"
        elif name in NATURAL_MODELS:
            family = "natural_width"
        else:
            family = "matched_bottleneck"
        print(f"[{experiment}] {name} ({family}) | train ...")

        out_path = out_dir / f"{experiment}__{name}__seed{seed}.json"
        if resume and out_path.is_file():
            # A restarted session (Kaggle caps sessions at 12h) should pick up
            # at the first unfinished model rather than redo completed ones.
            print(f"[{experiment}] SKIP {name}: {out_path.name} already exists")
            summary.append(json.loads(out_path.read_text()))
            continue

        # Seed BEFORE constructing the model: nn.Linear draws its weights from
        # the global RNG, and train_classifier only seeds afterwards, so until
        # this line every run started from a different initialisation and two
        # identical commands could not reproduce each other.
        torch.manual_seed(seed)
        model = build_model(
            name,
            in_features  = split.n_features,
            out_features = split.n_classes,
            T = T,
            natural_hidden = natural_hidden,
            representation = representation,
            pool = pool,
            dtype = dtype,
            so3c_kwargs = so3c_kwargs,
        )
        ckpt_path = (pathlib.Path(ckpt_dir) / f"{experiment}__{name}__seed{seed}.pt"
                     if ckpt_dir else None)
        cfg = TrainConfig(
            epochs=epochs, batch_size=batch_size, lr=lr,
            seed=seed, cosine_schedule=True, grad_clip=1.0,
            device=device, eval_chunk_size=eval_chunk_size,
            ckpt_path=str(ckpt_path) if ckpt_path else None,
            resume=resume, max_seconds=max_seconds,
        )
        train_res = train_classifier(
            model, split.X_train, split.y_train, split.X_val, split.y_val, cfg,
        )
        test_res = evaluate_test(model, split.X_test, split.y_test,
                                 split.n_classes, eval_chunk_size)

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
            "test_metrics":  {k: v for k, v in test_res.items()
                              if not k.startswith("_")},
            "config":        asdict(cfg),
            "history":       train_res.history,
        }
        out_path.write_text(json.dumps(record, indent=2))

        # Per-jet test scores go beside the JSON as a compact npz (~1.6 MB
        # for a 404k-jet canonical test split).
        if "_scores" in test_res:
            import numpy as np
            np.savez_compressed(
                out_dir / f"{experiment}__{name}__seed{seed}__scores.npz",
                scores=test_res["_scores"].astype(np.float32),
                labels=test_res["_labels"].astype(np.int8),
            )

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
        rej = test_res.get("bg_rej_30")
        rej_s = f" 1/eB@0.3={rej:.0f}" if rej is not None else ""
        print(f"   val_acc={line['val_acc']:.3f}  test_acc={line['test_acc']:.3f}"
              f"{auc_s}{rej_s}  ({line['walltime']:.1f}s)")

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
