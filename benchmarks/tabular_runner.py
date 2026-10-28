"""
benchmarks.tabular_runner
-------------------------
Shared driver for tabular real-data experiments.

Trains every model variant on a DatasetSplit, evaluates on the test
split, and writes one JSON per (model, seed) into ``results_dir``.
Reports up to three tables side-by-side:

- ``matched_bottleneck``: every model uses Linear -> hidden=6 ->
  activation -> Linear. Apples-to-apples comparison; isolates whether
  the geometry helps at equal compression.
- ``natural_width``: MLPs use a wide hidden layer (default 256, or the
  width in a ``<act>_mlp_w<H>`` name) while SO33 stays at 6.
  Apples-to-oranges; tests whether SO33 is competitive in practice given
  its hard 6-dim bottleneck.
- ``equivariant_set``: the per-constituent invariant / equivariant models
  (Architectures A and B), constituents representation only.

Each JSON carries the run's provenance (git commit, versions, command
line), the training-guard outcome (``diverged``), and, for models with SO33
activations, the trained connection norms and caps. A run whose logits
are non-finite is recorded as diverged, with no AUC.

Used by the per-dataset runners (run_higgs, run_top_tagging, run_neutral)
which only differ in their dataset loader.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from dataclasses import asdict
from typing import Any, Iterable

import torch
import torch.nn as nn

from .datasets import DatasetSplit
from .equivariance import model_equivariance
from .models import (
    build_model, max_constituent_norm, model_family, is_set_model,
    MATCHED_MODELS, NATURAL_MODELS, SET_MODELS,
)
from .provenance import run_metadata
from .train import train_classifier, TrainConfig, so33_activations


def evaluate_test(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
    return_scores: bool = False,
):
    """Return test accuracy, binary AUC, and background rejection.

    For binary tasks we also report background rejection ``1/eps_B`` at
    signal efficiencies ``eps_S in {0.3, 0.5}``. Background rejection at
    fixed signal efficiency is the field-standard headline metric for
    jet tagging (LorentzNet, PELICAN, LGN all report ``1/eps_B`` at
    ``eps_S = 0.3``), so reporting it makes our top-tagging numbers
    directly comparable to the published literature. It is a no-op extra
    for non-tagging tabular tasks (still cheap to compute).

    If any logit is non-finite the model has diverged: the result carries
    ``diverged=True`` and ``nonfinite_frac``, non-finite rows count as wrong
    in ``test_acc``, and AUC / background rejection are not reported (an
    argmax over NaN logits would otherwise look like chance accuracy).

    With ``return_scores=True`` returns ``(metrics, scores)``, where
    ``scores`` is the per-sample signal probability (binary tasks) or None.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
    finite = torch.isfinite(logits).all(dim=-1)
    nonfinite_frac = 1.0 - finite.double().mean().item()
    pred = logits.argmax(dim=-1)
    acc = ((pred == y_test) & finite).float().mean().item()

    out: dict[str, Any] = {"test_acc": acc}
    scores = None
    if nonfinite_frac > 0.0:
        out["nonfinite_frac"] = nonfinite_frac
        out["diverged"] = True
        warnings.warn(
            f"{nonfinite_frac:.2%} of the test logits are non-finite: the "
            f"model diverged, so test_auc and bg_rej_* are not reported.",
            RuntimeWarning, stacklevel=2,
        )
        return (out, None) if return_scores else out

    if n_classes == 2:
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            y_true = y_test.cpu().numpy()
            out["test_auc"] = float(roc_auc_score(y_true, scores))
            # Background rejection 1/eps_B at fixed signal efficiency eps_S.
            fpr, tpr, _ = roc_curve(y_true, scores)   # tpr ascending
            for eff, key in ((0.3, "bg_rej_30"), (0.5, "bg_rej_50")):
                # First operating point that reaches the target signal
                # efficiency; its false-positive rate is eps_B.
                idx = int((tpr >= eff).argmax())
                eps_b = float(fpr[idx])
                out[key] = float("inf") if eps_b <= 0.0 else 1.0 / eps_b
        except ImportError:
            warnings.warn(
                "scikit-learn is not installed: test_auc and bg_rej_* are "
                "omitted from the results. Install it with "
                "`pip install scikit-learn`.",
                RuntimeWarning, stacklevel=2,
            )
        except Exception as e:
            warnings.warn(
                f"could not compute test_auc/bg_rej_*: "
                f"{type(e).__name__}: {e}",
                RuntimeWarning, stacklevel=2,
            )
    return (out, scores) if return_scores else out


def connection_cap_kwargs(mode: str | float | None, r_max: float | None) -> dict:
    """build_model kwargs for the Architecture-B connection cap.

    ``"auto"`` derives a provably blow-up-free cap from the largest
    training-constituent norm ``r_max``; ``"off"``/None keeps the legacy
    cap of 1.0; a number is used as the cap itself.
    """
    if mode is None or mode == "off":
        return {}
    if mode == "auto":
        return {"input_norm_bound": r_max} if r_max else {}
    return {"connection_cap": float(mode)}


def so33_summary(model: nn.Module) -> dict | None:
    """Trained connection of every SO33 activation (None if the model has none)."""
    acts = so33_activations(model)
    if not acts:
        return None
    return {
        "coeff_norms": [a.coeff_norm() for a in acts],
        "effective_connection_norms": [a.effective_connection_norm() for a in acts],
        "connection_caps": [a.connection_cap for a in acts],
        "coeffs": [a.coeffs.detach().cpu().tolist() for a in acts],
    }


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
    connection_cap: str | float | None = "auto",
    weight_decay: float = 0.0,
    early_stop_patience: int | None = None,
    save_scores: bool = False,
    equivariance_probe: bool = True,
    run_meta: dict | None = None,
) -> list[dict]:
    """Train a set of model variants on ``split`` and write one JSON per run.

    ``connection_cap`` ("auto" | "off" | float) sets the Architecture-B
    connection cap on the constituents path (see ``connection_cap_kwargs``).
    ``save_scores`` also writes the per-sample test scores and labels to
    ``<experiment>__<model>__seed<N>__scores.npz``. ``equivariance_probe``
    records eps_sigma / eps_m of Architecture-B activations at their trained
    connections on a sample of test constituents.

    Returns
    -------
    summary : list of {model, family, n_params, val_acc, test_acc, test_auc?, walltime}
              records, one per (model, seed). Also printed as a table.
    """
    out_dir = pathlib.Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = run_meta if run_meta is not None else run_metadata()

    print(f"[{experiment}] {split.summary()}")

    if models is None:
        # SET_MODELS need per-particle (B, K, 5) input — only valid for the
        # constituents path. Skip them for flat tabular experiments
        # (HIGGS, Adult, top_tagging aggregated).
        models = MATCHED_MODELS + NATURAL_MODELS
        if representation == "constituents":
            models = models + SET_MODELS

    stability: dict[str, Any] | None = None
    cap_kwargs: dict = {}
    if representation == "constituents":
        r_max = max_constituent_norm(split.X_train)
        cap_kwargs = connection_cap_kwargs(connection_cap, r_max)
        stability = {"connection_cap_mode": connection_cap,
                     "max_train_constituent_norm": r_max}
        print(f"[{experiment}] stability: max train-constituent norm = {r_max:.2f}, "
              f"Architecture-B connection cap mode = {connection_cap!r}")

    summary: list[dict] = []
    for name in models:
        family = model_family(name)
        if is_set_model(name) and representation != "constituents":
            print(f"[{experiment}] SKIP {name}: requires representation=constituents")
            continue
        print(f"[{experiment}] {name} ({family}) | train ...")

        model = build_model(
            name,
            in_features  = split.n_features,
            out_features = split.n_classes,
            T = T,
            natural_hidden = natural_hidden,
            representation = representation,
            pool = pool,
            **cap_kwargs,
        )
        cfg = TrainConfig(
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, early_stop_patience=early_stop_patience,
            seed=seed, cosine_schedule=True, grad_clip=1.0,
        )
        train_res = train_classifier(
            model, split.X_train, split.y_train, split.X_val, split.y_val, cfg,
        )
        test_res, scores = evaluate_test(
            model, split.X_test, split.y_test, split.n_classes, return_scores=True,
        )
        diverged = bool(train_res.diverged or test_res.get("diverged", False))

        record: dict[str, Any] = {
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
            "diverged":      diverged,
            "train_metrics": {
                "final_train_acc": train_res.final_train_acc,
                "final_val_acc":   train_res.final_val_acc,
                "best_val_acc":    train_res.best_val_acc,
                "final_val_loss":  train_res.final_val_loss,
                "best_val_loss":   train_res.best_val_loss,
                "best_epoch":      train_res.best_epoch,
                "restored_best":   train_res.restored_best,
            },
            "train_guard": {
                "nonfinite_steps": train_res.nonfinite_steps,
                "diverged":        train_res.diverged,
                "first_nonfinite": train_res.first_nonfinite,
            },
            "test_metrics":  test_res,
            "config":        asdict(cfg),
            "history":       train_res.history,
            "provenance":    meta,
        }
        so33 = so33_summary(model)
        if so33 is not None:
            record["so33"] = so33
        if stability is not None:
            record["stability"] = stability
        if (equivariance_probe and representation == "constituents"
                and not diverged and so33 is not None):
            eq = model_equivariance(model, split.X_test[:2000], seed=seed)
            if eq is not None:
                record["equivariance"] = eq
        out_path = out_dir / f"{experiment}__{name}__seed{seed}.json"
        out_path.write_text(json.dumps(record, indent=2))
        if save_scores and scores is not None:
            import numpy as np
            np.savez_compressed(
                out_dir / f"{experiment}__{name}__seed{seed}__scores.npz",
                scores=scores.astype(np.float32),
                labels=split.y_test.cpu().numpy().astype(np.int8),
            )

        line = {
            "model":     name,
            "family":    family,
            "n_params":  train_res.n_params,
            "val_acc":   train_res.final_val_acc,
            "test_acc":  test_res["test_acc"],
            "test_auc":  test_res.get("test_auc"),
            "bg_rej_30": test_res.get("bg_rej_30"),
            "walltime":  train_res.walltime_sec,
            "diverged":  diverged,
        }
        summary.append(line)

        auc_s = f" auc={line['test_auc']:.3f}" if line["test_auc"] is not None else ""
        rej = test_res.get("bg_rej_30")
        rej_s = f" 1/eB@0.3={rej:.0f}" if rej is not None else ""
        div_s = "  DIVERGED" if diverged else ""
        print(f"   val_acc={line['val_acc']:.3f}  test_acc={line['test_acc']:.3f}"
              f"{auc_s}{rej_s}  ({line['walltime']:.1f}s){div_s}")

    _print_summary(experiment, summary)
    return summary


def _print_summary(experiment: str, summary: list[dict]) -> None:
    has_auc = any(r["test_auc"] is not None for r in summary)
    has_rej = any(r.get("bg_rej_30") is not None for r in summary)

    def fmt_table(rows: list[dict], heading: str) -> None:
        if not rows:
            return
        print(f"\n[{experiment}] {heading}")
        cols = "params  val_acc  test_acc"
        if has_auc:
            cols += "  test_auc"
        if has_rej:
            cols += "  1/eB@0.3"
        print(f"{'model':<24} {cols}")
        print("-" * (25 + len(cols)))
        for r in rows:
            line = (f"{r['model']:<24} {r['n_params']:>6}  "
                    f"{r['val_acc']:>7.3f}  {r['test_acc']:>8.3f}")
            if has_auc:
                line += f"  {r['test_auc']:>8.3f}" if r["test_auc"] is not None else "       —"
            if has_rej:
                rej = r.get("bg_rej_30")
                line += f"  {rej:>8.0f}" if rej is not None else "         —"
            if r.get("diverged"):
                line += "  diverged"
            print(line)

    matched = [r for r in summary if r["family"] == "matched_bottleneck"]
    natural = [r for r in summary if r["family"] == "natural_width"]
    equivariant = [r for r in summary if r["family"] == "equivariant_set"]
    fmt_table(matched, "matched bottleneck (hidden=6)")
    fmt_table(natural, "natural width MLPs")
    fmt_table(equivariant, "equivariant / invariant set models")
