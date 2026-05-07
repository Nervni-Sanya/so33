"""
benchmarks.run_synthetic_dataeff
--------------------------------
Data-efficiency learning curve on the synthetic Lorentz dataset.

Tests the hypothesis: a geometric inductive bias matched to the data
should require less data to reach a given accuracy. We sweep training
fractions {1%, 5%, 10%, 50%, 100%} of a 10k-sample dataset and record
val accuracy for each (model, fraction).

Run:
    python -m benchmarks.run_synthetic_dataeff --quick
    python -m benchmarks.run_synthetic_dataeff --models so33,relu_bottleneck,relu_mlp
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from benchmarks.synthetic import generate_causal_dataset
from benchmarks.models import build_model, MATCHED_MODELS, NATURAL_MODELS
from benchmarks.train import train_classifier, TrainConfig


FRACTIONS = (0.01, 0.05, 0.10, 0.50, 1.00)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",   action="store_true",
                        help="Smoke run: 500 samples, 5 epochs, fewer fractions.")
    parser.add_argument("--models",  type=str,
                        default=",".join(MATCHED_MODELS + ("relu_mlp",)),
                        help="Comma-separated model names.")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--n",       type=int, default=10_000,
                        help="Total dataset size (train+val combined).")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args(argv)

    fractions: tuple[float, ...] = FRACTIONS
    if args.quick:
        args.n      = 500
        args.epochs = 5
        fractions   = (0.10, 0.50, 1.00)

    n_train_full = int(0.7 * args.n)
    n_val        = args.n - n_train_full

    print(f"[dataeff] generating {args.n} samples, {n_train_full} train / {n_val} val ...")
    X, y = generate_causal_dataset(n_samples=args.n, seed=args.seed)
    X_train_full, y_train_full = X[:n_train_full], y[:n_train_full]
    X_val,        y_val        = X[n_train_full:], y[n_train_full:]

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    curves: dict[str, dict[float, float]] = {}

    for name in model_names:
        curve_for_model: dict[float, dict[str, float]] = {}
        for frac in fractions:
            n_use = max(2, int(round(frac * n_train_full)))
            # Use a fresh random subset per fraction to avoid favouring
            # the front of the shuffled dataset.
            torch.manual_seed(1000 + args.seed)
            idx = torch.randperm(n_train_full)[:n_use]
            Xtr, ytr = X_train_full[idx], y_train_full[idx]

            print(f"[dataeff] {name} | frac={frac:.2f} ({n_use} samples) ...")
            model = build_model(name, in_features=6, out_features=2, T=0.3)
            cfg   = TrainConfig(
                epochs=args.epochs,
                # Reduce batch size when training set is tiny so we still
                # take meaningful gradient steps per epoch.
                batch_size=min(128, max(8, n_use // 4)),
                lr=3e-3, seed=args.seed,
                cosine_schedule=True, grad_clip=1.0,
            )
            res = train_classifier(model, Xtr, ytr, X_val, y_val, cfg)
            curve_for_model[frac] = {
                "n_train":         n_use,
                "final_val_acc":   res.final_val_acc,
                "best_val_acc":    res.best_val_acc,
                "final_train_acc": res.final_train_acc,
                "walltime_sec":    res.walltime_sec,
                "n_params":        res.n_params,
            }
            print(f"   n_train={n_use:>5d}  val_acc={res.final_val_acc:.3f}  "
                  f"best_val={res.best_val_acc:.3f}  ({res.walltime_sec:.1f}s)")

        record = {
            "experiment":       "synthetic_dataeff",
            "model":            name,
            "seed":             args.seed,
            "fractions":        list(fractions),
            "curve":            {f"{k:.2f}": v for k, v in curve_for_model.items()},
            "n_total":          args.n,
            "n_train_full":     n_train_full,
        }
        out = results_dir / f"synthetic_dataeff__{name}__seed{args.seed}.json"
        out.write_text(json.dumps(record, indent=2))
        curves[name] = {k: v["final_val_acc"] for k, v in curve_for_model.items()}

    # Summary table
    print("\n[dataeff] val_acc by fraction")
    header = f"{'model':<24} " + " ".join(f"{f:>6.2f}" for f in fractions)
    print(header)
    print("-" * len(header))
    for name in model_names:
        cells = " ".join(f"{curves[name][f]:>6.3f}" for f in fractions)
        print(f"{name:<24} {cells}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
