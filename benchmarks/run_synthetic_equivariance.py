"""
benchmarks.run_synthetic_equivariance
-------------------------------------
Equivariance check on the synthetic Lorentz-structured dataset.

For each (model, seed) we:
  1. Generate the 10k-sample causal dataset.
  2. Train a classifier on the train split.
  3. Sample N_TRANSFORMS random SO(3,3) elements at varying rapidities.
  4. Apply each transform g to the held-out validation inputs and measure
     how the prediction distribution changes.

A truly equivariant model should keep the same per-sample classification
when inputs are transformed by elements of the *symmetry* group of the
data-generating process. SO33 is not strictly equivariant under SO(3,3)
(the activation has a learned, not equivariant, connection) but should
degrade more gracefully than ReLU/Tanh baselines on Lorentz-structured
data.

Metrics:
  - val_acc_clean       : accuracy on untransformed validation inputs.
  - val_acc_transformed : accuracy after applying g (averaged over all g).
  - prediction_consistency : fraction of validation inputs that retain
        their predicted class under transformation (averaged over g).
  - per_rapidity        : same metrics broken out by rapidity.

Run:
    python -m benchmarks.run_synthetic_equivariance --quick
    python -m benchmarks.run_synthetic_equivariance --models so33,relu_bottleneck
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from benchmarks.synthetic import (
    generate_causal_dataset,
    random_so33_element,
    transform_inputs,
)
from benchmarks.models import build_model, MATCHED_MODELS
from benchmarks.train import train_classifier, TrainConfig


RAPIDITIES = (0.1, 0.3, 0.6)
N_TRANSFORMS_PER_RAPIDITY = 5


def evaluate_equivariance(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    rapidities: tuple[float, ...] = RAPIDITIES,
    n_transforms: int = N_TRANSFORMS_PER_RAPIDITY,
    seed: int = 0,
) -> dict:
    """Compute clean accuracy and transformed accuracy at multiple rapidities."""
    model.eval()
    with torch.no_grad():
        clean_pred = model(X_val).argmax(dim=-1)
        clean_acc  = (clean_pred == y_val).float().mean().item()

    gen = torch.Generator().manual_seed(seed)

    per_rapidity: dict[float, dict[str, float]] = {}
    all_transformed_accs:    list[float] = []
    all_consistency_rates:   list[float] = []

    for rho in rapidities:
        rho_accs:          list[float] = []
        rho_consistencies: list[float] = []
        for _ in range(n_transforms):
            g  = random_so33_element(rapidity=rho, generator=gen)
            Xp = transform_inputs(X_val, g)
            with torch.no_grad():
                pred_t = model(Xp).argmax(dim=-1)
            acc_t          = (pred_t == y_val).float().mean().item()
            consistency    = (pred_t == clean_pred).float().mean().item()
            rho_accs.append(acc_t)
            rho_consistencies.append(consistency)

        per_rapidity[rho] = {
            "transformed_acc_mean":     sum(rho_accs)          / n_transforms,
            "transformed_acc_min":      min(rho_accs),
            "consistency_mean":         sum(rho_consistencies) / n_transforms,
        }
        all_transformed_accs.extend(rho_accs)
        all_consistency_rates.extend(rho_consistencies)

    return {
        "val_acc_clean":           clean_acc,
        "val_acc_transformed":     sum(all_transformed_accs)  / len(all_transformed_accs),
        "prediction_consistency":  sum(all_consistency_rates) / len(all_consistency_rates),
        "per_rapidity":            {str(k): v for k, v in per_rapidity.items()},
        "n_transforms":            n_transforms,
        "rapidities":              list(rapidities),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",   action="store_true",
                        help="Smoke run: 500 samples, 5 epochs.")
    parser.add_argument("--models",  type=str, default=",".join(MATCHED_MODELS),
                        help="Comma-separated model names (default: all matched-bottleneck).")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--n",       type=int, default=10_000,
                        help="Total dataset size.")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args(argv)

    if args.quick:
        args.n      = 500
        args.epochs = 5

    n_train = int(0.7 * args.n)

    print(f"[equivariance] generating {args.n} samples ...")
    X, y = generate_causal_dataset(n_samples=args.n, seed=args.seed)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:], y[n_train:]

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    summary: list[dict] = []

    for name in model_names:
        print(f"[equivariance] {name} | train ...")
        model = build_model(name, in_features=6, out_features=2, T=0.3)
        cfg   = TrainConfig(
            epochs=args.epochs, batch_size=128, lr=3e-3, seed=args.seed,
            cosine_schedule=True, grad_clip=1.0,
        )
        train_res = train_classifier(model, X_train, y_train, X_val, y_val, cfg)
        eq_res = evaluate_equivariance(
            model, X_val, y_val, seed=args.seed,
        )
        record = {
            "experiment":       "synthetic_equivariance",
            "model":            name,
            "seed":             args.seed,
            "n_samples":        args.n,
            "n_params":         train_res.n_params,
            "walltime_sec":     train_res.walltime_sec,
            "train_metrics":    {
                "final_train_acc": train_res.final_train_acc,
                "final_val_acc":   train_res.final_val_acc,
                "epochs_run":      train_res.epochs_run,
            },
            "equivariance":     eq_res,
        }
        out = results_dir / f"synthetic_equivariance__{name}__seed{args.seed}.json"
        out.write_text(json.dumps(record, indent=2))
        summary.append({
            "model":         name,
            "n_params":      train_res.n_params,
            "clean_acc":     eq_res["val_acc_clean"],
            "transf_acc":    eq_res["val_acc_transformed"],
            "consistency":   eq_res["prediction_consistency"],
            "Δ":             eq_res["val_acc_clean"] - eq_res["val_acc_transformed"],
        })
        print(f"   clean_acc={eq_res['val_acc_clean']:.3f}  "
              f"transf_acc={eq_res['val_acc_transformed']:.3f}  "
              f"consistency={eq_res['prediction_consistency']:.3f}")

    # Print summary table
    print("\n[equivariance] summary")
    print(f"{'model':<24} {'params':>8} {'clean':>7} {'transf':>7} {'cons':>7} {'Δ':>7}")
    print("-" * 64)
    for r in summary:
        print(f"{r['model']:<24} {r['n_params']:>8} "
              f"{r['clean_acc']:>7.3f} {r['transf_acc']:>7.3f} "
              f"{r['consistency']:>7.3f} {r['Δ']:>+7.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
