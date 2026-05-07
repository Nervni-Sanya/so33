"""
benchmarks.run_synthetic_ood
----------------------------
OOD generalization experiment on the synthetic Lorentz dataset.

Train on low-rapidity samples (small absolute Lorentz norm), evaluate
on high-rapidity samples. The hypothesis: a model with the right
geometric prior should generalize across the full rapidity range,
whereas a generic MLP should degrade more.

We split the 10k dataset by the median absolute Lorentz norm of the
terminal states. The "low" half is split 70/30 into train/val; the
full "high" half is the held-out OOD test set.

Run:
    python -m benchmarks.run_synthetic_ood --quick
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from benchmarks.synthetic import (
    generate_causal_dataset,
    boost_split,
    lorentz_norm_squared,
)
from benchmarks.models import build_model, MATCHED_MODELS, NATURAL_MODELS
from benchmarks.train import train_classifier, TrainConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",   action="store_true",
                        help="Smoke run: 500 samples, 5 epochs.")
    parser.add_argument("--models",  type=str,
                        default=",".join(MATCHED_MODELS + ("relu_mlp",)),
                        help="Comma-separated model names.")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--n",       type=int, default=10_000)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args(argv)

    if args.quick:
        args.n      = 500
        args.epochs = 5

    print(f"[ood] generating {args.n} samples ...")
    X, y = generate_causal_dataset(n_samples=args.n, seed=args.seed)

    # Split by Lorentz norm magnitude.
    (X_low, y_low), (X_high, y_high) = boost_split(X, y)
    norm = lorentz_norm_squared(X).abs()
    print(f"[ood] split: low={len(X_low)}, high={len(X_high)} | "
          f"|norm| min={norm.min():.3f} median={norm.median():.3f} max={norm.max():.3f}")

    n_train = int(0.7 * len(X_low))
    perm    = torch.randperm(len(X_low))
    Xtr, ytr = X_low[perm[:n_train]], y_low[perm[:n_train]]
    Xva, yva = X_low[perm[n_train:]], y_low[perm[n_train:]]
    Xood, yood = X_high, y_high

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    summary = []

    for name in model_names:
        print(f"[ood] {name} | train ({len(Xtr)}) ...")
        model = build_model(name, in_features=6, out_features=2, T=0.3)
        cfg   = TrainConfig(
            epochs=args.epochs, batch_size=128, lr=3e-3, seed=args.seed,
            cosine_schedule=True, grad_clip=1.0,
        )
        res = train_classifier(model, Xtr, ytr, Xva, yva, cfg)

        # OOD evaluation
        model.eval()
        with torch.no_grad():
            ood_pred = model(Xood).argmax(dim=-1)
            ood_acc  = (ood_pred == yood).float().mean().item()

        record = {
            "experiment":     "synthetic_ood",
            "model":          name,
            "seed":           args.seed,
            "n_train":        len(Xtr),
            "n_val":          len(Xva),
            "n_ood":          len(Xood),
            "n_params":       res.n_params,
            "walltime_sec":   res.walltime_sec,
            "id_val_acc":     res.final_val_acc,
            "id_best_val":    res.best_val_acc,
            "ood_acc":        ood_acc,
            "ood_gap":        res.final_val_acc - ood_acc,
        }
        out = results_dir / f"synthetic_ood__{name}__seed{args.seed}.json"
        out.write_text(json.dumps(record, indent=2))
        summary.append(record)
        print(f"   id_val={res.final_val_acc:.3f}  ood={ood_acc:.3f}  "
              f"gap={record['ood_gap']:+.3f}")

    # Summary table
    print("\n[ood] generalization gap (id_val - ood_acc; lower is better)")
    print(f"{'model':<24} {'id_val':>7} {'ood':>7} {'gap':>7}")
    print("-" * 50)
    for r in summary:
        print(f"{r['model']:<24} {r['id_val_acc']:>7.3f} "
              f"{r['ood_acc']:>7.3f} {r['ood_gap']:>+7.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
