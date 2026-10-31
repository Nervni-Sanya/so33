"""
benchmarks.run_higgs
--------------------
HIGGS (UCI) benchmark.

Tier-1 secondary anchor: 28-feature derived kinematic representation,
binary classification (Higgs signal vs. background). Reports both
matched-bottleneck (Linear -> 6 -> activation -> Linear) and
natural-width (Linear(28 -> 256) -> activation -> Linear) tables so
the reader sees both the scientific and the engineering comparison.

The train/val/test split depends on the seed, so all models of one seed
share a split and can be compared pairwise across seeds
(``python -m benchmarks.stats --experiment higgs ...``).

Run:
    # Smoke (synthetic stand-in, no download required):
    python -m benchmarks.run_higgs --quick

    # Real data — download HIGGS.csv.gz first:
    curl -L https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz \\
        -o data/HIGGS.csv.gz
    python -m benchmarks.run_higgs --cache-dir data --max-samples 200000 --epochs 30

    # Eight seeds for confidence intervals:
    python -m benchmarks.run_higgs --cache-dir data --seeds 0-7
"""

from __future__ import annotations

import argparse
import sys

from .datasets import load_higgs, synthetic_tabular
from .provenance import experiment_tag, parse_seeds, run_metadata
from .tabular_runner import run_tabular_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick",       action="store_true",
                   help="Smoke run: synthetic Gaussian stand-in (1k samples, 5 epochs).")
    p.add_argument("--cache-dir",   type=str, default="data",
                   help="Directory containing HIGGS.csv.gz.")
    p.add_argument("--max-samples", type=int, default=200_000,
                   help="Sub-sample HIGGS to this many rows (default 200k of 11M).")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--seeds",       type=str, default=None,
                   help="Seeds to run in one process, e.g. '0-7'. Overrides --seed.")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--models",      type=str, default=None,
                   help="Comma-separated subset of model names.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Stop after this many epochs without a validation-loss "
                        "improvement and restore the best weights.")
    args = p.parse_args(argv)

    models = (
        [m.strip() for m in args.models.split(",")] if args.models
        else None
    )
    meta = run_metadata(argv)

    for seed in parse_seeds(args.seeds, default=args.seed):
        if args.quick:
            split = synthetic_tabular(n_samples=1_000, n_features=28,
                                      n_classes=2, name="higgs_synthetic",
                                      seed=seed)
            epochs = 5
            experiment = experiment_tag(
                "higgs_quick",
                {"wd": args.weight_decay, "es": args.early_stop_patience},
                {"wd": 0.0, "es": None})
        else:
            split = load_higgs(cache_dir=args.cache_dir,
                               max_samples=args.max_samples,
                               seed=seed, standardise=True)
            epochs = args.epochs
            experiment = experiment_tag(
                "higgs",
                {"n": args.max_samples, "epochs": args.epochs,
                 "wd": args.weight_decay, "es": args.early_stop_patience},
                {"n": 200_000, "epochs": 30, "wd": 0.0, "es": None})

        kwargs = dict(
            experiment=experiment,
            split=split,
            seed=seed,
            epochs=epochs,
            results_dir=args.results_dir,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            run_meta=meta,
        )
        if models is not None:
            kwargs["models"] = models

        run_tabular_experiment(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
