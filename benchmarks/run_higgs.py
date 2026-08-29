"""
benchmarks.run_higgs
--------------------
HIGGS (UCI) benchmark.

Tier-1 secondary anchor: 28-feature derived kinematic representation,
binary classification (Higgs signal vs. background). Reports both
matched-bottleneck (Linear -> 6 -> activation -> Linear) and
natural-width (Linear(28 -> 256) -> activation -> Linear) tables so
the reader sees both the scientific and the engineering comparison.

Run:
    # Smoke (synthetic stand-in, no download required):
    python -m benchmarks.run_higgs --quick

    # Real data — download HIGGS.csv.gz first:
    curl -L https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz \\
        -o data/HIGGS.csv.gz
    python -m benchmarks.run_higgs --cache-dir data --max-samples 200000 --epochs 30
"""

from __future__ import annotations

import argparse
import sys

import torch

from .datasets import load_higgs, synthetic_tabular
from .tabular_runner import run_tabular_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick",       action="store_true",
                   help="Smoke run: synthetic Gaussian stand-in (1k samples, 5 epochs).")
    p.add_argument("--cache-dir",   type=str, default="data",
                   help="Directory containing HIGGS.csv.gz.")
    p.add_argument("--max-samples", type=int, default=200_000,
                   help="Sub-sample HIGGS to this many rows (default 200k of 11M).")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu | cuda | cuda:0 ...")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--feature-set", choices=["all", "low", "high"], default="all",
                   help="all: 28 features. low: the 21 low-level kinematic "
                        "ones. high: the 7 high-level invariant masses "
                        "(m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb).")
    p.add_argument("--models",      type=str, default=None,
                   help="Comma-separated subset of model names.")
    args = p.parse_args(argv)

    if args.quick:
        split = synthetic_tabular(n_samples=1_000, n_features=28,
                                  n_classes=2, name="higgs_synthetic",
                                  seed=args.seed)
        epochs = 5
        experiment = "higgs_quick"
    else:
        split = load_higgs(cache_dir=args.cache_dir,
                           max_samples=args.max_samples,
                           seed=args.seed, standardise=True,
                           feature_set=args.feature_set)
        epochs = args.epochs
        # The feature set goes into the experiment name: results are keyed
        # by experiment/model/seed, so "all" and "low" would otherwise
        # overwrite each other.
        experiment = ("higgs" if args.feature_set == "all"
                      else f"higgs_{args.feature_set}")

    models = (
        [m.strip() for m in args.models.split(",")] if args.models
        else None
    )

    kwargs = dict(
        experiment=experiment,
        split=split,
        seed=args.seed,
        epochs=epochs,
        results_dir=args.results_dir,
        device=args.device,
        dtype=torch.float32 if args.dtype == "float32" else torch.float64,
        batch_size=args.batch_size,
    )
    if models is not None:
        kwargs["models"] = models

    run_tabular_experiment(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
