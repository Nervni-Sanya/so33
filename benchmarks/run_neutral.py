"""
benchmarks.run_neutral
----------------------
Neutral tabular sanity check (UCI Adult or breast-cancer fallback).

Tier-2 sanity check: data with no Lorentz structure. The expectation
is that SO33 lands close to ReLU/GELU baselines — neither obviously
better nor worse. A large win OR a collapse would both be flags
worth investigating before reporting headline numbers.

Run:
    python -m benchmarks.run_neutral --quick
    python -m benchmarks.run_neutral --epochs 30
"""

from __future__ import annotations

import argparse
import sys

from .datasets import load_neutral_tabular, synthetic_tabular
from .tabular_runner import run_tabular_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick",       action="store_true",
                   help="Smoke run: 14-D Gaussian stand-in.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--models",      type=str, default=None)
    args = p.parse_args(argv)

    if args.quick:
        split = synthetic_tabular(n_samples=1_000, n_features=14,
                                  n_classes=2, name="neutral_synthetic",
                                  seed=args.seed)
        epochs = 5
        experiment = "neutral_quick"
    else:
        split = load_neutral_tabular(seed=args.seed, standardise=True)
        epochs = args.epochs
        experiment = "neutral"

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
    )
    if models is not None:
        kwargs["models"] = models

    run_tabular_experiment(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
