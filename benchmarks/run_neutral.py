"""
benchmarks.run_neutral
----------------------
Neutral tabular sanity check (UCI Adult or breast-cancer fallback).

Tier-2 sanity check: data with no Lorentz structure. The expectation
is that SO33 lands close to ReLU/GELU baselines — neither obviously
better nor worse. A large win OR a collapse would both be flags
worth investigating before reporting headline numbers.

Controls for the parameter-efficiency comparison: ``tanh_mlp_w24`` is a
width-24 tanh MLP with the same parameter count as ``so33_multi``
(2,594 vs 2,654), and ``--weight-decay`` / ``--early-stop-patience``
regularise the width-256 MLPs, which otherwise overfit.

Run:
    python -m benchmarks.run_neutral --quick
    python -m benchmarks.run_neutral --epochs 30
    python -m benchmarks.run_neutral --seeds 0-2 \\
        --models tanh_mlp_w24,so33_multi
    python -m benchmarks.run_neutral --seeds 0-2 --models relu_mlp,tanh_mlp,gelu_mlp \\
        --weight-decay 1e-4 --early-stop-patience 5
"""

from __future__ import annotations

import argparse
import sys

from .datasets import load_neutral_tabular, synthetic_tabular
from .provenance import experiment_tag, parse_seeds, run_metadata
from .tabular_runner import run_tabular_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick",       action="store_true",
                   help="Smoke run: 14-D Gaussian stand-in.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--seeds",       type=str, default=None,
                   help="Seeds to run in one process, e.g. '0-2'. Overrides --seed.")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--models",      type=str, default=None)
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
            split = synthetic_tabular(n_samples=1_000, n_features=14,
                                      n_classes=2, name="neutral_synthetic",
                                      seed=seed)
            epochs = 5
            experiment = "neutral_quick"
        else:
            split = load_neutral_tabular(seed=seed, standardise=True)
            epochs = args.epochs
            experiment = experiment_tag(
                "neutral",
                {"epochs": args.epochs, "wd": args.weight_decay,
                 "es": args.early_stop_patience},
                {"epochs": 30, "wd": 0.0, "es": None})

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
