"""
benchmarks.run_top_tagging
--------------------------
Top Tagging Reference (arXiv:1902.09914) benchmark.

Tier-1 *headline* anchor: per-jet 4-momentum aggregated to a 6-D
representation that aligns directly with the (3,3) signature
(see ``aggregate_jet_to_6d`` in benchmarks.datasets). This is the
experiment where SO33's geometric prior should pay off most, since
the data has explicit Lorentzian structure.

Reports matched-bottleneck and natural-width tables. Note: with only
6 features the natural-width MLP also has ~6 -> 256 -> 2 = much
larger param count than matched, but this matches the "engineering"
interpretation (let baselines use what dimensionality they want).

Run:
    # Smoke (synthetic 6-D stand-in):
    python -m benchmarks.run_top_tagging --quick

    # Real data — convert Zenodo HDF5 to npz and place in data/:
    #   data/top_tagging_train.npz, top_tagging_val.npz, top_tagging_test.npz
    python -m benchmarks.run_top_tagging --cache-dir data --max-samples 100000
"""

from __future__ import annotations

import argparse
import sys

from .datasets import (
    load_top_tagging, load_top_tagging_constituents, synthetic_tabular,
)
from .tabular_runner import run_tabular_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick",       action="store_true",
                   help="Smoke run: 6-D Gaussian stand-in (1k samples, 5 epochs).")
    p.add_argument("--cache-dir",   type=str, default="data",
                   help="Directory containing top_tagging_*.npz.")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--models",      type=str, default=None)
    p.add_argument("--natural-hidden", type=int, default=256,
                   help="Hidden width of the natural-width baselines (*_mlp). "
                        "Set it to build a PARAMETER-MATCHED generic baseline "
                        "against the so3c set models (e.g. 1293 ~ 9.05k params, "
                        "matching so3c_equivariant_set) so the comparison "
                        "isolates the geometric prior from raw capacity.")
    p.add_argument("--representation", choices=["aggregated", "constituents"],
                   default="aggregated",
                   help="aggregated: jet-level 6-D summary (secondary baseline; "
                        "saturates because it hands every model the jet mass). "
                        "constituents: per-particle Deep Sets (headline experiment "
                        "where the geometric prior can use substructure).")
    p.add_argument("--n-constituents", type=int, default=32,
                   help="Leading constituents per jet (constituents mode only).")
    p.add_argument("--normalize", choices=["global", "per_component", "none"],
                   default="global",
                   help="Constituent normalisation. 'global' (default) preserves "
                        "the Lorentz invariant E^2-p^2 (best for SO33); "
                        "'per_component' z-scores each component (destroys it).")
    p.add_argument("--pool", choices=["mean", "sum"], default="mean",
                   help="Deep Sets pooling over constituents.")
    p.add_argument("--canonical-splits", action="store_true",
                   help="Use the published Kasieczka train/val/test split "
                        "(reads top_tagging_{train,val,test}.npz separately) "
                        "instead of a random 70/15/15 re-split. Makes the "
                        "test AUC directly comparable to published numbers. "
                        "constituents mode only.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap the number of canonical-train jets (memory/time); "
                        "val and test are always loaded in full. Only used "
                        "with --canonical-splits.")
    args = p.parse_args(argv)

    if args.quick:
        split = synthetic_tabular(n_samples=1_000, n_features=6,
                                  n_classes=2, name="top_tagging_synthetic",
                                  seed=args.seed)
        epochs = 5
        experiment = "top_tagging_quick"
        rep = "flat"
    elif args.representation == "constituents":
        split = load_top_tagging_constituents(
            cache_dir=args.cache_dir, max_samples=args.max_samples,
            n_constituents=args.n_constituents, seed=args.seed, standardise=True,
            normalize=args.normalize,
            use_canonical_splits=args.canonical_splits,
            max_train_samples=args.max_train_samples,
        )
        epochs = args.epochs
        experiment = ("top_tagging_canonical" if args.canonical_splits
                      else "top_tagging_constituents")
        rep = "constituents"
    else:
        split = load_top_tagging(cache_dir=args.cache_dir,
                                 max_samples=args.max_samples,
                                 seed=args.seed, standardise=True)
        epochs = args.epochs
        experiment = "top_tagging"
        rep = "flat"

    models = (
        [m.strip() for m in args.models.split(",")] if args.models
        else None
    )

    kwargs = dict(
        experiment=experiment,
        split=split,
        seed=args.seed,
        epochs=epochs,
        representation=rep,
        pool=args.pool,
        natural_hidden=args.natural_hidden,
        results_dir=args.results_dir,
    )
    if models is not None:
        kwargs["models"] = models

    run_tabular_experiment(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
