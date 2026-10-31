"""
benchmarks.run_top_tagging
--------------------------
Top Tagging Reference (arXiv:1902.09914) benchmark.

Two representations:
- ``aggregated``: per-jet 4-momentum aggregated to a 6-D representation
  (see ``aggregate_jet_to_6d`` in benchmarks.datasets). Saturates, because
  it hands every model the jet mass.
- ``constituents``: the leading-K constituents per jet as a (K, 5) set
  (normalised E, px, py, pz + mask) for the per-particle Deep Sets models
  and the invariant / equivariant Architectures A and B. ``--canonical-splits``
  uses the published Kasieczka train / val / test files.

Reports matched-bottleneck, natural-width and equivariant-set tables. Note:
with only 6 features the natural-width MLP also has ~6 -> 256 -> 2 = much
larger param count than matched, but this matches the "engineering"
interpretation (let baselines use what dimensionality they want).

Result files are ``<experiment>__<model>__seed<N>.json``. Any non-default
protocol option (normalisation, pooling, K, sample caps, epochs, connection
cap, weight decay, early stopping) is appended to the experiment name, so
runs with different settings never overwrite each other.

Architecture B stability: lifted massless constituents are not bounded by
any equivariant input map, and with the legacy connection cap the ODE
diverges on top-tagging data. ``--connection-cap auto`` (default) gives the
Architecture-B channels a provably blow-up-free cap computed from the
largest training-constituent norm; ``off`` reproduces the legacy setting.
Check a configuration first with
``python -m benchmarks.diagnose_archb_stability --cache-dir data``.

Run:
    # Smoke (synthetic 6-D stand-in):
    python -m benchmarks.run_top_tagging --quick

    # Real data — download first (benchmarks.download_top_tagging) into data/:
    #   data/top_tagging_train.npz, top_tagging_val.npz, top_tagging_test.npz
    python -m benchmarks.run_top_tagging --cache-dir data --max-samples 100000

    # Canonical protocol, Architecture B, three seeds, with per-jet scores:
    python -m benchmarks.run_top_tagging --representation constituents \\
        --canonical-splits --models so33_equivariant_eta_bounded \\
        --seeds 0-2 --save-scores
"""

from __future__ import annotations

import argparse
import sys

from .datasets import (
    load_top_tagging, load_top_tagging_constituents, synthetic_tabular,
)
from .models import BASELINE_SET_MODELS
from .provenance import experiment_tag, parse_seeds, run_metadata
from .tabular_runner import run_tabular_experiment


def _cap_arg(text: str) -> str | float:
    if text in ("auto", "off"):
        return text
    return float(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick",       action="store_true",
                   help="Smoke run: 6-D Gaussian stand-in (1k samples, 5 epochs).")
    p.add_argument("--cache-dir",   type=str, default="data",
                   help="Directory containing top_tagging_*.npz.")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--seeds",       type=str, default=None,
                   help="Seeds to run in one process, e.g. '0-2' or '0,3'. "
                        "Overrides --seed.")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--models",      type=str, default=None)
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
                        "test is always loaded in full, and val too (or, if the "
                        "val file is missing, a fixed 5%% of the capped train). "
                        "Only used with --canonical-splits.")
    p.add_argument("--connection-cap", type=_cap_arg, default="auto",
                   help="Architecture-B connection cap: 'auto' (default; provably "
                        "blow-up-free for the training-constituent norms), 'off' "
                        "(legacy cap 1.0, diverges on massless constituents) or a "
                        "number.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Stop after this many epochs without a validation-loss "
                        "improvement and restore the best weights.")
    p.add_argument("--save-scores", action="store_true",
                   help="Also write per-jet test scores and labels to "
                        "<experiment>__<model>__seed<N>__scores.npz.")
    args = p.parse_args(argv)

    seeds = parse_seeds(args.seeds, default=args.seed)
    models = (
        [m.strip() for m in args.models.split(",")] if args.models
        else None
    )
    physical = sorted(set(models or ()) & set(BASELINE_SET_MODELS))
    if args.normalize == "per_component" and physical:
        # PFN / EFN read pT fractions and angles, LorentzNet Minkowski
        # products: a per-component z-score destroys both.
        p.error(f"{', '.join(physical)} need physical four-momenta: use "
                f"--normalize global (default) or none")
    meta = run_metadata(argv)
    common = {"epochs": args.epochs, "wd": args.weight_decay,
              "es": args.early_stop_patience}
    common_defaults = {"epochs": 30, "wd": 0.0, "es": None}

    split = None
    for seed in seeds:
        if args.quick:
            split = synthetic_tabular(n_samples=1_000, n_features=6,
                                      n_classes=2, name="top_tagging_synthetic",
                                      seed=seed)
            epochs = 5
            experiment = experiment_tag(
                "top_tagging_quick",
                {"wd": args.weight_decay, "es": args.early_stop_patience},
                {"wd": 0.0, "es": None})
            rep = "flat"
        elif args.representation == "constituents":
            # The canonical split (including the fixed val carve when the val
            # file is missing) does not depend on the seed, so it is loaded
            # once; the internal 70/15/15 split depends on the seed and is
            # reloaded per seed.
            if split is None or not args.canonical_splits:
                split = load_top_tagging_constituents(
                    cache_dir=args.cache_dir, max_samples=args.max_samples,
                    n_constituents=args.n_constituents, seed=seed, standardise=True,
                    normalize=args.normalize,
                    use_canonical_splits=args.canonical_splits,
                    max_train_samples=args.max_train_samples,
                )
            epochs = args.epochs
            base = ("top_tagging_canonical" if args.canonical_splits
                    else "top_tagging_constituents")
            options = {"norm": args.normalize, "pool": args.pool,
                       "k": args.n_constituents,
                       "cap": args.connection_cap, **common}
            defaults = {"norm": "global", "pool": "mean", "k": 32,
                        "cap": "auto", **common_defaults}
            if args.canonical_splits:
                options["ntrain"] = args.max_train_samples
                defaults["ntrain"] = None
            else:
                options["n"] = args.max_samples
                defaults["n"] = 100_000
            experiment = experiment_tag(base, options, defaults)
            rep = "constituents"
        else:
            split = load_top_tagging(cache_dir=args.cache_dir,
                                     max_samples=args.max_samples,
                                     seed=seed, standardise=True)
            epochs = args.epochs
            experiment = experiment_tag(
                "top_tagging", {"n": args.max_samples, **common},
                {"n": 100_000, **common_defaults})
            rep = "flat"

        kwargs = dict(
            experiment=experiment,
            split=split,
            seed=seed,
            epochs=epochs,
            representation=rep,
            pool=args.pool,
            results_dir=args.results_dir,
            connection_cap=args.connection_cap,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            save_scores=args.save_scores,
            run_meta=meta,
        )
        if models is not None:
            kwargs["models"] = models

        run_tabular_experiment(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
