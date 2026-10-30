"""
benchmarks.run_jet_baselines
----------------------------
Jet-observable baselines for top tagging, the referee's item (i) "a cut or BDT
on (M_jet, tau32)", evaluated like the networks of ``benchmarks.run_top_tagging``:

    jet_mass        score = M_jet                       0 parameters
    mass_window     score = -|M_jet - 172.5 GeV|        0 parameters
    bdt_mass_tau32  boosted decision trees on (M_jet, tau32): scikit-learn's
                    HistGradientBoostingClassifier, score = its decision
                    function (the log-odds)

M_jet and tau32 = tau3 / tau2 (N-subjettiness with beta = 1, R0 = 0.8 and
exclusive-kT axes without minimisation; tau32 = 1 when tau2 = 0) come from
``benchmarks.jet_observables``. They are computed from exactly the inputs of the
networks: the leading ``--n-constituents`` (32) constituents of every jet, from
the same split (the published Kasieczka train / val / test with
``--canonical-splits``), read in raw GeV (``normalize="none"``).

Protocol. The jet mass and the mass window are fixed functions of the jet; the
window centre (``--mass-centre``, the top mass) is a physics input and is not
tuned. The BDT is fitted on train alone, with random_state = ``--seed``; its early
stopping monitors the log-loss on a stratified 10% of TRAIN (scikit-learn's
validation_fraction = 0.1, n_iter_no_change = 20; at most ``--bdt-max-iter`` =
500 trees of at most 31 leaves, learning rate 0.1), so val plays the same part as
for the networks (monitoring and the accuracy cut) and test is used once. For
every score the accuracy threshold is the cut that maximises the val accuracy
(``run_efp_baseline.best_threshold``) and test_acc is the test accuracy at that
same cut; test_auc and 1/eps_B at eps_S = 0.3 / 0.5 are
``tabular_runner.binary_metrics``. The referee's numbers on the canonical test
set at K = 32, for comparison: mass window AUC 0.941 and 1/eps_B = 39.7; M_jet
alone 0.921 and 17.7.

Writes ``<experiment>__<model>__seed<N>.json`` for the three models (family
``jet_observable``, the score-based schema of ``run_efp_baseline``; the experiment
name is the one ``run_top_tagging`` gives the same data options, so
``benchmarks.aggregate`` renders the rows next to the neural models) and, with
``--save-scores``, ``...__scores.npz`` with the per-jet test scores (float32) and
labels (int8). n_params is 0 for the two cuts and null for the BDT, whose size
(trees x leaves) is in ``model_size``. A non-default mass centre or BDT setting is
appended to the model name. The observables cost ~0.1 ms per jet on one core
(~3.5 min for the 2.0M canonical jets); ``--cache-observables`` keeps them,
keyed to a sha256 of the jets, for later runs (other BDT seeds or settings).

Run:
    # Canonical protocol, observables cached for the next seeds:
    python -m benchmarks.run_jet_baselines --canonical-splits --save-scores \\
        --cache-observables data/jet_observables_canonical.npz
    python -m benchmarks.run_jet_baselines --canonical-splits --seed 1 \\
        --cache-observables data/jet_observables_canonical.npz

    # Smoke test on toy jets (no data needed):
    python -m benchmarks.run_jet_baselines --toy-jets 2000
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np

from benchmarks.datasets import DatasetSplit
from benchmarks.jet_observables import (
    AXES, BETA, CHUNK_JETS, OBSERVABLES, R0, RATIO_IF_ZERO, compute_jet_observables,
)
from benchmarks.provenance import experiment_tag, run_metadata
from benchmarks.run_efp_baseline import (
    best_threshold, data_fingerprint, experiment_name, load_split,
)
from benchmarks.tabular_runner import binary_metrics

TOP_MASS = 172.5                     # GeV, default centre of the mass window
BDT_FEATURES = ("mass", "tau32")
BDT_DEFAULTS: dict[str, Any] = {
    "max_iter": 500, "learning_rate": 0.1, "max_leaf_nodes": 31,
    "min_samples_leaf": 20, "validation_fraction": 0.1, "n_iter_no_change": 20,
}
SPLITS = ("train", "val", "test")
CACHE_FORMAT = 1


# ─────────────────────────────────────────────────────────────────────────
# Observables (with an optional cache)
# ─────────────────────────────────────────────────────────────────────────

def observable_cache_key(split: DatasetSplit, fingerprint: str) -> dict[str, Any]:
    """Everything a cache of jet observables must match."""
    return {"format": CACHE_FORMAT, "observables": list(OBSERVABLES),
            "beta": BETA, "r0": R0, "axes": AXES, "ratio_if_zero": RATIO_IF_ZERO,
            "n_constituents": int(split.X_train.shape[1]),
            "sizes": [len(split.X_train), len(split.X_val), len(split.X_test)],
            "fingerprint": fingerprint}


def save_observable_cache(path: pathlib.Path, obs: list[dict[str, np.ndarray]],
                          meta: dict[str, Any]) -> None:
    """Write the observables of the three splits atomically (temporary file, rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{s}_{name}": o[name] for s, o in zip(SPLITS, obs) for name in OBSERVABLES}
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:   # a file object: np.savez must not append ".npz"
        np.savez(fh, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_observable_cache(path: pathlib.Path,
                          key: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """The cached observables, or ValueError (file left alone) if made for other data."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["meta"].item())
        bad = sorted(k for k in key if meta.get(k) != key[k])
        if bad:
            raise ValueError(
                f"observable cache {path} was computed for a different setting "
                f"(mismatch in {', '.join(bad)}); delete it or pass another "
                f"--cache-observables path.")
        return [{name: z[f"{s}_{name}"] for name in OBSERVABLES} for s in SPLITS]


def split_observables(split: DatasetSplit, chunk: int = CHUNK_JETS,
                      cache_path: str | None = None, label: str = ""
                      ) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """[train, val, test] jet observables (raw-GeV jets) and how they were obtained."""
    t0 = time.time()
    path = pathlib.Path(cache_path) if cache_path else None
    key = observable_cache_key(split, data_fingerprint(split)) if path else None
    cache_hit = False
    if path is not None and path.is_file():
        obs = load_observable_cache(path, key)
        cache_hit = True
        print(f"[{label}] jet observables loaded from {path}")
    else:
        print(f"[{label}] computing jet observables (M, pT, tau_1..3 with "
              f"exclusive-kT axes, beta = {BETA:g}, R0 = {R0:g})")
        obs = []
        for name, X in zip(SPLITS, (split.X_train, split.X_val, split.X_test)):
            Xn = X.numpy()
            obs.append(compute_jet_observables(Xn[..., :4], Xn[..., 4],
                                               chunk=chunk, label=name))
        if path is not None:
            save_observable_cache(path, obs, {**key, "numpy": np.__version__})
            print(f"[{label}] jet observables cached in {path}")
    for name, o in zip(SPLITS, obs):
        n_bad = int(sum((~np.isfinite(o[k])).sum() for k in OBSERVABLES))
        if n_bad:
            raise ValueError(f"{n_bad} non-finite jet observables in {name}; "
                             f"check the constituents.")
    n_empty = int(sum((o["n_constituents"] == 0).sum() for o in obs))
    if n_empty:
        print(f"[{label}] warning: {n_empty} jets have no constituents "
              f"(M = 0, tau21 = tau32 = 1).")
    info = {
        "names": list(OBSERVABLES),
        "observable_sec": time.time() - t0,
        "cache": cache_path,
        "cache_hit": cache_hit,
        "n_empty_jets": n_empty,
        # jets that use the tau_N / tau_{N-1} = 1 convention (denominator 0)
        "n_tau1_zero": int(sum((o["tau1"] == 0).sum() for o in obs)),
        "n_tau2_zero": int(sum((o["tau2"] == 0).sum() for o in obs)),
    }
    return obs, info


# ─────────────────────────────────────────────────────────────────────────
# Scores
# ─────────────────────────────────────────────────────────────────────────

def bdt_settings(args: argparse.Namespace) -> dict[str, Any]:
    """The HistGradientBoostingClassifier settings of a run."""
    return {"max_iter": args.bdt_max_iter, "learning_rate": args.bdt_learning_rate,
            "max_leaf_nodes": args.bdt_max_leaf_nodes,
            "min_samples_leaf": args.bdt_min_samples_leaf,
            "validation_fraction": args.bdt_validation_fraction,
            "n_iter_no_change": args.bdt_n_iter_no_change}


def model_names(args: argparse.Namespace) -> dict[str, str]:
    """Result names of the three scores, with any non-default setting appended."""
    short = {"max_iter": "iter", "learning_rate": "lr", "max_leaf_nodes": "leaves",
             "min_samples_leaf": "minleaf", "validation_fraction": "vf",
             "n_iter_no_change": "patience"}
    settings = bdt_settings(args)
    return {
        "jet_mass": "jet_mass",
        "mass_window": experiment_tag("mass_window", {"centre": args.mass_centre},
                                      {"centre": TOP_MASS}),
        "bdt": experiment_tag("bdt_mass_tau32",
                              {short[k]: v for k, v in settings.items()},
                              {short[k]: v for k, v in BDT_DEFAULTS.items()}),
    }


def bdt_features(o: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 2) BDT inputs (M_jet [GeV], tau32)."""
    return np.column_stack([o[name] for name in BDT_FEATURES])


def fit_bdt(F_train: np.ndarray, y_train: np.ndarray, settings: dict[str, Any],
            seed: int):
    """HistGradientBoostingClassifier on train, early-stopped on a stratified
    ``validation_fraction`` of train (scikit-learn's own split, seeded)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        loss="log_loss", early_stopping=True, scoring="loss",
        random_state=seed, **settings)
    return clf.fit(F_train, y_train)


def bdt_size(clf) -> dict[str, Any]:
    """Trees and leaves of a fitted HistGradientBoostingClassifier."""
    try:   # private attribute; only used for the record
        leaves = int(sum(p.get_n_leaf_nodes() for it in clf._predictors for p in it))
    except AttributeError:
        leaves = None
    return {"n_iter": int(clf.n_iter_), "n_leaves_total": leaves,
            "final_validation_loss": float(-clf.validation_score_[-1])
            if len(getattr(clf, "validation_score_", [])) else None}


def evaluate(scores: list[np.ndarray], ys: list[np.ndarray]
             ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """(train_metrics, test_metrics, extra) of [train, val, test] scores.

    The accuracy cut is chosen on val and applied unchanged to test.
    """
    from sklearn.metrics import roc_auc_score
    (s_train, s_val, s_test), (y_train, y_val, y_test) = scores, ys
    threshold, val_acc = best_threshold(s_val, y_val)
    test_acc = float(((s_test > threshold).astype(np.int64) == y_test).mean())
    train_metrics = {"final_val_acc": val_acc,
                     "val_auc": float(roc_auc_score(y_val, s_val))}
    test_metrics = {"test_acc": test_acc, **binary_metrics(s_test, y_test)}
    extra = {"val_threshold": threshold,
             "train_auc": float(roc_auc_score(y_train, s_train))}
    return train_metrics, test_metrics, extra


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default="data",
                   help="Directory containing top_tagging_*.npz.")
    p.add_argument("--canonical-splits", action="store_true",
                   help="Published Kasieczka train/val/test files instead of a "
                        "random 70/15/15 re-split.")
    p.add_argument("--max-samples", type=int, default=100_000,
                   help="Jets read for the internal 70/15/15 split.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap on the canonical train jets (val/test stay full).")
    p.add_argument("--n-constituents", type=int, default=32,
                   help="Leading constituents per jet that enter the observables.")
    p.add_argument("--seed", type=int, default=0,
                   help="Internal / toy split seed and the BDT's random_state.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-scores", action="store_true",
                   help="Also write per-jet test scores and labels to "
                        "<experiment>__<model>__seed<N>__scores.npz.")
    p.add_argument("--toy-jets", type=int, default=None, metavar="N",
                   help="Use N toy massless jets (benchmarks.synthetic) instead "
                        "of the top-tagging files.")
    p.add_argument("--mass-centre", type=float, default=TOP_MASS,
                   help="Centre of the mass window -|M_jet - centre| in GeV.")
    p.add_argument("--cache-observables", type=str, default=None, metavar="PATH.npz",
                   help="Load the jet observables from this file if it holds them "
                        "for the same jets, else compute and save them there.")
    p.add_argument("--chunk-jets", type=int, default=CHUNK_JETS,
                   help="Jets per clustering block (memory bound; at most "
                        "2^21 / K^2 jets are used per block).")
    p.add_argument("--bdt-max-iter", type=int, default=BDT_DEFAULTS["max_iter"],
                   help="Maximum number of boosting iterations (trees).")
    p.add_argument("--bdt-learning-rate", type=float,
                   default=BDT_DEFAULTS["learning_rate"])
    p.add_argument("--bdt-max-leaf-nodes", type=int,
                   default=BDT_DEFAULTS["max_leaf_nodes"])
    p.add_argument("--bdt-min-samples-leaf", type=int,
                   default=BDT_DEFAULTS["min_samples_leaf"])
    p.add_argument("--bdt-validation-fraction", type=float,
                   default=BDT_DEFAULTS["validation_fraction"],
                   help="Stratified fraction of TRAIN held out for early stopping.")
    p.add_argument("--bdt-n-iter-no-change", type=int,
                   default=BDT_DEFAULTS["n_iter_no_change"],
                   help="Early-stopping patience in iterations.")
    args = p.parse_args(argv)

    if args.toy_jets is not None and args.toy_jets < 20:
        p.error("--toy-jets needs at least 20 jets")
    if args.n_constituents < 1 or args.chunk_jets < 1:
        p.error("--n-constituents and --chunk-jets must be >= 1")
    if args.mass_centre <= 0.0:
        p.error("--mass-centre must be positive (GeV)")
    if (args.bdt_max_iter < 1 or args.bdt_max_leaf_nodes < 2
            or args.bdt_min_samples_leaf < 1 or args.bdt_n_iter_no_change < 1
            or args.bdt_learning_rate <= 0.0
            or not 0.0 < args.bdt_validation_fraction < 1.0):
        p.error("BDT settings: max-iter >= 1, max-leaf-nodes >= 2, "
                "min-samples-leaf >= 1, n-iter-no-change >= 1, learning-rate > 0, "
                "0 < validation-fraction < 1")

    import sklearn

    experiment = experiment_name(args)
    names = model_names(args)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(argv)

    split = load_split(args)
    print(f"[{experiment}] {split.summary()}")
    dataset = split.name
    ys = [t.numpy().astype(np.int64) for t in (split.y_train, split.y_val, split.y_test)]
    obs, info = split_observables(split, chunk=args.chunk_jets,
                                  cache_path=args.cache_observables, label=experiment)
    del split                      # the constituents are no longer needed
    y_train, y_test = ys[0], ys[2]
    info["train_medians"] = {
        name: {"signal": float(np.median(obs[0][name][y_train == 1])),
               "background": float(np.median(obs[0][name][y_train == 0]))}
        for name in ("mass", "tau21", "tau32")}

    settings = bdt_settings(args)
    config_common: dict[str, Any] = {
        "n_constituents": args.n_constituents, "normalize": "none",
        "canonical_splits": args.canonical_splits, "max_samples": args.max_samples,
        "max_train_samples": args.max_train_samples, "toy_jets": args.toy_jets,
        "cache_dir": args.cache_dir, "cache_observables": args.cache_observables,
        "chunk_jets": args.chunk_jets, "threshold": "max val accuracy",
        "mass_centre": args.mass_centre,
        "observables": {"beta": BETA, "r0": R0, "axes": AXES,
                        "recombination": "E_scheme", "minimisation": None,
                        "ratio_if_zero_denominator": RATIO_IF_ZERO},
        "bdt": {**settings, "features": list(BDT_FEATURES), "loss": "log_loss",
                "early_stopping": "stratified validation_fraction of train",
                "scoring": "loss", "random_state": args.seed},
        "sklearn_version": sklearn.__version__,
    }

    def scores_of(fn) -> list[np.ndarray]:
        return [np.asarray(fn(o), dtype=np.float64) for o in obs]

    runs: list[dict[str, Any]] = []
    t = time.time()
    runs.append({"kind": "jet_mass", "scores": scores_of(lambda o: o["mass"]),
                 "n_params": 0, "n_features": 1, "model_size": "M_jet (0 params)",
                 "definition": "M_jet [GeV]", "fit_sec": time.time() - t})
    t = time.time()
    centre = args.mass_centre
    runs.append({"kind": "mass_window",
                 "scores": scores_of(lambda o: -np.abs(o["mass"] - centre)),
                 "n_params": 0, "n_features": 1,
                 "model_size": f"-|M_jet - {centre:g} GeV| (0 params)",
                 "definition": f"-|M_jet - {centre:g} GeV|", "fit_sec": time.time() - t})
    t = time.time()
    print(f"[{experiment}] fitting the BDT on (M_jet, tau32) "
          f"({len(y_train)} train jets) ...")
    clf = fit_bdt(bdt_features(obs[0]), y_train, settings, args.seed)
    bdt = {**bdt_size(clf), "max_iter": args.bdt_max_iter}
    bdt["early_stopped"] = bdt["n_iter"] < args.bdt_max_iter
    runs.append({"kind": "bdt",
                 "scores": scores_of(lambda o: clf.decision_function(bdt_features(o))),
                 "n_params": None, "n_features": len(BDT_FEATURES),
                 "model_size": (f"{bdt['n_iter']} trees x <= {args.bdt_max_leaf_nodes} "
                                f"leaves (max_iter {args.bdt_max_iter}"
                                f"{', early-stopped' if bdt['early_stopped'] else ''})"),
                 "definition": "HistGradientBoostingClassifier.decision_function"
                               "(M_jet, tau32)",
                 "fit_sec": time.time() - t, "bdt": bdt})

    for run in runs:
        model = names[run["kind"]]
        train_metrics, test_metrics, extra = evaluate(run["scores"], ys)
        record: dict[str, Any] = {
            "experiment":   experiment,
            "dataset":      dataset,
            "model":        model,
            "family":       "jet_observable",
            "seed":         args.seed,
            "n_params":     run["n_params"],
            "model_size":   run["model_size"],
            "n_train":      len(ys[0]),
            "n_val":        len(ys[1]),
            "n_test":       len(ys[2]),
            "n_features":   run["n_features"],
            "walltime_sec": info["observable_sec"] + run["fit_sec"],
            "diverged":     False,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "config":       {**config_common, "score": run["definition"]},
            "score":        {"definition": run["definition"], **extra,
                             "fit_sec": run["fit_sec"]},
            "observables":  info,
            "provenance":   meta,
        }
        if "bdt" in run:
            record["bdt"] = run["bdt"]
        stem = f"{experiment}__{model}__seed{args.seed}"
        (results_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
        if args.save_scores:
            np.savez_compressed(results_dir / f"{stem}__scores.npz",
                                scores=run["scores"][2].astype(np.float32),
                                labels=y_test.astype(np.int8))
        tm = test_metrics
        print(f"[{experiment}] {model}: val_acc={train_metrics['final_val_acc']:.4f} "
              f"test_acc={tm['test_acc']:.4f} test_auc={tm['test_auc']:.4f} "
              f"1/eB@0.3={tm['bg_rej_30']:.1f} 1/eB@0.5={tm['bg_rej_50']:.1f}  "
              f"({record['walltime_sec']:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
