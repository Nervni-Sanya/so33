"""
benchmarks.figure_binned
------------------------
Where does the gain come from? Performance sliced by physical jet
variables: jet transverse momentum, jet mass, and constituent
multiplicity.

Top tagging is a balanced binary task, so there is no rare-class axis to
slice; the field-standard equivalent is binning by kinematics, which also
shows the mechanism -- a mass-driven tagger and a substructure-driven one
degrade in different bins.

Scores come from the ``*__scores.npz`` files written alongside each result
JSON, so no model is retrained. The test split is reloaded with
``normalize="none"`` to recover physical GeV (normalisation is applied
after splitting, so the ordering is identical); alignment is asserted by
comparing the reloaded labels against the labels stored with the scores.

Run:
    python -m benchmarks.figure_binned --experiment top_tagging_constituents
    python -m benchmarks.figure_binned --experiment top_tagging_canonical \
        --canonical-splits
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from benchmarks.datasets import load_top_tagging_constituents
from benchmarks.plotting import (
    DEFAULT_OUT_DIR, get_pyplot, save, style_for, write_csv,
)

N_BINS = 6


def _jet_variables(X: np.ndarray) -> dict[str, np.ndarray]:
    """(N, K, 5) physical constituents -> jet pT, mass, multiplicity."""
    p4 = X[..., :4]
    mask = X[..., 4]
    p4 = p4 * mask[..., None]
    P = p4.sum(axis=1)                                   # (N, 4) = (E,px,py,pz)
    pt = np.sqrt(P[:, 1] ** 2 + P[:, 2] ** 2)
    m2 = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(axis=1)
    return {
        "jet_pT": pt,
        "jet_mass": np.sqrt(np.clip(m2, 0.0, None)),
        "multiplicity": mask.sum(axis=1),
    }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _rejection(labels: np.ndarray, scores: np.ndarray, eff: float = 0.3) -> float:
    from sklearn.metrics import roc_curve
    if labels.min() == labels.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = int((tpr >= eff).argmax())
    return float("inf") if fpr[idx] <= 0 else 1.0 / float(fpr[idx])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--experiment", type=str, default="top_tagging_constituents")
    p.add_argument("--models", type=str,
                   default="so3c_equivariant_set,so3c_invariant_set,eta_invariants")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=str, default="data")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--n-constituents", type=int, default=32)
    p.add_argument("--canonical-splits", action="store_true")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    res_dir = pathlib.Path(args.results_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Physical (unnormalised) test split -- same split, GeV units.
    split = load_top_tagging_constituents(
        cache_dir=args.cache_dir, max_samples=args.max_samples,
        n_constituents=args.n_constituents, seed=args.seed,
        standardise=True, normalize="none",
        use_canonical_splits=args.canonical_splits,
    )
    X_test = split.X_test.numpy()
    y_ref = split.y_test.numpy()
    variables = _jet_variables(X_test)

    series: dict[str, dict] = {}
    for model in models:
        f = res_dir / f"{args.experiment}__{model}__seed{args.seed}__scores.npz"
        if not f.is_file():
            print(f"[binned] missing {f.name}; skipping {model}", file=sys.stderr)
            continue
        d = np.load(f)
        scores, labels = d["scores"], d["labels"].astype(np.int64)
        if labels.shape != y_ref.shape or not np.array_equal(labels, y_ref):
            print(f"[binned] ABORT: stored labels for {model} do not match the "
                  f"reloaded test split -- the split is not reproducible with "
                  f"these arguments, so binning would mis-associate jets.",
                  file=sys.stderr)
            return 1
        series[model] = dict(scores=scores, labels=labels)

    if not series:
        print("[binned] no score files found", file=sys.stderr)
        return 1

    plt = get_pyplot()
    fig, axes = (None, None)
    if plt is not None:
        fig, axes = plt.subplots(1, len(variables), figsize=(3.0 * len(variables), 3.0),
                                 sharey=True)
    csv_rows = []

    for ax_i, (var_name, values) in enumerate(variables.items()):
        # Equal-population bins so every point carries the same statistics.
        edges = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
        edges[-1] += 1e-6
        centres = 0.5 * (edges[:-1] + edges[1:])
        for model, data in series.items():
            aucs = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (values >= lo) & (values < hi)
                aucs.append(_auc(data["labels"][sel], data["scores"][sel]))
                csv_rows.append([var_name, model, f"{lo:.3f}", f"{hi:.3f}",
                                 int(sel.sum()), f"{aucs[-1]:.5f}"])
            if axes is not None:
                stl = style_for(model)
                axes[ax_i].plot(centres, aucs, color=stl["color"],
                                marker=stl["marker"], markersize=4,
                                linewidth=1.2, label=stl["label"])
        if axes is not None:
            label = {"jet_pT": r"jet $p_T$ [GeV]", "jet_mass": "jet mass [GeV]",
                     "multiplicity": "constituent multiplicity"}[var_name]
            axes[ax_i].set_xlabel(label)
    if axes is not None:
        axes[0].set_ylabel("AUC within bin")
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.10))

    write_csv(out_dir / f"binned_{args.experiment}.csv",
              ["variable", "model", "bin_lo", "bin_hi", "n_jets", "auc"], csv_rows)
    if fig is not None:
        save(fig, out_dir / f"binned_{args.experiment}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
