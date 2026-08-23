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

Panels: AUC vs jet pT, AUC vs jet mass, and background rejection vs jet
mass. Constituent multiplicity is available via --variables but is NOT
shown by default: it is censored at K (every saturated jet reports K), and
its low bins are almost pure background (signal fraction 0.008-0.17 at
K=32), so per-bin AUC there reflects class composition rather than
discrimination. Jet-length dependence is measured properly by
figure_k_robustness instead.

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


MIN_PER_CLASS = 50    # per-bin metrics need this many jets of EACH class


def _usable(labels: np.ndarray) -> bool:
    """A bin needs enough signal AND background to score.

    Total count is not enough: the lowest jet-mass bin is ~99.99% QCD, so a
    rejection at 30% signal efficiency there is set by a handful of signal
    jets and is pure noise (it produced a spurious 95 for the eta baseline
    before this guard).
    """
    if labels.size == 0:
        return False
    n_sig = int((labels == 1).sum())
    return min(n_sig, labels.size - n_sig) >= MIN_PER_CLASS


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if not _usable(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _rejection(labels: np.ndarray, scores: np.ndarray, eff: float = 0.3) -> float:
    from sklearn.metrics import roc_curve
    if not _usable(labels):
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
    p.add_argument("--panels", type=str, default="jet_pT:auc,jet_mass:auc,jet_mass:rej",
                   help="Comma-separated variable:metric panels. Variables: "
                        "jet_pT, jet_mass, multiplicity. Metrics: auc, rej.")
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

    panels = []
    for spec in args.panels.split(","):
        var, _, metric = spec.strip().partition(":")
        if var not in variables:
            print(f"[binned] unknown variable {var!r}", file=sys.stderr)
            return 1
        panels.append((var, metric or "auc"))

    plt = get_pyplot()
    fig = axes = None
    if plt is not None:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0))
        if len(panels) == 1:
            axes = [axes]
    csv_rows = []

    AXIS_LABEL = {"jet_pT": r"jet $p_T$ [GeV]", "jet_mass": "jet mass [GeV]",
                  "multiplicity": "retained constituents (censored at $K$)"}
    METRIC_LABEL = {"auc": "AUC within bin",
                    "rej": r"$1/\epsilon_B$ at $\epsilon_S=0.3$ within bin"}

    for ax_i, (var_name, metric) in enumerate(panels):
        values = variables[var_name]
        # Equal-population bins so every point carries the same statistics.
        edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
        if edges.size < 3:
            edges = np.unique(values)
            edges = np.append(edges, edges[-1] + 1.0)
        edges = edges.astype(float)
        edges[-1] += 1e-6
        centres = 0.5 * (edges[:-1] + edges[1:])
        fn = _auc if metric == "auc" else _rejection

        for model, data in series.items():
            ys = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (values >= lo) & (values < hi)
                ys.append(fn(data["labels"][sel], data["scores"][sel]))
                csv_rows.append([var_name, metric, model, f"{lo:.3f}", f"{hi:.3f}",
                                 int(sel.sum()),
                                 f"{data['labels'][sel].mean():.4f}" if sel.sum() else "",
                                 f"{ys[-1]:.5f}"])
            if axes is not None:
                stl = style_for(model)
                arr = np.array(ys, dtype=float)
                ok = np.isfinite(arr)
                axes[ax_i].plot(centres[ok], arr[ok], color=stl["color"],
                                marker=stl["marker"], markersize=4,
                                linewidth=1.2, label=stl["label"])
        if axes is not None:
            axes[ax_i].set_xlabel(AXIS_LABEL[var_name])
            axes[ax_i].set_ylabel(METRIC_LABEL[metric])
            if metric == "rej":
                axes[ax_i].set_yscale("log")

    if axes is not None:
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.12))

    write_csv(out_dir / f"binned_{args.experiment}.csv",
              ["variable", "metric", "model", "bin_lo", "bin_hi", "n_jets",
               "signal_fraction", "value"], csv_rows)
    if fig is not None:
        save(fig, out_dir / f"binned_{args.experiment}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
