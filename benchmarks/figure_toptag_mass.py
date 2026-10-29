"""
benchmarks.figure_toptag_mass
-----------------------------
How much of a tagger is the jet mass? Figure and numbers for the canonical
top-tagging test set (fig_toptag_mass_dominance.pdf in the rewrite).

Inputs: per-jet test scores written with ``--save-scores``
(``<experiment>__<model>__seed<N>__scores.npz``: scores, labels) and the
test constituents, from which the jet mass of the same leading-K
constituents is computed in GeV.

Panel (a): background rejection 1/eps_B versus signal efficiency eps_S for
each score file, the zero-parameter jet mass M and the mass window
-|M - centre|. Published working points can be added as stars with
``--published "Name:1/eps_B@0.3"`` (no published numbers are built in;
quote them from the papers).
Panel (b): AUC of each tagger within jet-mass bins; the dashed lines are
the inclusive AUCs.

Also writes ``<out-dir>/toptag_mass_dominance.json`` with the inclusive
AUC and 1/eps_B of every curve and the per-bin AUCs.

Run:
    python -m benchmarks.figure_toptag_mass --cache-dir data --canonical-splits \\
        --scores "Readout A=results/top_tagging_canonical__eta_invariants__seed0__scores.npz"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

DEFAULT_BINS = (0.0, 100.0, 140.0, 160.0, 180.0, 200.0, np.inf)


def jet_mass(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Invariant mass sqrt(max(E^2 - |p|^2, 0)) of the summed real constituents.

    p4 : (n, K, 4) (E, px, py, pz) in GeV; mask : (n, K), 1 for real constituents.
    """
    P = (p4 * mask[..., None]).sum(axis=1)
    return np.sqrt(np.clip(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(axis=1), 0.0, None))


def rejection_curve(scores: np.ndarray, labels: np.ndarray):
    """(eps_S, 1/eps_B) along the ROC curve, eps_B > 0 only."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    keep = fpr > 0
    return tpr[keep], 1.0 / fpr[keep]


def binned_auc(scores: np.ndarray, labels: np.ndarray, mass: np.ndarray,
               edges=DEFAULT_BINS, min_count: int = 50) -> list[dict]:
    from sklearn.metrics import roc_auc_score
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mass >= lo) & (mass < hi)
        n = int(sel.sum())
        ok = n >= min_count and len(np.unique(labels[sel])) == 2
        out.append({"lo": float(lo), "hi": float(hi), "n": n,
                    "auc": float(roc_auc_score(labels[sel], scores[sel])) if ok else None})
    return out


def _load_test(args):
    """Raw-GeV test constituents (n, K, 4), mask (n, K) and labels, in the
    order the runners score them."""
    if args.canonical_splits:
        # The canonical test file alone; no need to read the 1.2M train jets.
        from benchmarks.datasets import _stack_constituent_jets
        X, y, mask = _stack_constituent_jets(pathlib.Path(args.cache_dir), None,
                                             args.n_constituents, split="test")
        return X.numpy().astype(np.float64), mask.numpy().astype(np.float64), y.numpy()
    from benchmarks.datasets import load_top_tagging_constituents
    split = load_top_tagging_constituents(
        args.cache_dir, max_samples=args.max_samples, n_constituents=args.n_constituents,
        seed=args.seed, normalize="none")
    X = split.X_test.numpy().astype(np.float64)
    return X[..., :4], X[..., 4], split.y_test.numpy()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", action="append", default=[],
                   help='"Label=path/to/..._scores.npz"; repeatable.')
    p.add_argument("--cache-dir", type=str, default="data")
    p.add_argument("--canonical-splits", action="store_true")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--n-constituents", type=int, default=32)
    p.add_argument("--seed", type=int, default=0,
                   help="Split seed (internal split only; must match the scores).")
    p.add_argument("--mass-centre", type=float, default=172.5,
                   help="Centre of the zero-parameter mass window, GeV.")
    p.add_argument("--published", action="append", default=[],
                   help='"Name:value" of 1/eps_B at eps_S = 0.3, drawn as a star.')
    p.add_argument("--out-dir", type=str, default="paper/figures")
    args = p.parse_args(argv)

    p4, mask, labels = _load_test(args)
    mass = jet_mass(p4, mask)
    curves: list[tuple[str, np.ndarray]] = []
    for item in args.scores:
        label, _, path = item.partition("=")
        d = np.load(path)
        if len(d["scores"]) != len(labels) or not np.array_equal(d["labels"], labels):
            print(f"{path}: scores do not match the test split (length or label order)",
                  file=sys.stderr)
            return 1
        curves.append((label, d["scores"].astype(np.float64)))
    curves.append(("jet mass $M$", mass))
    curves.append((rf"mass window $-|M-{args.mass_centre:g}|$", -np.abs(mass - args.mass_centre)))

    from benchmarks.tabular_runner import binary_metrics
    summary = {"n_test": int(len(labels)), "n_constituents": args.n_constituents,
               "mass_bins_gev": [b for b in DEFAULT_BINS if np.isfinite(b)], "curves": {}}
    for label, s in curves:
        summary["curves"][label] = {**binary_metrics(s, labels),
                                    "binned_auc": binned_auc(s, labels, mass)}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for the figure", file=sys.stderr)
        return 1
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for label, s in curves:
        eff, rej = rejection_curve(s, labels)
        ax_a.plot(eff, rej, label=label)
    for item in args.published:
        name, _, value = item.partition(":")
        ax_a.plot([0.3], [float(value)], marker="*", markersize=10, linestyle="none",
                  label=f"{name} (published)")
    ax_a.set_yscale("log")
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_xlabel(r"signal efficiency $\varepsilon_S$")
    ax_a.set_ylabel(r"background rejection $1/\varepsilon_B$")
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(fontsize=7, frameon=False)

    edges = [b if np.isfinite(b) else 300.0 for b in DEFAULT_BINS]
    centres = [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
    for label, s in curves[:-2]:
        bins = summary["curves"][label]["binned_auc"]
        line, = ax_b.plot(centres, [b["auc"] if b["auc"] is not None else np.nan for b in bins],
                          marker="o", label=label)
        ax_b.axhline(summary["curves"][label]["test_auc"], linestyle="--",
                     color=line.get_color(), linewidth=0.8)
    ax_b.set_xticks(centres)
    ax_b.set_xticklabels([f"<{edges[1]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in
                         zip(edges[1:-2], edges[2:-1])] + [f">{edges[-2]:g}"], fontsize=7)
    ax_b.set_xlabel(r"jet mass bin [GeV] (leading constituents)")
    ax_b.set_ylabel("AUC within the bin")
    ax_b.grid(True, alpha=0.3)
    if len(curves) > 2:
        ax_b.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_toptag_mass_dominance.pdf")
    (out_dir / "toptag_mass_dominance.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'fig_toptag_mass_dominance.pdf'} and toptag_mass_dominance.json")
    for label, info in summary["curves"].items():
        print(f"  {label:<40} AUC {info['test_auc']:.4f}  1/eB@0.3 {info['bg_rej_30']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
