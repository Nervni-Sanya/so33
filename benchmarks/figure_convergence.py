"""
benchmarks.figure_convergence
-----------------------------
Validation accuracy per epoch for the set models, mean +- spread over
seeds. Reads the ``history`` list persisted in each result JSON.

Run:
    python -m benchmarks.figure_convergence
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np

from benchmarks.plotting import (
    DEFAULT_OUT_DIR, FIGSIZE, get_pyplot, load_results, save, style_for, write_csv,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--experiment", type=str, default="top_tagging_constituents")
    p.add_argument("--models", type=str,
                   default="so3c_equivariant_set,so3c_invariant_set,eta_invariants")
    p.add_argument("--metric", type=str, default="val_acc",
                   choices=("val_acc", "val_loss", "train_acc", "train_loss"))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    records = [r for r in load_results(args.results_dir, args.experiment)
               if r["model"] in wanted and r.get("history")]
    if not records:
        print(f"[convergence] no records with history for {args.experiment!r}. "
              f"Re-run the experiment after the history patch.", file=sys.stderr)
        return 1

    by_model: dict[str, list[list[float]]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append([h[args.metric] for h in r["history"]])

    plt = get_pyplot()
    fig = ax = None
    if plt is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE)

    csv_rows = []
    for model in wanted:
        curves = by_model.get(model)
        if not curves:
            continue
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])           # (seeds, epochs)
        mean, std = arr.mean(axis=0), arr.std(axis=0, ddof=1 if len(arr) > 1 else 0)
        epochs = np.arange(1, n + 1)
        for e, m, s in zip(epochs, mean, std):
            csv_rows.append([model, int(e), f"{m:.5f}", f"{s:.5f}", len(arr)])
        if ax is not None:
            stl = style_for(model)
            ax.plot(epochs, mean, color=stl["color"], label=stl["label"], linewidth=1.4)
            if len(arr) > 1:
                ax.fill_between(epochs, mean - std, mean + std,
                                color=stl["color"], alpha=0.18, linewidth=0)

    write_csv(pathlib.Path(args.out_dir) / f"convergence_{args.metric}.csv",
              ["model", "epoch", "mean", "std", "n_seeds"], csv_rows)
    if fig is not None:
        ax.set_xlabel("epoch")
        ax.set_ylabel({"val_acc": "validation accuracy",
                       "val_loss": "validation loss",
                       "train_acc": "training accuracy",
                       "train_loss": "training loss"}[args.metric])
        ax.legend(frameon=False, loc="lower right")
        save(fig, pathlib.Path(args.out_dir) / f"convergence_{args.metric}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
