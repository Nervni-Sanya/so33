"""
benchmarks.figure_k_robustness
------------------------------
Robustness to jet length: AUC and background rejection as a function of
the number of leading constituents K kept per jet.

Reads one results directory per K (``--sweep-root results_sweep`` with
subdirectories ``k4``, ``k8``, ...). Separate directories are mandatory:
the runner names files ``{experiment}__{model}__seed{seed}.json`` with no K
component, so a shared directory would silently overwrite cells.

Run:
    python -m benchmarks.figure_k_robustness --sweep-root results_sweep
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import defaultdict

from benchmarks.plotting import (
    DEFAULT_OUT_DIR, get_pyplot, load_results, mean_std, save, style_for, write_csv,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=str, default="results_sweep")
    p.add_argument("--experiment", type=str, default="top_tagging_constituents")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)

    root = pathlib.Path(args.sweep_root)
    cells = sorted(root.glob("k*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    if not cells:
        print(f"[k_robustness] no k* subdirectories under {root}", file=sys.stderr)
        return 1

    # (model, K) -> metric lists
    data: dict[str, dict[int, dict[str, tuple[float, float]]]] = defaultdict(dict)
    for cell in cells:
        k = int(re.sub(r"\D", "", cell.name))
        for r in load_results(cell, args.experiment):
            recs = data[r["model"]].setdefault(k, {"auc": [], "rej": []})
            recs["auc"].append(r["test_metrics"].get("test_auc"))
            recs["rej"].append(r["test_metrics"].get("bg_rej_30"))

    csv_rows = []
    plt = get_pyplot()
    fig = axes = None
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    for model, per_k in data.items():
        ks = sorted(per_k)
        stl = style_for(model)
        for metric, ax_i, ylabel in (("auc", 0, "AUC"),
                                     ("rej", 1, r"$1/\epsilon_B$ at $\epsilon_S=0.3$")):
            ys, es = [], []
            for k in ks:
                m, s = mean_std(per_k[k][metric])
                ys.append(m); es.append(s)
            if axes is not None:
                axes[ax_i].errorbar(ks, ys, yerr=es if any(es) else None,
                                    color=stl["color"], marker=stl["marker"],
                                    markersize=4, capsize=2, linewidth=1.2,
                                    label=stl["label"])
                axes[ax_i].set_xlabel("leading constituents kept, $K$")
                axes[ax_i].set_ylabel(ylabel)
                axes[ax_i].set_xscale("log", base=2)
        for k in ks:
            a, a_s = mean_std(per_k[k]["auc"])
            rj, rj_s = mean_std(per_k[k]["rej"])
            csv_rows.append([model, k, f"{a:.5f}", f"{a_s:.5f}", f"{rj:.1f}", f"{rj_s:.1f}"])

    write_csv(pathlib.Path(args.out_dir) / "k_robustness.csv",
              ["model", "K", "auc", "auc_std", "bg_rej_30", "bg_rej_30_std"], csv_rows)
    if fig is not None:
        axes[1].set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.12))
        save(fig, pathlib.Path(args.out_dir) / "k_robustness.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
