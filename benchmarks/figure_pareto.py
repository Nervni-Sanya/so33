"""
benchmarks.figure_pareto
------------------------
Headline figure: performance versus model size on canonical top tagging.

Two panels share a log-scaled parameter axis: AUC on the left, background
rejection 1/eps_B at eps_S = 0.3 on the right. Our models (measured on one
machine, mean +- std over seeds) use filled markers; published numbers use
open stars and are read from ``paper/figures/literature_reference.csv``.

The published values are transcribed and NOT independently verified (the
same caveat is flagged in paper/main.tex); the CSV carries a ``verified``
column and the figure labels them explicitly so the distinction survives
into the paper.

Run:
    python -m benchmarks.figure_pareto
    python -m benchmarks.figure_pareto --experiment top_tagging_constituents
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

from benchmarks.plotting import (
    DEFAULT_OUT_DIR, FIGSIZE_WIDE, LITERATURE_STYLE,
    get_pyplot, load_results, mean_std, save, style_for, write_csv,
)


def _load_literature(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        print(f"[pareto] no literature file at {path}; plotting our models only.")
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default="results",
                   help="Comma-separated result directories. Pass the "
                        "parameter-matched baseline dir alongside the main "
                        "one to show the generic model at our size.")
    p.add_argument("--experiment", type=str, default="top_tagging_canonical")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--literature", type=str,
                   default=str(DEFAULT_OUT_DIR / "literature_reference.csv"))
    p.add_argument("--include", type=str,
                   default="so3c_equivariant_set,so3c_invariant_set,"
                           "eta_invariants,relu_mlp",
                   help="Comma-separated models to plot, in the paper's "
                        "curated order. Empty string plots everything found "
                        "(useful for exploration, too cluttered for print).")
    p.add_argument("--exclude", type=str, default="relu_bottleneck",
                   help="Comma-separated models to keep out of the plot (they "
                        "stay in the CSV). The non-equivariant baseline sits at "
                        "AUC 0.76 and would flatten the interesting range.")
    args = p.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    records = []
    for d in args.results_dir.split(","):
        d = d.strip()
        if d:
            records.extend(load_results(d, args.experiment))
    if not records:
        print(f"[pareto] no records for experiment={args.experiment!r}", file=sys.stderr)
        return 1

    by_model: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        by_model[(r["model"], r["n_params"])].append(r)
    sizes_per_name: dict[str, set[int]] = defaultdict(set)
    for name, n_par in by_model:
        sizes_per_name[name].add(n_par)

    rows = []
    for (model, _n_par), recs in by_model.items():
        aucs = [r["test_metrics"].get("test_auc") for r in recs]
        rejs = [r["test_metrics"].get("bg_rej_30") for r in recs]
        if not any(a is not None for a in aucs):
            continue
        auc_m, auc_s = mean_std(aucs)
        rej_m, rej_s = mean_std(rejs)
        n_par = recs[0]["n_params"]
        label = model
        if len(sizes_per_name[model]) > 1:
            label = f"{model} ({n_par/1000:.1f}k)"
        rows.append({
            "model": model, "label": label,
            "n_params": n_par, "n_seeds": len(recs),
            "auc": auc_m, "auc_std": auc_s, "rej": rej_m, "rej_std": rej_s,
        })
    rows.sort(key=lambda r: r["n_params"])

    lit = _load_literature(pathlib.Path(args.literature))

    write_csv(
        out_dir / "pareto_data.csv",
        ["model", "n_params", "n_seeds", "auc", "auc_std", "bg_rej_30", "bg_rej_30_std", "origin"],
        [[r["model"], r["n_params"], r["n_seeds"], f"{r['auc']:.5f}", f"{r['auc_std']:.5f}",
          f"{r['rej']:.1f}", f"{r['rej_std']:.1f}", "measured"] for r in rows]
        + [[l["model"], l["n_params"], "", l["auc"], "", l["bg_rej_30"], "", "quoted"] for l in lit],
    )

    plt = get_pyplot()
    if plt is None:
        return 0

    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}
    plot_rows = [r for r in rows if r["model"] not in excluded]
    included = [m.strip() for m in args.include.split(",") if m.strip()]
    if included:
        plot_rows = [r for r in plot_rows if r["model"] in included]
        # Keep the generic baseline only at its parameter-matched size: the
        # point of showing it is "same capacity, no geometry".
        matched = max((r["n_params"] for r in plot_rows
                       if r["model"].startswith("so3c")), default=None)
        if matched is not None:
            plot_rows = [r for r in plot_rows
                         if r["model"].startswith(("so3c", "eta"))
                         or abs(r["n_params"] - matched) / matched < 0.25]

    fig, (ax_auc, ax_rej) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ax, key, std_key, ylabel in (
        (ax_auc, "auc", "auc_std", "AUC"),
        (ax_rej, "rej", "rej_std", r"background rejection $1/\epsilon_B$ at $\epsilon_S=0.3$"),
    ):
        for r in plot_rows:
            stl = style_for(r["model"])
            # r["label"] carries the size suffix when one model name appears
            # at two parameter counts (the generic baseline at 1.8k and at 9k).
            lbl = r.get("label", r["model"])
            legend_label = stl["label"] if lbl == r["model"] else f"{stl['label']} [{lbl.split('(')[-1][:-1]}]"
            ax.errorbar(r["n_params"], r[key], yerr=r[std_key] or None,
                        color=stl["color"], marker=stl["marker"], markersize=6,
                        capsize=2, linestyle="none", label=legend_label, zorder=3)
        offsets = [(5, 4), (5, -9), (-6, 6), (-6, -11)]
        for i, l in enumerate(lit):
            val = float(l["auc"] if key == "auc" else l["bg_rej_30"])
            ax.plot(float(l["n_params"]), val, markerfacecolor="none",
                    markersize=9, zorder=2, **LITERATURE_STYLE)
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(l["model"], (float(l["n_params"]), val),
                        textcoords="offset points", xytext=(dx, dy),
                        ha="left" if dx > 0 else "right",
                        fontsize=6, color="#666666")
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel(ylabel)
        ax.set_xlim(2e3, 2e6)
    ax_rej.set_yscale("log")
    # Focus the AUC axis on the band where every equivariant model lives.
    finite = [r["auc"] for r in plot_rows] + [float(l["auc"]) for l in lit]
    lo = min(finite)
    ax_auc.set_ylim(lo - 0.008, 1.0)

    handles, labels = ax_auc.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lb in zip(handles, labels):
        seen.setdefault(lb, h)
    star = plt.Line2D([], [], markerfacecolor="none", markersize=9,
                      color=LITERATURE_STYLE["color"], marker="*", linestyle="none")
    seen["published (quoted, not re-verified)"] = star
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))

    save(fig, out_dir / "pareto_params_vs_performance.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
