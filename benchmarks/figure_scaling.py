"""
benchmarks.figure_scaling
-------------------------
Does the architecture keep improving with size, and along which axis?

Two curves over the same log parameter axis:

  channels  -- the geometry axis: more complex channels, readout width fixed
  hidden    -- the generic-capacity axis: wider readout MLP, channels fixed

They are plotted together because their divergence is the result. Growing
`hidden` grows the ordinary MLP part, which a parameter-matched generic set
model already showed does not help (0.762 AUC at 9k, no better than at
1.8k). If the channel curve rises while the width curve flattens, capacity
has to be spent on geometry; if both rise together, that claim fails and
the paper must say so.

Reads the directory layout the Kaggle notebook produces:

    results_scaling/channels_<C>/*.json
    results_scaling/width_<H>/*.json

Run:
    python -m benchmarks.figure_scaling --results-dir results_scaling
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

from benchmarks.plotting import (
    DEFAULT_OUT_DIR, FIGSIZE_WIDE, LITERATURE_STYLE,
    get_pyplot, save, write_csv,
)

AXIS_STYLE = {
    "channels": dict(color="#111111", marker="o",
                     label="channels (geometry axis)"),
    "width":    dict(color="#b2182b", marker="s",
                     label="readout width (generic-capacity axis)"),
}


def _load_axis(root: pathlib.Path, prefix: str) -> list[dict]:
    """Collect (knob, params, auc, rejection) for one scaling axis."""
    rows = []
    for d in sorted(root.glob(f"{prefix}_*")):
        if not d.is_dir():
            continue
        try:
            knob = int(d.name.split("_")[-1])
        except ValueError:
            continue
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            tm = r.get("test_metrics", {})
            if tm.get("test_auc") is None:
                continue
            rows.append({
                "axis": prefix, "knob": knob, "model": r["model"],
                "n_params": r["n_params"], "auc": tm["test_auc"],
                "rej": tm.get("bg_rej_30", float("nan")),
                "hours": r.get("walltime_sec", float("nan")) / 3600.0,
            })
    rows.sort(key=lambda r: r["n_params"])
    return rows


def _load_literature(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default="results_scaling")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--literature", type=str,
                   default=str(DEFAULT_OUT_DIR / "literature_reference.csv"))
    args = p.parse_args(argv)

    root = pathlib.Path(args.results_dir)
    if not root.is_dir():
        print(f"[scaling] no such directory: {root}", file=sys.stderr)
        return 1

    series = {"channels": _load_axis(root, "channels"),
              "width": _load_axis(root, "width")}
    if not any(series.values()):
        print(f"[scaling] no results under {root}/(channels|width)_*",
              file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out_dir)
    csv_rows = [[r["axis"], r["knob"], r["model"], r["n_params"],
                 f"{r['auc']:.5f}", f"{r['rej']:.1f}", f"{r['hours']:.2f}"]
                for rows in series.values() for r in rows]
    write_csv(out_dir / "scaling.csv",
              ["axis", "knob", "model", "n_params", "auc", "bg_rej_30", "hours"],
              csv_rows)

    plt = get_pyplot()
    if plt is None:
        return 0
    fig, (ax_auc, ax_rej) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    lit = _load_literature(pathlib.Path(args.literature))

    for ax, key, ylabel, logy in (
        (ax_auc, "auc", "AUC", False),
        (ax_rej, "rej", r"background rejection $1/\epsilon_B$ at $\epsilon_S=0.3$", True),
    ):
        for axis, rows in series.items():
            if not rows:
                continue
            stl = AXIS_STYLE[axis]
            ax.plot([r["n_params"] for r in rows], [r[key] for r in rows],
                    color=stl["color"], marker=stl["marker"], markersize=5,
                    linewidth=1.3, label=stl["label"], zorder=3)
        for l in lit:
            val = float(l["auc"] if key == "auc" else l["bg_rej_30"])
            ax.plot(float(l["n_params"]), val, markerfacecolor="none",
                    markersize=9, zorder=2, **LITERATURE_STYLE)
            ax.annotate(l["model"], (float(l["n_params"]), val),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6, color=LITERATURE_STYLE["color"])
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = ax_auc.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lb in zip(handles, labels):
        seen.setdefault(lb, h)
    seen["published (quoted, not re-verified)"] = plt.Line2D(
        [], [], markerfacecolor="none", markersize=9, linestyle="none",
        color=LITERATURE_STYLE["color"], marker="*")
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.14))

    save(fig, out_dir / "scaling_params_vs_performance.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
