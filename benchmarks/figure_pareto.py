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

The form is emphasis. Measured models wear categorical colours; everything
published -- the stars and PELICAN's own size sweep -- is context in muted
ink. A scatter can put any two marks side by side, so its colours must pass
the palette validator on all pairs, which the reference palette allows for
three series at most: the generic baselines are folded into one series
(plotting.MODEL_STYLE) and the script refuses to colour a fourth. Published
points are named in panel (b) only, where the log axis spreads them apart;
in panel (a) they all share a band a few hundredths of AUC tall.

Run:
    python -m benchmarks.figure_pareto \
        --results-dir results_matched_canonical,results_kappa/k64
    python -m benchmarks.figure_pareto --experiment top_tagging_constituents
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections import defaultdict

from benchmarks.plotting import (
    DEFAULT_OUT_DIR, FIGSIZE_WIDE, INK_MUTED, INK_SECONDARY, LITERATURE_STYLE,
    MAX_SCATTER_SERIES, get_pyplot, load_results, mean_std, save, style_for,
    write_csv,
)

# Where a published point's name may go, tried in order: right, left, above,
# below, then the diagonals. The outer rings sit further out and draw a
# hairline leader back to their point. Offsets in points, with the anchor.
_RING = ((5, 0, "left", "center"), (-5, 0, "right", "center"),
         (0, 6, "center", "bottom"), (0, -6, "center", "top"),
         (5, 5, "left", "bottom"), (-5, 5, "right", "bottom"),
         (5, -5, "left", "top"), (-5, -5, "right", "top"))
NAME_SLOTS = tuple((dx * reach, dy * reach, ha, va, reach > 1)
                   for reach in (1, 3, 5) for dx, dy, ha, va in _RING)


def _load_literature(path: pathlib.Path) -> list[dict]:
    """Published points, one per model.

    The file keeps every evaluation we verified, including the cases where
    two papers report the same model differently -- ParticleNet is 1615 +- 93
    in the LorentzNet table and 1298 +- 46 in PELICAN's, a 24% spread that is
    worth knowing before reading anyone's rejection number to three digits.
    Only rows marked primary go on the figure.
    """
    if not path.is_file():
        print(f"[pareto] no literature file at {path}; plotting our models only.")
        return []
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get("primary", "yes") == "yes"]


def _load_scaling(path: pathlib.Path) -> list[dict]:
    """PELICAN's own size sweep (its table 2).

    This is the comparison that matters for a small-model claim, and it is
    unforgiving: PELICAN reaches AUC 0.9850 and rejection 1494 at 3k
    parameters. Any Pareto argument has to be made against this curve, not
    against the 208k headline model.
    """
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return sorted(csv.DictReader(fh), key=lambda r: float(r["n_params"]))


def _name_points(fig, ax, names, markers) -> list[str]:
    """Name points on ``ax`` so that every name is legible and unambiguous.

    ``names`` holds (text, x, y, marker size in points) in data units;
    ``markers`` holds (x, y, size in points) for every mark on the axes. Each
    name takes the first slot in NAME_SLOTS whose box stays inside the axes
    and clears every marker and every name already placed. A name beside its
    point, without a leader, must also sit nearer its own mark than any other,
    or it reads as naming the neighbour. A name with no such slot is left out
    rather than drawn over another -- the CSV twin still carries it. Points
    closer together than a marker radius share one joint name. Returns the
    names left out.
    """
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    fig.canvas.draw()                  # fix limits and layout before measuring
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    centres, taken = [], []
    for x, y, size in markers:
        px, py = ax.transData.transform((x, y))
        r = size * fig.dpi / 72.0 / 2.0
        centres.append((px, py))
        taken.append(Bbox.from_extents(px - r, py - r, px + r, py + r))

    def gap(box, px, py):
        return math.hypot(max(box.x0 - px, 0.0, px - box.x1),
                          max(box.y0 - py, 0.0, py - box.y1))

    # LorentzNet and PELICAN overlap in panel (b). Two names would compete for
    # the same few slots and neither could say which star is which, so points
    # that share a spot are named once, together.
    spots: list[list] = []
    for text, x, y, size in names:
        px, py = ax.transData.transform((x, y))
        radius = size * fig.dpi / 72.0 / 2.0
        for spot in spots:
            if math.hypot(px - spot[3], py - spot[4]) <= radius:
                spot[0].append(text)
                break
        else:
            spots.append([[text], x, y, px, py, radius])

    left_out = []
    for texts, x, y, ox, oy, same_spot in spots:
        text = " / ".join(texts)
        for dx, dy, ha, va, leader in NAME_SLOTS:
            t = ax.annotate(
                text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontsize=6, color=INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.5,
                                shrinkA=0, shrinkB=5) if leader else None)
            t.set_in_layout(False)
            # The text alone: an annotation's own extent includes its leader.
            t.update_positions(renderer)
            box = Text.get_window_extent(t, renderer).padded(1.0)
            clear = (frame.x0 <= box.x0 and box.x1 <= frame.x1
                     and frame.y0 <= box.y0 and box.y1 <= frame.y1
                     and not any(box.overlaps(o) for o in taken))
            if clear and not leader:
                own = gap(box, ox, oy)
                clear = all(gap(box, px, py) >= own for px, py in centres
                            if math.hypot(px - ox, py - oy) > same_spot)
            if clear:
                taken.append(box)
                break
            t.remove()
        else:
            left_out.append(text)
    return left_out


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
    p.add_argument("--scaling", type=str,
                   default=str(DEFAULT_OUT_DIR / "pelican_scaling.csv"),
                   help="A published model's own parameter sweep, drawn as a "
                        "curve. Empty string to omit it.")
    p.add_argument("--include", type=str,
                   default="so3c_covariant_set,eta_invariants,relu_mlp,gelu_mlp",
                   help="Comma-separated models to plot, in the paper's "
                        "curated order. Empty string plots everything found "
                        "(useful for exploration, too cluttered for print). "
                        f"Either way at most {MAX_SCATTER_SERIES} colours are "
                        "drawn.")
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
    scaling = _load_scaling(pathlib.Path(args.scaling)) if args.scaling else []

    write_csv(
        out_dir / "pareto_data.csv",
        ["model", "n_params", "n_seeds", "auc", "auc_std", "bg_rej_30", "bg_rej_30_std", "origin"],
        [[r["model"], r["n_params"], r["n_seeds"], f"{r['auc']:.5f}", f"{r['auc_std']:.5f}",
          f"{r['rej']:.1f}", f"{r['rej_std']:.1f}", "measured"] for r in rows]
        + [[l["model"], l["n_params"], "", l["auc"], "", l["bg_rej_30"], "", "published"]
           for l in lit]
        + [[f"PELICAN({r['n_params']})", r["n_params"], "", r["auc"], "",
            r["bg_rej_30"], "", "published"] for r in scaling],
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

    # Folded models share a colour, so count colours, not models. Past the
    # all-pairs cap two of them could not be told apart, and no choice of
    # colours fixes that -- stop rather than draw it.
    colours = {style_for(r["model"])["color"]: style_for(r["model"])["label"]
               for r in plot_rows}
    if len(colours) > MAX_SCATTER_SERIES:
        print(f"[pareto] {len(colours)} coloured series "
              f"({'; '.join(colours.values())}), but a scatter carries at most "
              f"{MAX_SCATTER_SERIES}. Narrow --include, or facet.", file=sys.stderr)
        return 2

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
        if scaling:
            ax.plot([float(r["n_params"]) for r in scaling],
                    [float(r["auc"] if key == "auc" else r["bg_rej_30"])
                     for r in scaling],
                    color=INK_MUTED, linewidth=1.0, marker=".", markersize=4,
                    linestyle="--", zorder=1,
                    label="PELICAN, its own size sweep")
        for l in lit:
            val = float(l["auc"] if key == "auc" else l["bg_rej_30"])
            ax.plot(float(l["n_params"]), val, markerfacecolor="none",
                    markersize=9, zorder=2, **LITERATURE_STYLE)
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel(ylabel)
    ax_rej.set_yscale("log")
    ax_auc.set_title("(a)", loc="left", fontsize=8, color=INK_SECONDARY)
    ax_rej.set_title("(b)", loc="left", fontsize=8, color=INK_SECONDARY)
    # Every point on the axis. A fixed 2e3-2e6 range clipped both ends: the
    # small half of PELICAN's sweep (248-1000 parameters), EFP at 1k, and
    # ParT at 2.1M.
    xs = ([r["n_params"] for r in plot_rows] + [float(l["n_params"]) for l in lit]
          + [float(r["n_params"]) for r in scaling])
    if xs:
        for ax in (ax_auc, ax_rej):
            ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    # Headroom above the best rejection, so the names at the top of panel (b)
    # have somewhere to go other than on top of each other.
    rejs = ([float(l["bg_rej_30"]) for l in lit]
            + [float(r["bg_rej_30"]) for r in scaling]
            + [r["rej"] for r in plot_rows if math.isfinite(r["rej"])])
    if rejs:
        ax_rej.set_ylim(top=max(rejs) * 2.0)
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
    seen["published, verified against the source table (named in b)"] = star
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))

    fig.tight_layout()
    markers = ([(float(l["n_params"]), float(l["bg_rej_30"]), 9) for l in lit]
               + [(float(r["n_params"]), float(r["bg_rej_30"]), 4) for r in scaling]
               + [(r["n_params"], r["rej"], 6) for r in plot_rows])
    names = sorted(((l["model"], float(l["n_params"]), float(l["bg_rej_30"]), 9)
                    for l in lit), key=lambda n: -n[2])
    left_out = _name_points(fig, ax_rej, names, markers)
    if left_out:
        print(f"[pareto] no free spot to name {', '.join(left_out)} in panel (b); "
              f"they stay in the CSV.", file=sys.stderr)

    save(fig, out_dir / "pareto_params_vs_performance.pdf")
    save(fig, out_dir / "pareto_params_vs_performance.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
