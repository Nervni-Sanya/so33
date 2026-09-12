"""
benchmarks.figure_boost_robustness
----------------------------------
AUC of a trained top tagger when its test jets are pushed through random
Lorentz transformations of growing size.

Two panels, one axis each:

(a) AUC against the boost scale, every model. The broken
    ``so3c_equivariant_set`` -- a flow whose connection is built from
    invariants, so it applies the same rotation in every frame instead of
    the conjugated one -- falls towards and then below chance, while every
    exactly equivariant model is a flat line.
(b) AUC minus its unboosted value, exactly equivariant models only, on an
    axis zoomed to the resolution that matters. One shared axis cannot do
    both jobs: the flat lines sit within 0.015 of one another on an axis
    that must reach down to the collapse, i.e. inside two percent of its
    height.

The x variable is the ``boost_scale`` passed to
``so3c.lift.random_lorentz_pair``: each of the three rapidity components is
drawn from N(0, s). It is a scale, not a rapidity -- a draw at s = 2 can
reach rapidity 5 -- and the axis label says so.

Error bars are the spread over random group elements at that scale when
there is one seed, or over seeds when there are several; the CSV twin
records which. The exact models are dodged horizontally by a few hundredths
in both panels so their overlapping markers and error bars stay readable;
the dodge is cosmetic and does not reach the CSV.

Reads the output of benchmarks/run_boost_robustness.py:
    python -m benchmarks.figure_boost_robustness --results-dir results_boost
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

from benchmarks.plotting import (
    DEFAULT_OUT_DIR, FIGSIZE_WIDE, get_pyplot, load_results, mean_std, save,
    style_for, write_csv,
)

# Chart chrome from the validated reference palette (dataviz skill). Series
# colours come from plotting.MODEL_STYLE; text never wears a series colour.
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BROKEN = "so3c_equivariant_set"
ORDER = ("so3c_message_set", BROKEN, "so3c_covariant_set", "so3c_invariant_set")
XLABEL = r"boost scale $s$ (rapidity components $\sim\mathcal{N}(0, s)$)"


def _curves(records: list[dict]) -> dict[str, dict]:
    """model -> per-scale mean, spread and provenance."""
    # Keyed by model AND tag: a beams variant carries the same model name as
    # the beamless one, and merging them would average two different models
    # as though they were seeds of one.
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if "curve" in r:
            by_model[r["model"] + str(r.get("tag") or "")].append(r)

    out: dict[str, dict] = {}
    for model, recs in by_model.items():
        scales = sorted({pt["rapidity"] for r in recs for pt in r["curve"]})
        cells = {}
        for s in scales:
            pts = [pt for r in recs for pt in r["curve"] if pt["rapidity"] == s]
            if len(recs) == 1:
                mean, spread, over = pts[0]["auc"], pts[0].get("auc_std", 0.0), "draws"
            else:
                mean, spread = mean_std([pt["auc"] for pt in pts])
                over = "seeds"
            cells[s] = dict(auc=mean, std=spread, over=over,
                            draws=pts[0].get("draws", 1), n_seeds=len(recs))
        out[model] = dict(
            cells=cells,
            n_params=recs[0].get("n_params"),
            base=recs[0]["model"],
            tag=str(recs[0].get("tag") or ""),
            dtypes=sorted({str(r.get("dtype", "not recorded")) for r in recs}),
        )
    return out


def _style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.5, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, length=2)


def _mark(ax, xs, ys, es, model, label):
    stl = style_for(model)
    ax.errorbar(xs, ys, yerr=es if any(es) else None,
                color=stl["color"], marker=stl["marker"], markersize=6,
                markeredgecolor="white", markeredgewidth=1.0,
                linewidth=1.4, elinewidth=0.8, capsize=2,
                label=label, zorder=3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default="results_boost")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dtype-note", type=str, default=None,
                   help="Precision to print on panel (b) when the records do "
                        "not carry it. Records written before the dtype field "
                        "was added need this; newer ones are read directly.")
    args = p.parse_args(argv)

    curves = _curves(load_results(args.results_dir))
    if not curves:
        print(f"[boost_robustness] no boost records under {args.results_dir}",
              file=sys.stderr)
        return 1
    def rank(key):
        base = curves[key]["base"]
        return (ORDER.index(base) if base in ORDER else len(ORDER), key)

    models = sorted(curves, key=rank)

    rows = []
    for m in models:
        c = curves[m]
        base_scale = min(c["cells"])
        base = c["cells"][base_scale]["auc"]
        for s, cell in sorted(c["cells"].items()):
            rows.append([m, c["n_params"], s, f"{cell['auc']:.5f}",
                         f"{cell['std']:.5f}", f"{cell['auc'] - base:+.5f}",
                         cell["over"], cell["draws"], cell["n_seeds"],
                         "/".join(c["dtypes"])])
    out_dir = pathlib.Path(args.out_dir)
    write_csv(out_dir / "boost_robustness.csv",
              ["model", "n_params", "boost_scale", "auc", "auc_std",
               "delta_auc", "spread_over", "draws", "n_seeds", "dtype"], rows)

    plt = get_pyplot()
    if plt is None:
        return 0

    recorded = {d for m in models for d in curves[m]["dtypes"]}
    dtype_note = (next(iter(recorded))
                  if len(recorded) == 1 and "not recorded" not in recorded
                  else args.dtype_note)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for ax in (ax_a, ax_b):
        _style_axis(ax)

    # The exact models sit within 0.015 of one another in (a) and on top of
    # one another at zero in (b); a small horizontal dodge keeps each
    # marker visible in both panels.
    exact = [m for m in models if curves[m]["base"] != BROKEN]
    dodge = {m: (i - (len(exact) - 1) / 2) * 0.06 for i, m in enumerate(exact)}

    # (a) every model, full AUC range.
    lo = 1.0
    for m in models:
        c = curves[m]
        xs = sorted(c["cells"])
        ys = [c["cells"][x]["auc"] for x in xs]
        es = [c["cells"][x]["std"] for x in xs]
        lo = min(lo, min(y - e for y, e in zip(ys, es)))
        label = style_for(c["base"])["label"] + (" + beams" if c["tag"] else "")
        _mark(ax_a, [x + dodge.get(m, 0.0) for x in xs], ys, es, c["base"], label)
        if curves[m]["base"] == BROKEN:
            # Label the one series the panel is about, at its endpoint, in ink.
            ax_a.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]),
                          textcoords="offset points", xytext=(7, 0),
                          va="center", fontsize=7, color=INK_SECONDARY)
    x_min = min(min(curves[m]["cells"]) for m in models)
    ax_a.axhline(0.5, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax_a.text(x_min, 0.5, " chance", va="bottom", ha="left",
              fontsize=7, color=INK_MUTED)
    ax_a.set_ylim(max(0.0, lo - 0.03), 1.0)
    ax_a.set_xlabel(XLABEL)
    ax_a.set_ylabel("test AUC")
    ax_a.set_title("(a) all models", fontsize=8, loc="left",
                   color=INK_SECONDARY)

    # (b) exact models, change from the unboosted value, zoomed.
    lim = 0.0
    base_label = None
    for m in exact:
        c = curves[m]
        xs = sorted(c["cells"])
        base = c["cells"][xs[0]]["auc"]
        base_label = xs[0] if base_label is None else base_label
        ds = [c["cells"][x]["auc"] - base for x in xs]
        es = [c["cells"][x]["std"] for x in xs]
        lim = max(lim, max(abs(d) + e for d, e in zip(ds, es)))
        _mark(ax_b, [x + dodge[m] for x in xs], ds, es, c["base"], None)
    ax_b.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    lim = max(1.6 * lim, 1e-4)
    ax_b.set_ylim(-lim, lim)
    ax_b.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_b.set_xlabel(XLABEL)
    ax_b.set_ylabel(f"AUC $-$ AUC at $s={base_label:g}$")
    ax_b.set_title("(b) exactly equivariant models, zoomed", fontsize=8,
                   loc="left", color=INK_SECONDARY)
    if dtype_note:
        ax_b.text(0.98, 0.04, dtype_note, transform=ax_b.transAxes,
                  ha="right", va="bottom", fontsize=7, color=INK_MUTED)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    save(fig, out_dir / "boost_robustness.pdf")
    save(fig, out_dir / "boost_robustness.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
