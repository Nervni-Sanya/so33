"""
benchmarks.plotting
-------------------
Shared plotting helpers for the paper figures.

Conventions follow the existing ``figure_equivariance.py``: matplotlib is
imported lazily behind the Agg backend (so figure scripts degrade to
CSV-only output on a machine without it), figures are small single-column
PDFs, and every figure writes a CSV twin next to it so the numbers stay
inspectable and editable without re-running anything.

seaborn is deliberately not used (absent in this environment).
"""

from __future__ import annotations

import csv
import pathlib
import statistics as st
from typing import Any, Iterable, Sequence

DEFAULT_OUT_DIR = pathlib.Path("paper/figures")
FIGSIZE = (4.6, 3.2)
FIGSIZE_WIDE = (7.2, 3.2)

# Chart ink from the reference palette (dataviz skill, references/palette.md).
# Labels and annotations wear these, never a series colour.
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

# A scatter can put any two marks side by side, and the reference palette
# passes all pairs for three colours at most; a figure that would colour more
# has to fold or facet instead.
MAX_SCATTER_SERIES = 3

# One colour/marker per model, shared across every figure so a reader can
# track a model between plots. Distinct markers keep the figures legible in
# black and white, and are the secondary encoding the palette rules require.
#
# Every colour is a light-mode slot of the reference categorical palette. Its
# slot order only guarantees neighbours when a figure uses slots 1..N; these
# colours follow the model, so a figure that draws a subset skips slots and
# puts side by side colours the order kept apart. Each figure's co-occurring
# set is therefore validated on its own -- all pairs for the scatter, and all
# pairs for the three-line figures too, which then holds however their lines
# cross. validate_palette.py, all with --mode light --surface "#ffffff" (the
# figures are print PDFs on white); dE is OKLab x100, CVD the worse of
# protan/deutan:
#
#   "#2a78d6,#008300,#1baf7a,#eda100"      boost_robustness (a), drawn order
#       -> PASS, worst adjacent CVD 9.1, normal 15.6
#   "#2a78d6,#1baf7a,#eda100" --pairs all  boost_robustness (b), exact models
#       -> PASS, CVD 9.1, normal 22.9
#   "#008300,#eda100,#4a3aa7" --pairs all  convergence, binned x2, k_robustness
#       -> PASS, CVD 16.2, normal 30.3
#   "#1baf7a,#eb6834" --pairs all          pareto (scatter)
#       -> PASS, CVD 9.2, normal 27.6
#   "#1baf7a,#4a3aa7,#eb6834" --pairs all  pareto with eta included
#       -> PASS, CVD 9.2, normal 27.6
#   "#4a3aa7,#e34948" --pairs all          the two SO(3,3) baselines
#       -> PASS, CVD 22.7, normal 33.6
#
# so3c_equivariant_set left orange #eb6834 on 2026-09-11: four figures draw it
# beside so3c_invariant_set with no so3c_covariant_set between them, and
# "#eb6834,#eda100" measures normal dE 13.7, under the floor of 15, whatever
# colour eta takes. Of the single changes, green is the only one that passes
# every set above without a CVD warning and also lets eta share a figure with
# either generic baseline.
# Pairs that FAIL, so these models must never share a figure: the generic
# MLP with so3c_invariant_set (normal 13.7), so3c_interaction_set (12.9),
# so3c_equivariant_set (CVD 3.2) or relu_bottleneck (normal 7.1), and
# so3c_interaction_set with relu_bottleneck (normal 13.2). Aqua, yellow and
# magenta sit below 3:1 on white, which obliges a relief channel: every
# figure writes a CSV twin (the table view) and keeps a legend with markers.
_GENERIC_SET_MLP = dict(color="#eb6834", marker="h",
                        label="generic Deep Sets MLP (ReLU, GELU)")
MODEL_STYLE: dict[str, dict[str, Any]] = {
    "so3c_message_set":      dict(color="#2a78d6", marker="o", label="SO3C message passing"),
    "so3c_equivariant_set":  dict(color="#008300", marker="X", label="SO3C flow, invariant connection"),
    "so3c_covariant_set":    dict(color="#1baf7a", marker="P", label="SO3C flow, covariant connection"),
    "so3c_invariant_set":    dict(color="#eda100", marker="s", label="SO3C invariant (no flow)"),
    "so3c_interaction_set":  dict(color="#e87ba4", marker="D", label="SO3C interaction"),
    "eta_invariants":        dict(color="#4a3aa7", marker="^", label=r"$\eta$-invariants (SO(3,3))"),
    # Folded into one series: the parameter-matched generic baselines are one
    # comparison (same capacity, no geometry) and land on one point on the
    # canonical protocol -- 9053 parameters, AUC 0.7636 / 0.7633, rejection
    # 8.5 / 8.6. The CSV twins keep them apart.
    "relu_mlp":              _GENERIC_SET_MLP,
    "gelu_mlp":              _GENERIC_SET_MLP,
    "relu_bottleneck":       dict(color="#e34948", marker="v", label="ReLU bottleneck"),
}
LITERATURE_STYLE = dict(color=INK_MUTED, marker="*", linestyle="none")


def style_for(model: str) -> dict[str, Any]:
    # A model without an entry is context, drawn in muted ink rather than
    # given a colour that nothing has validated.
    return MODEL_STYLE.get(
        model, dict(color=INK_MUTED, marker="x", label=model)
    )


def get_pyplot():
    """Return pyplot with the Agg backend, or None if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # pragma: no cover
        print(f"[plotting] matplotlib unavailable ({e}); CSV only.")
        return None
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })
    return plt


def mean_std(xs: Sequence[float]) -> tuple[float, float]:
    """Mean and sample std; std is 0.0 for a single observation."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return st.mean(xs), st.stdev(xs)


def write_csv(path: pathlib.Path, header: Sequence[str],
              rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[plotting] wrote {path}")


def save(fig, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[plotting] wrote {path}")


def load_results(results_dir: pathlib.Path | str,
                 experiment: str | None = None) -> list[dict]:
    """Load result JSONs, optionally filtered by experiment name.

    Records without a ``model`` key (e.g. the diagnose_equivariant dump)
    are skipped.
    """
    import json
    out = []
    for p in sorted(pathlib.Path(results_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(r, dict) or "model" not in r:
            continue
        if experiment is not None and r.get("experiment") != experiment:
            continue
        out.append(r)
    return out
