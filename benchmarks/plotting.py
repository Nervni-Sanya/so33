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

# One colour/marker per model, shared across every figure so a reader can
# track a model between plots. Greys and distinct markers keep the figures
# legible in black and white.
MODEL_STYLE: dict[str, dict[str, Any]] = {
    "so3c_equivariant_set":  dict(color="#1a1a1a", marker="o", label="SO3C equivariant (flow)"),
    "so3c_invariant_set":    dict(color="#c1121f", marker="s", label="SO3C invariant"),
    "so3c_interaction_set":  dict(color="#e07a00", marker="D", label="SO3C interaction"),
    "eta_invariants":        dict(color="#0353a4", marker="^", label=r"$\eta$-invariants (SO(3,3))"),
    "relu_bottleneck":       dict(color="#6c757d", marker="v", label="ReLU bottleneck"),
}
LITERATURE_STYLE = dict(color="#888888", marker="*", linestyle="none")


def style_for(model: str) -> dict[str, Any]:
    return MODEL_STYLE.get(
        model, dict(color="#444444", marker="x", label=model)
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
