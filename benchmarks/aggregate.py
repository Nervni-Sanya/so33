"""
benchmarks.aggregate
--------------------
Aggregate JSON result files into summary tables.

Each ``run_<experiment>.py`` writes:
    results/<experiment>__<model>__seed<N>.json

This script scans the results directory, groups by (experiment, model)
across seeds, and prints a markdown-style table. Designed to be paste-
ready into the paper or a notebook.

Run:
    python -m benchmarks.aggregate
    python -m benchmarks.aggregate --experiment synthetic_equivariance
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def load_results(results_dir: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def group_by_experiment_model(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["experiment"], r["model"])].append(r)
    return groups


def render_equivariance(groups: dict[tuple[str, str], list[dict]]) -> str:
    rows = []
    for (exp, model), recs in groups.items():
        if exp != "synthetic_equivariance":
            continue
        clean   = [r["equivariance"]["val_acc_clean"]          for r in recs]
        transf  = [r["equivariance"]["val_acc_transformed"]    for r in recs]
        cons    = [r["equivariance"]["prediction_consistency"] for r in recs]
        rows.append((model, len(recs), _mean_std(clean), _mean_std(transf), _mean_std(cons)))

    if not rows:
        return ""
    lines = [
        "## synthetic_equivariance",
        "",
        "| model | n_seeds | clean_acc | transformed_acc | consistency |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, n, c, t, k in sorted(rows, key=lambda r: -r[2][0]):
        lines.append(
            f"| {model} | {n} | {c[0]:.3f}±{c[1]:.3f} | "
            f"{t[0]:.3f}±{t[1]:.3f} | {k[0]:.3f}±{k[1]:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_dataeff(groups: dict[tuple[str, str], list[dict]]) -> str:
    pertinent = [(m, recs) for (e, m), recs in groups.items() if e == "synthetic_dataeff"]
    if not pertinent:
        return ""

    # Discover the fraction grid.
    fractions = sorted({float(k) for _, recs in pertinent for k in recs[0]["curve"].keys()})
    lines = [
        "## synthetic_dataeff (val_acc)",
        "",
        "| model | " + " | ".join(f"{f:.2f}" for f in fractions) + " |",
        "|---|" + "|".join(["---:"] * len(fractions)) + "|",
    ]
    for model, recs in sorted(pertinent):
        means = []
        for f in fractions:
            xs = [r["curve"][f"{f:.2f}"]["final_val_acc"] for r in recs]
            m, _s = _mean_std(xs)
            means.append(f"{m:.3f}")
        lines.append(f"| {model} | " + " | ".join(means) + " |")
    return "\n".join(lines) + "\n"


def render_tabular(groups: dict[tuple[str, str], list[dict]]) -> str:
    """Render real-data tabular experiments (HIGGS / Top Tagging / Neutral).

    Splits each experiment into matched-bottleneck and natural-width
    sub-tables based on the per-record ``family`` field. Reports
    val_acc, test_acc, and AUC (when present).
    """
    tabular_experiments: dict[str, list[dict]] = defaultdict(list)
    for (exp, _model), recs in groups.items():
        for r in recs:
            if "family" in r and "test_metrics" in r:
                tabular_experiments[exp].append(r)

    if not tabular_experiments:
        return ""

    out_lines: list[str] = []
    for exp in sorted(tabular_experiments.keys()):
        records = tabular_experiments[exp]
        out_lines.append(f"## {exp}")
        out_lines.append("")
        for family in ("matched_bottleneck", "natural_width"):
            family_recs = [r for r in records if r["family"] == family]
            if not family_recs:
                continue
            by_model: dict[str, list[dict]] = defaultdict(list)
            for r in family_recs:
                by_model[r["model"]].append(r)

            has_auc = any(r["test_metrics"].get("test_auc") is not None
                          for r in family_recs)

            label = "matched bottleneck (hidden=6)" if family == "matched_bottleneck" else "natural width MLPs"
            out_lines.append(f"### {label}")
            out_lines.append("")
            header_cells = ["model", "n_seeds", "params", "val_acc", "test_acc"]
            if has_auc:
                header_cells.append("test_auc")
            out_lines.append("| " + " | ".join(header_cells) + " |")
            out_lines.append("|" + "|".join(["---"] + ["---:"] * (len(header_cells) - 1)) + "|")

            rows = []
            for model, recs in by_model.items():
                n_params  = recs[0]["n_params"]
                val_acc   = _mean_std([r["train_metrics"]["final_val_acc"] for r in recs])
                test_acc  = _mean_std([r["test_metrics"]["test_acc"]       for r in recs])
                if has_auc:
                    test_auc = _mean_std([r["test_metrics"]["test_auc"]
                                          for r in recs
                                          if r["test_metrics"].get("test_auc") is not None])
                else:
                    test_auc = None
                rows.append((model, len(recs), n_params, val_acc, test_acc, test_auc))

            sort_key = -1
            if has_auc:
                rows.sort(key=lambda r: -(r[5][0] if r[5] else -1))
            else:
                rows.sort(key=lambda r: -r[4][0])

            for model, n, p, v, t, a in rows:
                cells = [model, str(n), str(p),
                         f"{v[0]:.3f}±{v[1]:.3f}",
                         f"{t[0]:.3f}±{t[1]:.3f}"]
                if has_auc:
                    cells.append(f"{a[0]:.3f}±{a[1]:.3f}" if a else "—")
                out_lines.append("| " + " | ".join(cells) + " |")
            out_lines.append("")
    return "\n".join(out_lines)


def render_ood(groups: dict[tuple[str, str], list[dict]]) -> str:
    rows = []
    for (exp, model), recs in groups.items():
        if exp != "synthetic_ood":
            continue
        id_acc  = [r["id_val_acc"] for r in recs]
        ood_acc = [r["ood_acc"]    for r in recs]
        gap     = [r["ood_gap"]    for r in recs]
        rows.append((model, len(recs), _mean_std(id_acc), _mean_std(ood_acc), _mean_std(gap)))

    if not rows:
        return ""
    lines = [
        "## synthetic_ood",
        "",
        "| model | n_seeds | id_val | ood | gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, n, i, o, g in sorted(rows, key=lambda r: r[4][0]):
        lines.append(
            f"| {model} | {n} | {i[0]:.3f}±{i[1]:.3f} | "
            f"{o[0]:.3f}±{o[1]:.3f} | {g[0]:+.3f}±{g[1]:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Filter to a single experiment (default: all).")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write the markdown summary.")
    args = parser.parse_args(argv)

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"No results directory at {results_dir}", file=sys.stderr)
        return 1

    records = load_results(results_dir)
    if args.experiment:
        records = [r for r in records if r["experiment"] == args.experiment]
    if not records:
        print("No matching result files.", file=sys.stderr)
        return 1

    groups = group_by_experiment_model(records)

    parts: list[str] = []
    parts.append(render_equivariance(groups))
    parts.append(render_dataeff(groups))
    parts.append(render_ood(groups))
    parts.append(render_tabular(groups))
    out = "\n".join(p for p in parts if p)

    print(out)
    if args.out:
        pathlib.Path(args.out).write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
