"""
benchmarks.stats
----------------
Seed-level statistics for comparing models within one experiment.

For every model it reports the number of seeds, mean, sample standard
deviation (ddof = 1) and the 95% t-interval of the mean. Against a
baseline it reports
  • Welch's t-test on the two sets of seed values (unequal variances);
  • the paired per-seed difference (model - baseline) on the seeds both
    ran: mean, 95% t-interval and the paired t-test p-value.
Runs with the same seed share the data split (HIGGS, Adult and the
internal top-tagging split re-split per seed), so the paired interval is
the appropriate one; Welch is reported for completeness. With three seeds
both are wide — run eight or more (``--seeds 0-7`` on the runners).
Diverged runs are excluded and counted.

Run:
    python -m benchmarks.stats --experiment higgs --baseline relu_bottleneck
    python -m benchmarks.stats --experiment neutral --baseline tanh_mlp_w24 \\
        --models so33_multi,tanh_mlp --metric test_auc
    python -m benchmarks.stats --experiment boost_ood --baseline relu_mlp --metric ood_auc
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any

from benchmarks.aggregate import group_by_experiment_model, is_diverged, load_results


def metric_value(record: dict[str, Any], metric: str) -> float | None:
    """A metric from a result record: test-set metrics or top-level fields."""
    value = record.get("test_metrics", {}).get(metric, record.get(metric))
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def seed_values(records: list[dict], metric: str) -> tuple[dict[int, float], int]:
    """{seed: value} over non-diverged runs, and the number of diverged runs."""
    values, n_div = {}, 0
    for r in records:
        if is_diverged(r):
            n_div += 1
            continue
        v = metric_value(r, metric)
        if v is not None:
            values[int(r["seed"])] = v
    return values, n_div


def _t_interval(xs: list[float], level: float = 0.95) -> tuple[float, float, float, float]:
    """(mean, sample sd, low, high); the interval is NaN for fewer than 2 values."""
    from scipy import stats
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    half = stats.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n)
    return mean, sd, mean - half, mean + half


def compare(records: list[dict], experiment: str, baseline: str,
            metric: str = "test_auc", models: list[str] | None = None) -> list[dict]:
    """One row per model of ``experiment``: summary stats and the comparison
    with ``baseline`` (None entries where a test is undefined)."""
    from scipy import stats
    groups = group_by_experiment_model(records)
    by_model = {m: recs for (e, m), recs in groups.items() if e == experiment}
    if baseline not in by_model:
        raise ValueError(f"baseline {baseline!r} has no records in experiment {experiment!r}")
    names = models or sorted(by_model)
    base_vals, _ = seed_values(by_model[baseline], metric)
    rows = []
    for name in names:
        if name not in by_model:
            raise ValueError(f"model {name!r} has no records in experiment {experiment!r}")
        vals, n_div = seed_values(by_model[name], metric)
        xs = [vals[s] for s in sorted(vals)]
        row: dict[str, Any] = {"model": name, "n": len(xs), "n_diverged": n_div}
        if xs:
            row["mean"], row["sd"], row["ci_low"], row["ci_high"] = _t_interval(xs)
        if name != baseline and len(xs) >= 2 and len(base_vals) >= 2:
            b = [base_vals[s] for s in sorted(base_vals)]
            row["welch_p"] = float(stats.ttest_ind(xs, b, equal_var=False).pvalue)
            common = sorted(set(vals) & set(base_vals))
            if len(common) >= 2:
                diffs = [vals[s] - base_vals[s] for s in common]
                d_mean, d_sd, d_lo, d_hi = _t_interval(diffs)
                row.update(paired_n=len(common), diff_mean=d_mean,
                           diff_ci_low=d_lo, diff_ci_high=d_hi)
                row["paired_p"] = (float(stats.ttest_rel([vals[s] for s in common],
                                                         [base_vals[s] for s in common]).pvalue)
                                   if d_sd > 0 else (0.0 if d_mean != 0 else 1.0))
        rows.append(row)
    return rows


def render(rows: list[dict], experiment: str, baseline: str, metric: str) -> str:
    def f(x, fmt=".4f"):
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)

    lines = [
        f"## {experiment}: {metric} vs {baseline}",
        "",
        "| model | n | mean ± sd | 95% CI of mean | Welch p | paired diff [95% CI] | paired p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: -(r.get("mean") or -1)):
        name = r["model"] + (f" ({r['n_diverged']} diverged)" if r["n_diverged"] else "")
        if r["n"] == 0:
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        diff = (f"{r['diff_mean']:+.4f} [{f(r['diff_ci_low'], '+.4f')}, {f(r['diff_ci_high'], '+.4f')}]"
                if "diff_mean" in r else "—")
        lines.append(
            f"| {name} | {r['n']} | {f(r['mean'])} ± {f(r['sd'])} | "
            f"[{f(r['ci_low'])}, {f(r['ci_high'])}] | {f(r.get('welch_p'), '.3f')} | "
            f"{diff} | {f(r.get('paired_p'), '.3f')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated models (default: all in the experiment).")
    p.add_argument("--metric", type=str, default="test_auc",
                   help="test_auc / test_acc / bg_rej_30 (tabular) or "
                        "id_auc / ood_auc (boost_ood).")
    p.add_argument("--out", type=str, default=None, help="Optional markdown output path.")
    args = p.parse_args(argv)

    records = load_results(pathlib.Path(args.results_dir))
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    if models is not None and args.baseline not in models:
        models = [args.baseline] + models
    rows = compare(records, args.experiment, args.baseline, args.metric, models)
    out = render(rows, args.experiment, args.baseline, args.metric)
    print(out)
    if args.out:
        pathlib.Path(args.out).write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
