"""
benchmarks.figure_boost_ood
---------------------------
Figures for the boost-OOD experiment, from the records written by
``benchmarks.run_boost_ood``:

  fig_boost_ood_rapidity.pdf : test AUC versus boost rapidity (binned over
      the union of the ID and OOD test sets), mean over seeds per model.
      The shaded band is the training range.
  fig_boost_ood_readout.pdf  : distribution of the Architecture-B readout
      feature <h, h>_eta (one channel) for ID and OOD inputs, under the
      Euclidean input bound (so33_equivariant) and the eta rescaling
      (so33_equivariant_eta_bounded). The flow conserves <h, h>_eta, so
      this is <beta(x), beta(x)>_eta: rapidity-independent for beta_eta,
      but shrinking with ||x||_2 for beta_euc, which is what moves the
      OOD inputs out of the region the head was trained on.
  fig_boost_ood_coeff_norms.pdf : training trajectory of the connection
      coefficient norm ||c||_2 (mean over channels and seeds) for every
      model with SO33 activations. With one particle per sample the
      Architecture-B readout does not depend on c, so only the Frobenius
      penalty acts on it and ||c|| decays towards 0.

Run (after run_boost_ood):
    python -m benchmarks.figure_boost_ood --results-dir results
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np

READOUT_MODELS = (("so33_equivariant", r"$\beta_\mathrm{euc}$"),
                  ("so33_equivariant_eta_bounded", r"$\beta_\eta$"))


def load_records(results_dir: pathlib.Path, experiment: str) -> list[dict]:
    recs = []
    for p in sorted(results_dir.glob(f"{experiment}__*__seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("experiment") == experiment and "binned_auc" in r:
            recs.append(r)
    return recs


def binned_auc_by_model(recs: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """model -> (bin centres, mean AUC over seeds, sd over seeds)."""
    by_model: dict[str, list[list[dict]]] = collections.defaultdict(list)
    for r in recs:
        if not r.get("diverged"):
            by_model[r["model"]].append(r["binned_auc"])
    out = {}
    for model, runs in by_model.items():
        centres = np.array([(b["lo"] + b["hi"]) / 2 for b in runs[0]])
        aucs = np.array([[b["auc"] for b in run] for run in runs], dtype=float)
        mean = np.nanmean(aucs, axis=0) if len(aucs) else aucs
        sd = (np.nanstd(aucs, axis=0, ddof=1) if len(aucs) > 1
              else np.zeros_like(mean))
        out[model] = (centres, mean, sd)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--experiment", type=str, default="boost_ood")
    p.add_argument("--out-dir", type=str, default="paper/figures")
    p.add_argument("--channel", type=int, default=0,
                   help="Architecture-B channel whose <h,h> is histogrammed.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed whose per-sample file is used for the histograms.")
    args = p.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for the figures", file=sys.stderr)
        return 1

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = load_records(results_dir, args.experiment)
    if not recs:
        print(f"no {args.experiment} records with binned_auc under {results_dir}",
              file=sys.stderr)
        return 1

    # (a) AUC versus rapidity
    curves = binned_auc_by_model(recs)
    train_rap = recs[0]["train_rapidity"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axvspan(0.0, train_rap, color="0.9", label="training range")
    styles = ("-", "--", ":", "-.")       # identical curves (e.g. frozen c) stay visible
    for i, model in enumerate(sorted(curves, key=lambda m: -np.nanmean(curves[m][1]))):
        centres, mean, sd = curves[model]
        line, = ax.plot(centres, mean, marker="o", markersize=3, label=model,
                        linestyle=styles[i % len(styles)])
        if np.any(sd > 0):
            ax.fill_between(centres, mean - sd, mean + sd, alpha=0.15,
                            color=line.get_color(), linewidth=0)
    ax.set_xlabel("boost rapidity of the test sample")
    ax.set_ylabel("test AUC in rapidity bin")
    ax.set_ylim(0.45, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    path_a = out_dir / "fig_boost_ood_rapidity.pdf"
    fig.savefig(path_a)
    print(f"Wrote {path_a}")

    # (c) ||c|| training trajectories (from the per-epoch history)
    traj: dict[str, list[list[float]]] = collections.defaultdict(list)
    for r in recs:
        hist = [h for h in r.get("history", []) if "coeff_norms" in h]
        if hist and not r.get("diverged"):
            traj[r["model"]].append([float(np.mean(h["coeff_norms"])) for h in hist])
    if traj:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for model in sorted(traj):
            n = min(len(t) for t in traj[model])
            curve = np.mean([t[:n] for t in traj[model]], axis=0)
            ax.plot(np.arange(1, n + 1), curve, label=model)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$\|c\|_2$ (mean over channels and seeds)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        path_c = out_dir / "fig_boost_ood_coeff_norms.pdf"
        fig.savefig(path_c)
        print(f"Wrote {path_c}")

    # (b) <h, h>_eta distributions, ID vs OOD, beta_euc vs beta_eta
    panels = []
    for model, label in READOUT_MODELS:
        f = results_dir / f"{args.experiment}__{model}__seed{args.seed}__samples.npz"
        if f.exists():
            d = np.load(f)
            if "eta_norms" in d:
                panels.append((label, d))
    if not panels:
        print("no Architecture-B per-sample files; skipping the readout figure")
        return 0
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4),
                             squeeze=False)
    for ax, (label, d) in zip(axes[0], panels):
        hh = d["eta_norms"][:, args.channel]
        lo, hi = np.percentile(hh, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        for split, name, style in ((0, "ID", "-"), (1, "OOD", "--")):
            for cls, colour in ((0, "tab:blue"), (1, "tab:red")):
                sel = (d["split"] == split) & (d["labels"] == cls)
                ax.hist(hh[sel], bins=bins, histtype="step", linestyle=style,
                        color=colour, density=True,
                        label=f"{name}, m={'1' if cls == 0 else '3'}")
        ax.set_title(f"Readout B, {label}")
        ax.set_xlabel(r"$\langle h, h\rangle_\eta$ (channel %d)" % args.channel)
        ax.set_ylabel("density")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path_b = out_dir / "fig_boost_ood_readout.pdf"
    fig.savefig(path_b)
    print(f"Wrote {path_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
