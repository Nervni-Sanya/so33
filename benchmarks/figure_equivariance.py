"""
benchmarks.figure_equivariance
------------------------------
Measure the empirical equivariance of SO33Activation as a function of
the connection-coefficient norm, for each of the three input-bound
modes. Writes a CSV (paper/figures/equivariance_data.csv), a two-panel
PDF (paper/figures/fig_equivariance_two_metrics.pdf: eps_sigma and eps_m)
and the single eps_m panel (paper/figures/equivariance_vs_norm.pdf).

Two quantities are measured at each (||c||, bound) cell
(see ``benchmarks.equivariance``):
  1. Activation-level rel err (eps_sigma)
       E_b [ || sigma(g x) - g sigma(x) ||_2 / || sigma(x) ||_2 ]
  2. Readout-feature invariance error (eps_m)
       E_b [ | <sigma(x),sigma(x)>_eta - <sigma(gx),sigma(gx)>_eta | ]

The flow conserves <v, v>_eta for every connection, so eps_m only tests
whether the input bound is invariant; it is blind to the connection.
eps_sigma measures equivariance of the activation itself, and is
non-zero for every c != 0 whatever the bound. The figure shows both.
Cells whose flow overflowed (non-finite or > 1e3) are drawn as crosses
on the top edge instead of being dropped.

Run:
    python -m benchmarks.figure_equivariance
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys

import torch

from so33 import SO33Activation
from so33.basis import ETA, N_BASIS
from benchmarks.equivariance import activation_equivariance
from benchmarks.synthetic import random_so33_element

# Values above this (or non-finite) are drawn as "overflow" markers.
PLOT_CEILING = 1e3


def make_act(mode: str | bool) -> SO33Activation:
    return SO33Activation(
        T=0.3, method="rk4", solver_options={"step_size": 0.03},
        adjoint=False, bound_input=mode,
    )


def measure(act: SO33Activation, x: torch.Tensor, gs: torch.Tensor,
            eta: torch.Tensor) -> tuple[float, float]:
    """Return (activation rel err, readout-feature abs err) averaged over gs."""
    res = activation_equivariance(act, x, gs)
    return res["eps_sigma"], res["eps_m"]


PANELS = {
    2: r"$\varepsilon_\sigma$: $\|\sigma(gx)-g\sigma(x)\|/\|\sigma(x)\|$",
    3: r"$\varepsilon_m$: $|m^2(\sigma x)-m^2(\sigma gx)|$",
}


def _plot(rows: list[tuple], pdf_path: pathlib.Path,
          columns: tuple[int, ...]) -> pathlib.Path | None:
    """One log-y panel per CSV column (2 = eps_sigma, 3 = eps_m)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping PDF.", file=sys.stderr)
        return None

    labels = ["none", "euclidean", "eta"]
    markers = {"none": "o", "euclidean": "s", "eta": "^"}
    # Overflow crosses of different bounds at the same ||c|| would coincide;
    # shift them sideways a little.
    offsets = {"none": 0.85, "euclidean": 1.0, "eta": 1.18}
    fig, axes = plt.subplots(1, len(columns), figsize=(4.6 * len(columns), 3.4),
                             squeeze=False)
    axes = axes[0]
    for ax, col in zip(axes, columns):
        for label in labels:
            data = [(r[0], r[col]) for r in rows if r[1] == label]
            ok = [(c, v) for c, v in data if math.isfinite(v) and v <= PLOT_CEILING]
            bad = [c for c, v in data if not (math.isfinite(v) and v <= PLOT_CEILING)]
            line, = ax.plot([c for c, _ in ok], [max(v, 1e-16) for _, v in ok],
                            marker=markers[label], label=f"bound = {label}")
            if bad:
                ax.plot([c * offsets[label] for c in bad], [PLOT_CEILING] * len(bad),
                        linestyle="none", marker="x", markersize=8,
                        color=line.get_color())
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, PLOT_CEILING * 4)
        ax.set_xlabel(r"$\|c\|_2$ (symlog axis, linear below 0.01)")
        ax.set_ylabel(PANELS[col])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[-1].text(0.02, 0.97,
                  rf"$\times$: diverged (non-finite or $>10^{{{int(math.log10(PLOT_CEILING))}}}$)",
                  transform=axes[-1].transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path)
    return pdf_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-inputs", type=int, default=256)
    p.add_argument("--n-groups", type=int, default=16)
    p.add_argument("--rapidity", type=float, default=0.6,
                   help="Sampling rapidity for the test group elements.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="paper/figures")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    eta = ETA.to(torch.float64)

    # Inputs of moderate magnitude (similar to lifted physical 4-momenta).
    x = torch.randn(args.n_inputs, 6, dtype=torch.float64) * 2.0

    # Group elements at the chosen rapidity scale.
    gen = torch.Generator().manual_seed(args.seed + 1)
    gs = torch.stack([
        random_so33_element(rapidity=args.rapidity, generator=gen,
                            dtype=torch.float64)
        for _ in range(args.n_groups)
    ])

    # Sweep over coefficient norms by rescaling a fixed direction. Using a
    # single random direction keeps the connection structure consistent
    # across cells so the trends are about magnitude, not random structure.
    base_dir = torch.randn(N_BASIS, dtype=torch.float64)
    base_dir = base_dir / base_dir.norm()
    norms = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    bounds = [("none", False), ("euclidean", True), ("eta", "eta")]

    rows = []
    for cnorm in norms:
        for label, mode in bounds:
            act = make_act(mode)
            with torch.no_grad():
                act.coeffs.copy_(cnorm * base_dir.to(act.coeffs.dtype))
            act_err, feat_err = measure(act, x, gs, eta)
            rows.append((cnorm, label, act_err, feat_err))
            print(f"||c||={cnorm:5.2f}  bound={label:9s}  "
                  f"act_rel_err={act_err:.3e}  feat_abs_err={feat_err:.3e}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "equivariance_data.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coeff_norm", "bound", "act_rel_err", "feat_abs_err"])
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # Two panels (eps_sigma, eps_m) for the rewrite; the single eps_m panel
    # keeps the original file name used by the first manuscript version.
    for name, columns in (("fig_equivariance_two_metrics.pdf", (2, 3)),
                          ("equivariance_vs_norm.pdf", (3,))):
        pdf_path = _plot(rows, out_dir / name, columns)
        if pdf_path is not None:
            print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
