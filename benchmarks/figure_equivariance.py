"""
benchmarks.figure_equivariance
------------------------------
Measure the empirical equivariance of SO33Activation as a function of
the connection-coefficient norm, for each of the three input-bound
modes. Writes a CSV (paper/figures/equivariance_data.csv) and a PDF
plot (paper/figures/equivariance_vs_norm.pdf).

Two quantities are measured at each (||c||, bound) cell:
  1. Activation-level rel err
       E_b [ || sigma(g x) - g sigma(x) ||_2 / || sigma(x) ||_2 ]
  2. Readout-feature invariance error
       E_b [ | <sigma(x),sigma(x)>_eta - <sigma(gx),sigma(gx)>_eta | ]

The latter is the more directly relevant quantity for the classifier
(both architectures read out eta-invariants of the activation output).

Run:
    python -m benchmarks.figure_equivariance
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import torch

from so33 import SO33Activation
from so33.basis import ETA, N_BASIS
from benchmarks.synthetic import random_so33_element


def make_act(mode: str | bool) -> SO33Activation:
    return SO33Activation(
        T=0.3, method="rk4", solver_options={"step_size": 0.03},
        adjoint=False, bound_input=mode,
    )


def measure(act: SO33Activation, x: torch.Tensor, gs: torch.Tensor,
            eta: torch.Tensor) -> tuple[float, float]:
    """Return (activation rel err, readout-feature abs err) averaged over gs."""
    act.eval()
    with torch.no_grad():
        y = act(x)
        m2_y = (y * eta * y).sum(-1)
        act_errs, feat_errs = [], []
        for g in gs:
            gx = x @ g.T
            ygx = act(gx)
            g_y = y @ g.T
            num = (ygx - g_y).norm(dim=-1)
            den = y.norm(dim=-1).clamp_min(1e-12)
            act_errs.append((num / den).mean().item())
            m2_gy = (ygx * eta * ygx).sum(-1)
            feat_errs.append((m2_y - m2_gy).abs().mean().item())
    return sum(act_errs)/len(act_errs), sum(feat_errs)/len(feat_errs)


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

    # PDF figure: feat_abs_err vs ||c|| per bound mode (log y).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping PDF.", file=sys.stderr)
        return 0

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    by_bound = {b: [] for _, b in [(0,"none"),(0,"euclidean"),(0,"eta")]}
    by_bound = {"none": [], "euclidean": [], "eta": []}
    for cnorm, label, _, feat in rows:
        by_bound[label].append((cnorm, feat))
    markers = {"none": "o", "euclidean": "s", "eta": "^"}
    for label, data in by_bound.items():
        xs = [c for c, _ in data]
        ys = [max(f, 1e-16) for _, f in data]
        ax.plot(xs, ys, marker=markers[label], label=f"bound = {label}")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\|c\|_2$  (connection coefficient norm)")
    ax.set_ylabel(r"$|m^2(\sigma x) - m^2(\sigma gx)|$  (mean over batch)")
    ax.set_title("Readout-feature equivariance vs connection norm")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    pdf_path = out_dir / "equivariance_vs_norm.pdf"
    fig.savefig(pdf_path)
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
