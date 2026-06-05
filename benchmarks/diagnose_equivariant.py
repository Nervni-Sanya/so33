"""
benchmarks.diagnose_equivariant
-------------------------------
Week-2 diagnostic: measure why `so33_equivariant` collapses to OOD AUC
~0.663 while `eta_invariants` (same lift, no SO33Activation) reaches 1.000.

Working hypothesis (after code inspection):
    The `bound_input=True` flag in `SO33Activation` normalises inputs by
    1 + ||x||_2, where ||.||_2 is the Euclidean norm. The Euclidean norm
    is NOT an SO(3,3) invariant: a boost of rapidity phi scales it
    roughly by cosh(phi). At training rapidity ~0.6 the rescaling is
    modest (~1.2x); at OOD rapidity ~2.5 it is ~6x. The activation's
    apparent equivariance error therefore grows with input rapidity.

This script trains `so33_equivariant` on the standard boost_ood setup,
then measures the empirical activation-level equivariance error on
freshly sampled inputs at the ID rapidity range and the OOD rapidity
range, separately. Results are written to
results/diagnose_equivariant__seed{N}.json.

Run:
    python -m benchmarks.diagnose_equivariant --seed 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from benchmarks.synthetic import (
    generate_boost_invariant_dataset,
    random_so33_element,
)
from benchmarks.models import build_model, _lift_4to6
from benchmarks.train import train_classifier, TrainConfig
from so33.basis import DIM


def _to_set(X6: torch.Tensor) -> torch.Tensor:
    """Convert flat (B, 6) lifted samples back to (B, K=1, 5) sets."""
    p4 = torch.stack([X6[:, 3], X6[:, 0], X6[:, 1], X6[:, 2]], dim=-1)
    mask = torch.ones(p4.shape[0], 1, dtype=p4.dtype)
    return torch.cat([p4.unsqueeze(1), mask.unsqueeze(-1)], dim=-1)


def _measure_eq_error(
    activations: list,
    p_set: torch.Tensor,
    rapidity_for_g: float,
    n_groups: int,
    generator: torch.Generator,
) -> dict[str, float]:
    """Empirical relative equivariance error of each activation.

    For each sampled g in SO(3,3), compute
        err = || act(g x) - g act(x) ||_2  /  || act(x) ||_2
    averaged over the batch. Aggregated mean and max over n_groups draws,
    averaged over the activations.

    Parameters
    ----------
    activations  : list of SO33Activation instances (the model's `.acts`).
    p_set        : (B, K=1, 5) input batch in the per-particle layout.
    rapidity_for_g : scale used by random_so33_element when drawing the
                     test group elements. Pass small (e.g. 0.5) for "near
                     identity" measurements; the input rapidity is what
                     varies across calls, not this.
    n_groups     : number of g draws to average over.
    generator    : torch.Generator for reproducible group sampling.
    """
    dtype = activations[0].dtype
    p4 = p_set[..., :4].to(dtype)
    p6 = _lift_4to6(p4)                            # (B, 1, 6)
    B = p6.shape[0]
    p_flat = p6.reshape(B, DIM)                    # (B, 6)

    errs_per_g: list[float] = []
    maxes_per_g: list[float] = []
    for _ in range(n_groups):
        g = random_so33_element(rapidity=rapidity_for_g,
                                generator=generator, dtype=dtype)
        gp = p_flat @ g.T                          # (B, 6)
        per_act = []
        per_act_max = []
        for act in activations:
            act.eval()
            with torch.no_grad():
                y    = act(p_flat)
                y_g  = act(gp)
                g_y  = y @ g.T
                num  = (y_g - g_y).norm(dim=-1)
                den  = y.norm(dim=-1).clamp_min(1e-12)
                rel  = (num / den)
                per_act.append(rel.mean().item())
                per_act_max.append(rel.max().item())
        errs_per_g.append(sum(per_act) / len(per_act))
        maxes_per_g.append(sum(per_act_max) / len(per_act_max))

    return {
        "mean_rel_err":  sum(errs_per_g)  / len(errs_per_g),
        "max_rel_err":   sum(maxes_per_g) / len(maxes_per_g),
        "rapidity_test_groups": rapidity_for_g,
        "n_group_draws": n_groups,
        "n_inputs":      B,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train-rapidity", type=float, default=0.6)
    p.add_argument("--ood-rapidity",   type=float, default=2.5)
    p.add_argument("--n-eq-samples", type=int, default=512,
                   help="Input batch size for the equivariance probe.")
    p.add_argument("--n-group-draws", type=int, default=16,
                   help="Random g draws to average each err over.")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args(argv)

    # 1. Build & train so33_equivariant on the exact run_boost_ood setup.
    half = args.n // 2
    Xtr, ytr, _ = generate_boost_invariant_dataset(
        n_samples=half, seed=args.seed,
        rapidity_range=(0.0, args.train_rapidity))
    n_val = int(0.15 * len(Xtr))
    Xva, yva = Xtr[:n_val], ytr[:n_val]
    Xtr, ytr = Xtr[n_val:], ytr[n_val:]

    model = build_model("so33_equivariant", in_features=4, out_features=2,
                        T=0.3, bound_input=True,
                        representation="constituents")
    cfg = TrainConfig(epochs=args.epochs, batch_size=128, lr=3e-3,
                      seed=args.seed, cosine_schedule=True, grad_clip=1.0)
    print(f"[diagnose] training so33_equivariant on boost_ood "
          f"(seed={args.seed}, epochs={args.epochs})")
    res = train_classifier(model, _to_set(Xtr), ytr, _to_set(Xva), yva, cfg)
    print(f"[diagnose] trained in {res.walltime_sec:.1f}s, "
          f"val_acc={res.final_val_acc:.3f}")

    # 2. Generate ID-rapidity and OOD-rapidity test inputs.
    Xid, _, _ = generate_boost_invariant_dataset(
        n_samples=args.n_eq_samples, seed=args.seed + 1000,
        rapidity_range=(0.0, args.train_rapidity))
    Xood, _, _ = generate_boost_invariant_dataset(
        n_samples=args.n_eq_samples, seed=args.seed + 2000,
        rapidity_range=(args.train_rapidity, args.ood_rapidity))

    # 3. Probe equivariance of each SO33Activation on each split.
    gen = torch.Generator().manual_seed(args.seed + 31337)
    activations = list(model.acts)

    print(f"[diagnose] measuring activation equivariance on "
          f"{args.n_eq_samples} inputs, {args.n_group_draws} group draws each")
    id_metrics = _measure_eq_error(
        activations, _to_set(Xid),
        rapidity_for_g=0.5, n_groups=args.n_group_draws, generator=gen,
    )
    ood_metrics = _measure_eq_error(
        activations, _to_set(Xood),
        rapidity_for_g=0.5, n_groups=args.n_group_draws, generator=gen,
    )

    # 4. Diagnostic stats on the trained connection.
    coeff_norms = [float(a.coeffs.detach().norm().item()) for a in activations]
    eucl_id  = float(Xid.norm(dim=-1).mean().item())
    eucl_ood = float(Xood.norm(dim=-1).mean().item())

    record = {
        "experiment":   "diagnose_equivariant",
        "seed":         args.seed,
        "train_val_acc": res.final_val_acc,
        "trained_coeff_norms": coeff_norms,
        "mean_input_norm_id":  eucl_id,
        "mean_input_norm_ood": eucl_ood,
        "id_rapidity_range":   [0.0, args.train_rapidity],
        "ood_rapidity_range":  [args.train_rapidity, args.ood_rapidity],
        "id_equivariance":  id_metrics,
        "ood_equivariance": ood_metrics,
        "ratio_ood_over_id_meanerr": (
            ood_metrics["mean_rel_err"] / max(id_metrics["mean_rel_err"], 1e-12)
        ),
    }

    print()
    print("[diagnose] post-training activation equivariance")
    print(f"  mean ||coeffs||                : "
          f"{sum(coeff_norms)/len(coeff_norms):.4f}")
    print(f"  mean Euclidean ||x||  ID  : {eucl_id:.3f}")
    print(f"  mean Euclidean ||x||  OOD : {eucl_ood:.3f}  "
          f"(ratio {eucl_ood/eucl_id:.2f}x)")
    print(f"  rel-eq err on ID inputs  : "
          f"mean={id_metrics['mean_rel_err']:.4f}  "
          f"max={id_metrics['max_rel_err']:.4f}")
    print(f"  rel-eq err on OOD inputs : "
          f"mean={ood_metrics['mean_rel_err']:.4f}  "
          f"max={ood_metrics['max_rel_err']:.4f}")
    print(f"  OOD/ID ratio of mean err : "
          f"{record['ratio_ood_over_id_meanerr']:.2f}x")
    print()
    print("[diagnose] If the OOD/ID ratio is >> 1, the bound_input "
          "hypothesis is supported: the Euclidean-norm normalisation "
          "rescales boosted inputs more than ID inputs, so the activation "
          "becomes input-rapidity-dependent.")

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"diagnose_equivariant__seed{args.seed}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"[diagnose] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
