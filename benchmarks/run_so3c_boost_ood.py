"""
benchmarks.run_so3c_boost_ood
-----------------------------
Headline experiment for the complexified prior: OOD generalisation across
SO(3, C) boosts on the "electromagnetic invariant" task.

The label is carried by an SO(3, C) invariant of z = E + i B:

    mode "im" (default): Im(z.z) = 2 E.B   — invisible to eta-only readouts,
                                             the parent SO(3,3) prior sees
                                             only Re(z.z);
    mode "re"          : Re(z.z)           — control, visible to both priors.

Every sample is scrambled by a random SO(3, C) element; training uses a LOW
boost-norm regime, OOD evaluation a HIGH one the models never saw. Expected
pattern (mode "im"):

    so3c_invariants / so3c_flow : ~1.0 AUC in-distribution AND OOD
    relu_mlp                    : learns 2 x.y approximately in-distribution,
                                  degrades under unseen boosts
    eta_only                    : ~0.5 AUC everywhere (structurally blind) —
                                  the gap so3c closes over the SO(3,3) prior

Run:
    python -m benchmarks.run_so3c_boost_ood --quick
    python -m benchmarks.run_so3c_boost_ood --n 20000 --epochs 40 \
        --train-boost 0.6 --ood-boost 2.5 --invariant-mode im
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from benchmarks.so3c_synthetic import generate_em_invariant_dataset
from benchmarks.so3c_models import (
    EtaOnlyClassifier,
    SO3CFlowClassifier,
    SO3CInvariantsClassifier,
)
from benchmarks.models import build_model
from benchmarks.train import train_classifier, TrainConfig


def _auc(model, X, y) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        if not torch.isfinite(logits).all():
            return float("nan")
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y.cpu().numpy(), scores))
    except Exception:
        return float("nan")


def _acc(model, X, y) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        if not torch.isfinite(logits).all():
            return float("nan")
        return (logits.argmax(dim=-1) == y).float().mean().item()


def _build(name: str):
    if name == "so3c_invariants":
        return SO3CInvariantsClassifier(out_features=2)
    if name == "so3c_flow":
        return SO3CFlowClassifier(out_features=2)
    if name == "eta_only":
        return EtaOnlyClassifier(out_features=2)
    # Fall back to the shared factory for baselines (flat (B, 6) models).
    bound = True if name.startswith("so33") else None
    return build_model(name, in_features=6, out_features=2,
                       T=0.3, bound_input=bound, representation="flat")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Smoke run: 1k samples, 5 epochs.")
    p.add_argument("--models", type=str,
                   default="so3c_invariants,so3c_flow,eta_only,relu_mlp,so33")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--invariant-mode", type=str, default="im",
                   choices=("im", "re"),
                   help="'im': label on Im(z.z)=2E.B (eta-blind); "
                        "'re': label on Re(z.z) (control).")
    p.add_argument("--train-boost", type=float, default=0.6,
                   help="Train/ID ||beta|| drawn from [0, this].")
    p.add_argument("--ood-boost", type=float, default=2.5,
                   help="OOD ||beta|| drawn from [train-boost, this].")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args(argv)

    if args.quick:
        args.n, args.epochs = 1_000, 5

    half = args.n // 2
    common = dict(invariant_mode=args.invariant_mode)
    Xtr, ytr, _ = generate_em_invariant_dataset(
        n_samples=half, seed=args.seed,
        boost_range=(0.0, args.train_boost), **common)
    Xid, yid, _ = generate_em_invariant_dataset(
        n_samples=half // 2, seed=args.seed + 100,
        boost_range=(0.0, args.train_boost), **common)
    Xood, yood, _ = generate_em_invariant_dataset(
        n_samples=half // 2, seed=args.seed + 200,
        boost_range=(args.train_boost, args.ood_boost), **common)

    n_val = int(0.15 * len(Xtr))
    Xva, yva = Xtr[:n_val], ytr[:n_val]
    Xtr, ytr = Xtr[n_val:], ytr[n_val:]

    print(f"[so3c_boost_ood] mode={args.invariant_mode} | train={len(Xtr)} "
          f"(||beta||<= {args.train_boost}) | id_test={len(Xid)} | "
          f"ood_test={len(Xood)} (||beta|| {args.train_boost}-{args.ood_boost})")

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    summary = []

    for name in names:
        model = _build(name)
        cfg = TrainConfig(epochs=args.epochs, batch_size=128, lr=3e-3,
                          seed=args.seed, cosine_schedule=True, grad_clip=1.0)
        print(f"[so3c_boost_ood] {name} | train ...")
        res = train_classifier(model, Xtr, ytr, Xva, yva, cfg)

        id_auc, id_acc = _auc(model, Xid, yid), _acc(model, Xid, yid)
        ood_auc, ood_acc = _auc(model, Xood, yood), _acc(model, Xood, yood)
        record = {
            "experiment": "so3c_boost_ood",
            "invariant_mode": args.invariant_mode,
            "model": name,
            "seed": args.seed,
            "n_params": res.n_params,
            "n_train": len(Xtr),
            "train_boost": args.train_boost,
            "ood_boost": args.ood_boost,
            "id_auc": id_auc, "id_acc": id_acc,
            "ood_auc": ood_auc, "ood_acc": ood_acc,
            "auc_gap": (id_auc - ood_auc) if id_auc == id_auc else float("nan"),
            "walltime_sec": res.walltime_sec,
        }
        (results_dir /
         f"so3c_boost_ood__{args.invariant_mode}__{name}__seed{args.seed}.json"
         ).write_text(json.dumps(record, indent=2))
        summary.append(record)
        print(f"   id_auc={id_auc:.3f} ood_auc={ood_auc:.3f} "
              f"gap={record['auc_gap']:+.3f}  ({res.walltime_sec:.1f}s)")

    print(f"\n[so3c_boost_ood] mode={args.invariant_mode} — ID vs OOD AUC")
    print(f"{'model':<20} {'params':>7} {'id_auc':>7} {'ood_auc':>8} {'gap':>7}")
    print("-" * 54)
    for r in sorted(summary, key=lambda r: (-(r['ood_auc'] if r['ood_auc'] == r['ood_auc'] else -1))):
        ood = f"{r['ood_auc']:.3f}" if r['ood_auc'] == r['ood_auc'] else "  nan"
        idv = f"{r['id_auc']:.3f}" if r['id_auc'] == r['id_auc'] else "  nan"
        gap = f"{r['auc_gap']:+.3f}" if r['auc_gap'] == r['auc_gap'] else "  nan"
        print(f"{r['model']:<20} {r['n_params']:>7} {idv:>7} {ood:>8} {gap:>7}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
