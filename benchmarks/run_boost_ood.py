"""
benchmarks.run_boost_ood
------------------------
Headline experiment: out-of-distribution generalisation across Lorentz
boosts.

The label is a boost-invariant (the particle's invariant mass band), but
every sample is scrambled by a random SO(3,3) boost. We train on a LOW
rapidity regime and test on a HIGH rapidity regime the model never saw.

A Lorentz-equivariant prior should generalise across rapidity for free,
because the discriminating quantity (the invariant) is unchanged by the
boost. A generic MLP has to memorise the boosted manifold it was trained
on and degrades out of distribution. This is the setting where SO33's
inductive bias is structurally advantaged — and, unlike raw in-distribution
accuracy, the advantage cannot be closed simply by giving the MLP more width.

Reports, per model: in-distribution test AUC/acc (rapidity like train),
out-of-distribution AUC/acc (higher rapidity), and the OOD gap.

Run:
    python -m benchmarks.run_boost_ood --quick
    python -m benchmarks.run_boost_ood --n 20000 --epochs 40 \
        --train-rapidity 0.6 --ood-rapidity 1.5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from benchmarks.synthetic import generate_boost_invariant_dataset
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Smoke run: 1k samples, 5 epochs.")
    p.add_argument("--models", type=str,
                   default="so33,so33_multi,so33_signature_only,relu_bottleneck,relu_mlp",
                   help="Comma-separated model names.")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-rapidity", type=float, default=0.6,
                   help="Train/ID rapidity drawn from [0, this].")
    p.add_argument("--ood-rapidity", type=float, default=2.5,
                   help="OOD rapidity drawn from [train-rapidity, this]. A wide "
                        "gap from --train-rapidity is where a generic MLP, which "
                        "memorises the training boost manifold, degrades most.")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args(argv)

    if args.quick:
        args.n, args.epochs = 1_000, 5

    half = args.n // 2
    # In-distribution: train + id-test share the low-rapidity regime.
    Xtr, ytr, _ = generate_boost_invariant_dataset(
        n_samples=half, seed=args.seed,
        rapidity_range=(0.0, args.train_rapidity))
    Xid, yid, _ = generate_boost_invariant_dataset(
        n_samples=half // 2, seed=args.seed + 100,
        rapidity_range=(0.0, args.train_rapidity))
    # Out-of-distribution: higher rapidity, unseen in training.
    Xood, yood, _ = generate_boost_invariant_dataset(
        n_samples=half // 2, seed=args.seed + 200,
        rapidity_range=(args.train_rapidity, args.ood_rapidity))

    n_val = int(0.15 * len(Xtr))
    Xva, yva = Xtr[:n_val], ytr[:n_val]
    Xtr, ytr = Xtr[n_val:], ytr[n_val:]

    print(f"[boost_ood] train={len(Xtr)} (rapidity<= {args.train_rapidity}) | "
          f"id_test={len(Xid)} | ood_test={len(Xood)} "
          f"(rapidity {args.train_rapidity}-{args.ood_rapidity})")

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    summary = []

    for name in names:
        # SO33 sees large-magnitude boosted inputs -> needs bound_input on.
        bound = True if name.startswith("so33") else None
        model = build_model(name, in_features=6, out_features=2, T=0.3,
                            bound_input=bound)
        cfg = TrainConfig(epochs=args.epochs, batch_size=128, lr=3e-3,
                          seed=args.seed, cosine_schedule=True, grad_clip=1.0)
        print(f"[boost_ood] {name} | train ...")
        res = train_classifier(model, Xtr, ytr, Xva, yva, cfg)

        id_auc, id_acc = _auc(model, Xid, yid), _acc(model, Xid, yid)
        ood_auc, ood_acc = _auc(model, Xood, yood), _acc(model, Xood, yood)
        record = {
            "experiment": "boost_ood",
            "model": name,
            "seed": args.seed,
            "n_params": res.n_params,
            "n_train": len(Xtr),
            "train_rapidity": args.train_rapidity,
            "ood_rapidity": args.ood_rapidity,
            "id_auc": id_auc, "id_acc": id_acc,
            "ood_auc": ood_auc, "ood_acc": ood_acc,
            "auc_gap": (id_auc - ood_auc) if id_auc == id_auc else float("nan"),
            "walltime_sec": res.walltime_sec,
        }
        (results_dir / f"boost_ood__{name}__seed{args.seed}.json").write_text(
            json.dumps(record, indent=2))
        summary.append(record)
        print(f"   id_auc={id_auc:.3f} ood_auc={ood_auc:.3f} "
              f"gap={record['auc_gap']:+.3f}  ({res.walltime_sec:.1f}s)")

    print("\n[boost_ood] ID vs OOD AUC (smaller gap = better generalisation)")
    print(f"{'model':<22} {'params':>7} {'id_auc':>7} {'ood_auc':>8} {'gap':>7}")
    print("-" * 56)
    for r in sorted(summary, key=lambda r: (-(r['ood_auc'] if r['ood_auc']==r['ood_auc'] else -1))):
        ood = f"{r['ood_auc']:.3f}" if r['ood_auc'] == r['ood_auc'] else "  nan"
        idv = f"{r['id_auc']:.3f}" if r['id_auc'] == r['id_auc'] else "  nan"
        gap = f"{r['auc_gap']:+.3f}" if r['auc_gap'] == r['auc_gap'] else "  nan"
        print(f"{r['model']:<22} {r['n_params']:>7} {idv:>7} {ood:>8} {gap:>7}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
