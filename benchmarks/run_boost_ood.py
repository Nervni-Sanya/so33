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
                   default=("so33,so33_multi,so33_signature_only,"
                            "relu_bottleneck,relu_mlp,"
                            "eta_invariants,so33_equivariant,"
                            "so33_equivariant_frozen,"
                            "so33_equivariant_unbounded"),
                   help="Comma-separated model names. eta_invariants and "
                        "so33_equivariant* are set-based; "
                        "so33_equivariant_frozen and *_unbounded are the "
                        "week-2 ablations for the equivariant-OOD failure.")
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

    # The two equivariant architectures consume per-particle sets (B, K, 5)
    # = (E, px, py, pz, mask). Repack the flat (B, 6) lifted-6-vector samples
    # as K=1 sets (one particle per "jet"): extract (E, px, py, pz) from the
    # (px, py, pz, E, 0, 0) lift, add a mask=1 channel.
    def to_set(X6: torch.Tensor) -> torch.Tensor:
        p4 = torch.stack([X6[:, 3], X6[:, 0], X6[:, 1], X6[:, 2]], dim=-1)  # E,px,py,pz
        mask = torch.ones(p4.shape[0], 1, dtype=p4.dtype)
        return torch.cat([p4.unsqueeze(1), mask.unsqueeze(-1)], dim=-1)     # (B,1,5)

    from benchmarks.models import SET_MODELS as _SET_MODELS
    set_names = set(_SET_MODELS)

    for name in names:
        is_set = name in set_names
        Xtr_m, Xva_m, Xid_m, Xood_m = (
            (to_set(Xtr), to_set(Xva), to_set(Xid), to_set(Xood))
            if is_set else (Xtr, Xva, Xid, Xood)
        )
        in_features = 4 if is_set else 6
        representation = "constituents" if is_set else "flat"
        bound = True if name.startswith("so33") else None
        model = build_model(name, in_features=in_features, out_features=2,
                            T=0.3, bound_input=bound,
                            representation=representation)
        cfg = TrainConfig(epochs=args.epochs, batch_size=128, lr=3e-3,
                          seed=args.seed, cosine_schedule=True, grad_clip=1.0)
        print(f"[boost_ood] {name} | train ...")
        res = train_classifier(model, Xtr_m, ytr, Xva_m, yva, cfg)

        id_auc, id_acc = _auc(model, Xid_m, yid), _acc(model, Xid_m, yid)
        ood_auc, ood_acc = _auc(model, Xood_m, yood), _acc(model, Xood_m, yood)
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
