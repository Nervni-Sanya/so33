"""
benchmarks.run_init_sensitivity
-------------------------------
Does the result depend on how the geodesic connection is initialised?

The flow's connection comes from HermitianMetric, whose output layer is
zero-initialised (so3c/metric.py) so that a(s) = 0 and the flow starts as
the identity map. That is a deliberately neutral start, not a tuned one --
but a reviewer is entitled to ask whether the reported numbers survive a
different initialisation, or whether the zero start is load-bearing.

This script retrains so3c_equivariant_set on the internal top-tagging
protocol under several initialisation schemes and reports the spread of
the final test AUC. Only the metric MLP's output layer is touched; the
rest of the model is built exactly as in the headline runs.

Schemes:
    zeros       -- the default: identity flow at step 0
    normal_0.01 -- small random connection
    normal_0.1  -- moderate
    normal_0.5  -- large enough that the initial flow is a real rotation
    orthogonal  -- orthogonal init, a common non-random-scale alternative

Run:
    python -m benchmarks.run_init_sensitivity
    python -m benchmarks.run_init_sensitivity --max-samples 20000 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import torch.nn as nn

from benchmarks.datasets import load_top_tagging_constituents
from benchmarks.models import build_model
from benchmarks.tabular_runner import evaluate_test
from benchmarks.train import train_classifier, TrainConfig


SCHEMES = ("zeros", "normal_0.01", "normal_0.1", "normal_0.5", "orthogonal")


def apply_init(model: nn.Module, scheme: str, seed: int) -> None:
    """Re-initialise the connection MLP's output layer in place."""
    layer = model.act.metric.net[-1]
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if scheme == "zeros":
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif scheme.startswith("normal_"):
            std = float(scheme.split("_")[1])
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen,
                                           dtype=layer.weight.dtype) * std)
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=gen,
                                         dtype=layer.bias.dtype) * std)
        elif scheme == "orthogonal":
            w = torch.empty(layer.weight.shape, dtype=torch.float32)
            nn.init.orthogonal_(w)
            layer.weight.copy_(w.to(layer.weight.dtype))
            nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"unknown scheme: {scheme!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=str, default="data")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--n-constituents", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="so3c_equivariant_set")
    p.add_argument("--schemes", type=str, default=",".join(SCHEMES))
    p.add_argument("--results-dir", type=str, default="results_init")
    args = p.parse_args(argv)

    split = load_top_tagging_constituents(
        cache_dir=args.cache_dir, max_samples=args.max_samples,
        n_constituents=args.n_constituents, seed=args.seed,
        normalize="global",
    )
    print(f"[init] {split.summary()}")

    out_dir = pathlib.Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        model = build_model(args.model, in_features=split.n_features,
                            out_features=split.n_classes,
                            representation="constituents")
        apply_init(model, scheme, seed=args.seed)
        init_norm = model.act.metric.net[-1].weight.abs().max().item()

        cfg = TrainConfig(epochs=args.epochs, batch_size=128, lr=3e-3,
                          seed=args.seed, cosine_schedule=True, grad_clip=1.0)
        print(f"[init] {scheme:<12} (max |W_out| = {init_norm:.3f}) | train ...")
        res = train_classifier(model, split.X_train, split.y_train,
                               split.X_val, split.y_val, cfg)
        test = evaluate_test(model, split.X_test, split.y_test, split.n_classes)
        # evaluate_test stashes the raw per-jet scores under "_scores" for
        # callers that persist them as npz; they are not JSON-serialisable.
        test = {k: v for k, v in test.items() if not k.startswith("_")}

        record = {
            "experiment": "init_sensitivity",
            "model": args.model,
            "init_scheme": scheme,
            "init_max_abs_weight": init_norm,
            "seed": args.seed,
            "n_params": res.n_params,
            "n_train": len(split.X_train),
            "walltime_sec": res.walltime_sec,
            "train_metrics": {"final_val_acc": res.final_val_acc},
            "test_metrics": test,
        }
        (out_dir / f"init_sensitivity__{args.model}__{scheme}__seed{args.seed}.json"
         ).write_text(json.dumps(record, indent=2))
        rows.append(record)
        print(f"   test_acc={test['test_acc']:.4f} auc={test.get('test_auc', float('nan')):.4f} "
              f"rej@0.3={test.get('bg_rej_30', float('nan')):.0f}  ({res.walltime_sec:.0f}s)")

    aucs = [r["test_metrics"].get("test_auc") for r in rows]
    aucs = [a for a in aucs if a is not None]
    print(f"\n[init] {args.model}: {len(rows)} schemes")
    print(f"{'scheme':<14}{'AUC':>9}{'rej@0.3':>10}")
    print("-" * 33)
    for r in rows:
        print(f"{r['init_scheme']:<14}{r['test_metrics'].get('test_auc', float('nan')):>9.4f}"
              f"{r['test_metrics'].get('bg_rej_30', float('nan')):>10.0f}")
    if aucs:
        print(f"\nspread: max-min = {max(aucs) - min(aucs):.4f} "
              f"(mean {sum(aucs)/len(aucs):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
