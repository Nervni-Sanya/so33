"""
benchmarks.measure_throughput
-----------------------------
Training and inference cost of every model in the top-tagging comparison,
measured on one machine.

What this can and cannot say
----------------------------
It measures OUR models on the machine that produced every number in this
branch, so the relative costs are directly comparable: the geodesic flow
against its own no-flow ablation, against the eta baseline, and against a
parameter-matched generic set model.

It does NOT compare against LGN / LorentzNet / PELICAN / ParticleNet. Those
are GPU-trained models of 4.5k-500k parameters; re-running them on this CPU
would take weeks, and quoting their published wall-clock next to ours would
compare different hardware, frameworks and batch sizes. Published timings
belong in the paper only with the hardware stated alongside.

Reported per model: trainable parameters, seconds per training epoch
(measured on a fixed number of batches and scaled), and inference
throughput in jets/second at several batch sizes.

Run:
    python -m benchmarks.measure_throughput
    python -m benchmarks.measure_throughput --train-batches 20
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

from benchmarks.datasets import load_top_tagging_constituents
from benchmarks.models import build_model


MODELS = ("so3c_equivariant_set", "so3c_invariant_set",
          "eta_invariants", "relu_mlp")


def _time_training(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                   batch_size: int, n_batches: int) -> float:
    """Median seconds per training step (forward + backward + step)."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    times = []
    for i in range(n_batches + 2):          # first two are warm-up
        xb = X[i * batch_size:(i + 1) * batch_size]
        yb = y[i * batch_size:(i + 1) * batch_size]
        if len(xb) < batch_size:
            break
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = crit(model(xb), yb) + model.regularization_loss()
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    times = sorted(times[2:])
    return times[len(times) // 2] if times else float("nan")


@torch.no_grad()
def _time_inference(model: nn.Module, X: torch.Tensor,
                    batch_size: int, n_batches: int) -> float:
    """Jets per second at a given batch size (median over batches)."""
    model.eval()
    times = []
    for i in range(n_batches + 2):
        xb = X[i * batch_size:(i + 1) * batch_size]
        if len(xb) < batch_size:
            break
        t0 = time.perf_counter()
        model(xb)
        times.append(time.perf_counter() - t0)
    times = sorted(times[2:])
    if not times:
        return float("nan")
    return batch_size / times[len(times) // 2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=str, default="data")
    p.add_argument("--n-constituents", type=int, default=32)
    p.add_argument("--models", type=str, default=",".join(MODELS))
    p.add_argument("--natural-hidden", type=int, default=1293,
                   help="Width of the generic baseline; the default matches "
                        "so3c_equivariant_set's parameter count.")
    p.add_argument("--train-batch", type=int, default=128)
    p.add_argument("--train-batches", type=int, default=15)
    p.add_argument("--infer-batches", type=int, default=8)
    p.add_argument("--infer-sizes", type=str, default="128,512,2048")
    p.add_argument("--n-train-epoch", type=int, default=1_211_000,
                   help="Jets per epoch used to scale the per-step time into "
                        "a per-epoch estimate (default: canonical train size).")
    p.add_argument("--out-dir", type=str, default="paper/figures")
    args = p.parse_args(argv)

    # A small slice is enough: we time per-batch cost, not convergence.
    split = load_top_tagging_constituents(
        cache_dir=args.cache_dir, max_samples=20_000,
        n_constituents=args.n_constituents, seed=0, normalize="global",
    )
    X, y = split.X_train, split.y_train
    infer_sizes = [int(s) for s in args.infer_sizes.split(",") if s.strip()]
    print(f"[throughput] device=cpu torch={torch.__version__} "
          f"threads={torch.get_num_threads()} | K={args.n_constituents}")

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model = build_model(name, in_features=split.n_features,
                            out_features=split.n_classes,
                            representation="constituents",
                            natural_hidden=args.natural_hidden)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        step_s = _time_training(model, X, y, args.train_batch, args.train_batches)
        epoch_s = step_s * (args.n_train_epoch / args.train_batch)
        row = {"model": name, "n_params": n_params,
               "train_step_ms": step_s * 1e3, "epoch_estimate_h": epoch_s / 3600.0}
        for bs in infer_sizes:
            row[f"infer_jets_per_s_bs{bs}"] = _time_inference(
                model, X, bs, args.infer_batches)
        rows.append(row)
        print(f"  {name:<22} {n_params:>6} par | step {step_s*1e3:7.1f} ms | "
              f"epoch~{epoch_s/3600:5.2f} h | " +
              " ".join(f"bs{bs}:{row[f'infer_jets_per_s_bs{bs}']:.0f}/s"
                       for bs in infer_sizes))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "throughput.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[throughput] wrote {csv_path}")

    meta = {"torch": torch.__version__, "threads": torch.get_num_threads(),
            "device": "cpu", "n_constituents": args.n_constituents,
            "train_batch": args.train_batch,
            "epoch_scaled_to_n_jets": args.n_train_epoch}
    (out_dir / "throughput_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
