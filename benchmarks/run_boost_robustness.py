"""
benchmarks.run_boost_robustness
-------------------------------
Does a trained top tagger still work when the jet is boosted?

The canonical benchmark cannot answer this. Its test set carries the same
boost distribution as its training set, so a model that has merely
memorised the lab frame scores exactly like one that is genuinely Lorentz
invariant. That is how `so3c_equivariant_set` reached 0.9743 on canonical
data while not being invariant at all: its connection a(s) is a function of
invariants, which do not move under z -> Qz, so the flow applies the *same*
rotation in the new frame instead of the conjugated one.

This script trains each model once and then re-evaluates the SAME model on
the SAME test jets pushed through random Lorentz transformations of
increasing rapidity. A model whose invariance is exact traces a flat line;
one that only looks invariant falls off it.

Boosting after the global normalisation is legitimate: `normalize="global"`
divides every component by one scalar, which commutes with the Lorentz
action, so a boosted normalised jet is the normalised version of a boosted
jet.

Each rapidity averages over several random group elements, since a single
draw picks one direction and the answer depends on where it points
relative to the jet axis.

Precision. This diagnostic defaults to float64 and should stay there. The
bivector lift is quadratic in the momenta, so a rapidity-2 boost inflates
the input range by ~e^2 = 7.4 and the lifted range by ~55; in float32 that
costs enough significant digits to fake a real effect. Measured on the same
exactly-equivariant model, same weights, same jets: float64 drops 0.0001
across rapidity 0 -> 2, float32 drops 0.0232. The canonical benchmark runs
in float32 without any such problem because its test jets sit at rapidity
0, where invariance holds to ~2e-7.

Run:
    python -m benchmarks.run_boost_robustness --cache-dir data \\
        --models so3c_covariant_set,so3c_message_set --n 20000 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from benchmarks.datasets import load_top_tagging_constituents
from benchmarks.models import build_model
from benchmarks.train import TrainConfig, forward_in_chunks, train_classifier
from so3c.lift import random_lorentz_pair


def _auc(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
         chunk: int) -> float:
    logits = forward_in_chunks(model, X, chunk)
    if not torch.isfinite(logits).all():
        return float("nan")
    scores = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y.cpu().numpy(), scores))


def boost_jets(X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 Lorentz matrix to the 4-momenta, leaving the mask alone."""
    p4 = X[..., :4].to(L.dtype) @ L.T
    return torch.cat([p4.to(X.dtype), X[..., 4:]], dim=-1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=str, default="data")
    p.add_argument("--models", type=str,
                   default="so3c_invariant_set,so3c_equivariant_set,"
                           "so3c_covariant_set,so3c_message_set")
    p.add_argument("--n", type=int, default=20_000,
                   help="Jets loaded (internal 70/15/15 re-split). The point "
                        "here is the SHAPE of the curve, not the level, so a "
                        "cheap training run is enough.")
    p.add_argument("--n-constituents", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-chunk-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                   help="float64 by default: see the note on precision above.")
    p.add_argument("--rapidities", type=str, default="0,0.5,1.0,1.5,2.0,3.0")
    p.add_argument("--draws", type=int, default=4,
                   help="Random group elements averaged per rapidity.")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--neighbors", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--results-dir", type=str, default="results_boost")
    args = p.parse_args(argv)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    rapidities = [float(r) for r in args.rapidities.split(",")]
    so3c_kwargs = {k: v for k, v in (("rounds", args.rounds),
                                     ("neighbors", args.neighbors),
                                     ("channels", args.channels))
                   if v is not None} or None

    split = load_top_tagging_constituents(
        cache_dir=args.cache_dir, max_samples=args.n,
        n_constituents=args.n_constituents, seed=args.seed,
        standardise=True, normalize="global",
    )
    print(f"[boost_robustness] {split.summary()}")

    out_dir = pathlib.Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name in [m.strip() for m in args.models.split(",")]:
        torch.manual_seed(args.seed)
        model = build_model(
            name, in_features=4, out_features=split.n_classes,
            representation="constituents", dtype=dtype,
            so3c_kwargs=so3c_kwargs,
        )
        n_params = sum(q.numel() for q in model.parameters())
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                          seed=args.seed, device=args.device,
                          eval_chunk_size=args.eval_chunk_size)
        train_classifier(model, split.X_train, split.y_train,
                         split.X_val, split.y_val, cfg)
        model.eval()

        X = split.X_test.to(args.device)
        y = split.y_test.to(args.device)
        curve = []
        for rap in rapidities:
            gen = torch.Generator().manual_seed(1000 + int(rap * 100))
            if rap == 0.0:
                aucs = [_auc(model, X, y, args.eval_chunk_size)]
            else:
                aucs = []
                for _ in range(args.draws):
                    L, _z = random_lorentz_pair(rot_scale=1.0, boost_scale=rap,
                                                generator=gen)
                    aucs.append(_auc(model, boost_jets(X, L.to(args.device)),
                                     y, args.eval_chunk_size))
            t = torch.tensor(aucs)
            curve.append({"rapidity": rap, "auc": t.mean().item(),
                          "auc_std": t.std().item() if len(aucs) > 1 else 0.0,
                          "draws": len(aucs)})
            print(f"  {name:<22} rapidity {rap:<4} AUC {t.mean():.4f}"
                  f" +- {t.std() if len(aucs) > 1 else 0.0:.4f}")

        drop = curve[0]["auc"] - curve[-1]["auc"]
        print(f"  {name:<22} drop over the range: {drop:+.4f}")
        record = {"model": name, "n_params": n_params, "seed": args.seed,
                  "dtype": args.dtype,
                  "n_jets": args.n, "epochs": args.epochs,
                  "n_constituents": args.n_constituents,
                  "so3c_kwargs": so3c_kwargs, "curve": curve, "drop": drop}
        (out_dir / f"boost__{name}__seed{args.seed}.json").write_text(
            json.dumps(record, indent=2))
        rows.append(record)

    print()
    print("%-24s%9s%10s%10s%9s" % ("model", "params", "AUC@0", "AUC@max", "drop"))
    for r in rows:
        print("%-24s%9d%10.4f%10.4f%+9.4f"
              % (r["model"], r["n_params"], r["curve"][0]["auc"],
                 r["curve"][-1]["auc"], -r["drop"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
