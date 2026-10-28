"""
benchmarks.run_boost_ood
------------------------
Out-of-distribution generalisation across boosts (Table 1).

Each sample is a single particle four-vector embedded in R^{3,3} as
(px, py, pz, E, 0, 0), labelled by its invariant mass band (m = 1 or 3).
Every sample is boosted along one spatial axis a in {0, 1, 2} mixed with
the time axis 3, with rapidity drawn from a range. These are SO(1,3)
boosts in the (a, 3) planes; only 3 of the 9 non-compact so(3,3)
generators are exercised. We train on a LOW rapidity range and test both
in distribution and on a HIGH rapidity range the model never saw.

A model that only reads boost invariants generalises across rapidity by
construction; a generic MLP fits the training boost manifold and degrades
out of distribution. With one particle per sample every pairwise term
vanishes, and the per-particle eta-norm is conserved by the SO33 flow, so
the connection coefficients of Architecture B get no gradient from the
loss here (their trained norm collapses to ~0).

Reports, per model: in-distribution and out-of-distribution test AUC/acc,
the gap, and the AUC in rapidity bins over the union of both test sets.
For models with SO33 activations it records the trained coefficient
norms, connection caps and (Architecture B) the equivariance errors
eps_sigma / eps_m at the trained connection on ID and OOD inputs. With
``--save-samples`` (default) it writes per-sample scores, labels,
rapidities and Architecture-B readout features <h, h>_eta to
``<experiment>__<model>__seed<N>__samples.npz`` for
``benchmarks.figure_boost_ood``.

Run:
    python -m benchmarks.run_boost_ood --quick
    python -m benchmarks.run_boost_ood --seeds 0-2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from benchmarks.equivariance import model_equivariance
from benchmarks.models import (
    build_model, is_set_model, max_constituent_norm, _lift_4to6,
)
from benchmarks.provenance import experiment_tag, parse_seeds, run_metadata
from benchmarks.synthetic import generate_boost_invariant_dataset
from benchmarks.tabular_runner import connection_cap_kwargs, so33_summary
from benchmarks.train import train_classifier, TrainConfig

DEFAULT_MODELS = ("so33,so33_multi,so33_signature_only,"
                  "relu_bottleneck,relu_mlp,"
                  "eta_invariants,so33_equivariant,"
                  "so33_equivariant_frozen,"
                  "so33_equivariant_unbounded,"
                  "so33_equivariant_eta_bounded")


def _scores(model, X) -> np.ndarray | None:
    """Signal probabilities, or None if any logit is non-finite (diverged)."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
    if not torch.isfinite(logits).all():
        return None
    return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def _auc(scores: np.ndarray | None, y: np.ndarray) -> float:
    if scores is None or len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def _acc(scores: np.ndarray | None, y: np.ndarray) -> float:
    if scores is None:
        return float("nan")
    return float(((scores > 0.5).astype(np.int64) == y).mean())


def binned_auc(scores: np.ndarray | None, y: np.ndarray, rapidity: np.ndarray,
               edges: np.ndarray, min_count: int = 20) -> list[dict]:
    """AUC in rapidity bins [edges[i], edges[i+1]); NaN for small or one-class bins."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rapidity >= lo) & (rapidity < hi)
        n = int(sel.sum())
        auc = (_auc(scores[sel], y[sel]) if scores is not None and n >= min_count
               else float("nan"))
        out.append({"lo": float(lo), "hi": float(hi), "n": n, "auc": auc})
    return out


def readout_eta_norms(model, X_set: torch.Tensor) -> np.ndarray | None:
    """(B, channels) <h, h>_eta of each Architecture-B channel (K = 1 sets)."""
    acts = getattr(model, "acts", None)
    if not acts or not hasattr(model, "eta"):
        return None
    p = _lift_4to6(X_set[:, 0, :4].to(model.dtype))
    cols = []
    with torch.no_grad():
        for a in acts:
            h = a(p)
            cols.append((h * model.eta * h).sum(-1))
    return torch.stack(cols, dim=-1).cpu().numpy()


def to_set(X6: torch.Tensor) -> torch.Tensor:
    """Repack flat (B, 6) lifted samples as (B, K=1, 5) sets (E, px, py, pz, mask)."""
    p4 = torch.stack([X6[:, 3], X6[:, 0], X6[:, 1], X6[:, 2]], dim=-1)  # E,px,py,pz
    mask = torch.ones(p4.shape[0], 1, dtype=p4.dtype)
    return torch.cat([p4.unsqueeze(1), mask.unsqueeze(-1)], dim=-1)     # (B,1,5)


def _cap_arg(text: str) -> str | float:
    if text in ("auto", "off"):
        return text
    return float(text)


def run_seed(args, seed: int, names: list[str], experiment: str,
             results_dir: pathlib.Path, meta: dict) -> list[dict]:
    half = args.n // 2
    # In-distribution: train + id-test share the low-rapidity regime.
    Xtr, ytr, _ = generate_boost_invariant_dataset(
        n_samples=half, seed=seed,
        rapidity_range=(0.0, args.train_rapidity))
    Xid, yid, rid = generate_boost_invariant_dataset(
        n_samples=half // 2, seed=seed + 100,
        rapidity_range=(0.0, args.train_rapidity))
    # Out-of-distribution: higher rapidity, unseen in training.
    Xood, yood, rood = generate_boost_invariant_dataset(
        n_samples=half // 2, seed=seed + 200,
        rapidity_range=(args.train_rapidity, args.ood_rapidity))

    n_val = int(0.15 * len(Xtr))
    Xva, yva = Xtr[:n_val], ytr[:n_val]
    Xtr, ytr = Xtr[n_val:], ytr[n_val:]

    print(f"[{experiment}] seed {seed}: train={len(Xtr)} (rapidity<= {args.train_rapidity}) | "
          f"id_test={len(Xid)} | ood_test={len(Xood)} "
          f"(rapidity {args.train_rapidity}-{args.ood_rapidity})")

    y_all = torch.cat([yid, yood]).numpy()
    r_all = torch.cat([rid, rood]).numpy()
    split_all = np.concatenate([np.zeros(len(yid), np.int8), np.ones(len(yood), np.int8)])
    edges = np.arange(0.0, args.ood_rapidity + 1e-9, args.rapidity_bin)
    if edges[-1] < args.ood_rapidity:
        edges = np.append(edges, args.ood_rapidity)
    edges[-1] += 1e-9                      # include the top edge

    r_max_set = max_constituent_norm(to_set(Xtr))
    summary = []
    for name in names:
        # The equivariant architectures consume per-particle sets (B, K, 5).
        is_set = is_set_model(name)
        Xtr_m, Xva_m, Xid_m, Xood_m = (
            (to_set(Xtr), to_set(Xva), to_set(Xid), to_set(Xood))
            if is_set else (Xtr, Xva, Xid, Xood)
        )
        in_features = 4 if is_set else 6
        representation = "constituents" if is_set else "flat"
        bound = True if name.startswith("so33") else None
        cap_kwargs = (connection_cap_kwargs(args.connection_cap, r_max_set)
                      if is_set else {})
        model = build_model(name, in_features=in_features, out_features=2,
                            T=0.3, bound_input=bound,
                            representation=representation, **cap_kwargs)
        cfg = TrainConfig(epochs=args.epochs, batch_size=128, lr=3e-3,
                          seed=seed, cosine_schedule=True, grad_clip=1.0)
        print(f"[{experiment}] {name} | train ...")
        res = train_classifier(model, Xtr_m, ytr, Xva_m, yva, cfg)

        s_id, s_ood = _scores(model, Xid_m), _scores(model, Xood_m)
        id_auc, ood_auc = _auc(s_id, yid.numpy()), _auc(s_ood, yood.numpy())
        diverged = bool(res.diverged or s_id is None or s_ood is None)
        s_all = None if diverged else np.concatenate([s_id, s_ood])
        record = {
            "experiment": experiment,
            "model": name,
            "seed": seed,
            "n_params": res.n_params,
            "n_train": len(Xtr),
            "train_rapidity": args.train_rapidity,
            "ood_rapidity": args.ood_rapidity,
            "id_auc": id_auc, "id_acc": _acc(s_id, yid.numpy()),
            "ood_auc": ood_auc, "ood_acc": _acc(s_ood, yood.numpy()),
            "auc_gap": (id_auc - ood_auc) if id_auc == id_auc else float("nan"),
            "binned_auc": binned_auc(s_all, y_all, r_all, edges),
            "walltime_sec": res.walltime_sec,
            "diverged": diverged,
            "train_guard": {"nonfinite_steps": res.nonfinite_steps,
                            "diverged": res.diverged,
                            "first_nonfinite": res.first_nonfinite},
            "config": res.config,
            "history": res.history,
            "provenance": meta,
        }
        so33 = so33_summary(model)
        if so33 is not None:
            record["so33"] = so33
        if is_set:
            record["stability"] = {"connection_cap_mode": args.connection_cap,
                                   "max_train_constituent_norm": r_max_set}
            if not diverged:
                eq_id = model_equivariance(model, Xid_m, seed=seed,
                                           n_groups=args.probe_groups,
                                           rapidity=args.probe_rapidity)
                if eq_id is not None:
                    record["equivariance"] = {
                        "id": eq_id,
                        "ood": model_equivariance(model, Xood_m, seed=seed,
                                                  n_groups=args.probe_groups,
                                                  rapidity=args.probe_rapidity),
                    }
        (results_dir / f"{experiment}__{name}__seed{seed}.json").write_text(
            json.dumps(record, indent=2))
        if args.save_samples:
            arrays = {"labels": y_all.astype(np.int8),
                      "rapidity": r_all.astype(np.float32),
                      "split": split_all}
            if s_all is not None:
                arrays["scores"] = s_all.astype(np.float32)
            hh = (readout_eta_norms(model, torch.cat([Xid_m, Xood_m]))
                  if is_set and not diverged else None)
            if hh is not None:
                arrays["eta_norms"] = hh.astype(np.float32)
            np.savez_compressed(
                results_dir / f"{experiment}__{name}__seed{seed}__samples.npz", **arrays)
        summary.append(record)
        cn = (f"  ||c||={['%.3g' % c for c in so33['coeff_norms']]}" if so33 else "")
        print(f"   id_auc={id_auc:.3f} ood_auc={ood_auc:.3f} "
              f"gap={record['auc_gap']:+.3f}  ({res.walltime_sec:.1f}s)"
              f"{'  DIVERGED' if diverged else ''}{cn}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="Smoke run: 1k samples, 5 epochs.")
    p.add_argument("--models", type=str, default=DEFAULT_MODELS,
                   help="Comma-separated model names. eta_invariants and "
                        "so33_equivariant* are set-based; the *_frozen, "
                        "*_unbounded and *_eta_bounded variants are the "
                        "ablations for the equivariant-OOD failure mode.")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None,
                   help="Seeds to run in one process, e.g. '0-2'. Overrides --seed.")
    p.add_argument("--train-rapidity", type=float, default=0.6,
                   help="Train/ID rapidity drawn from [0, this].")
    p.add_argument("--ood-rapidity", type=float, default=2.5,
                   help="OOD rapidity drawn from [train-rapidity, this]. A wide "
                        "gap from --train-rapidity is where a generic MLP, which "
                        "memorises the training boost manifold, degrades most.")
    p.add_argument("--rapidity-bin", type=float, default=0.25,
                   help="Bin width for the AUC-vs-rapidity record.")
    p.add_argument("--connection-cap", type=_cap_arg, default="auto",
                   help="Architecture-B connection cap: 'auto' (provably "
                        "blow-up-free for the training norms), 'off' (legacy "
                        "1.0) or a number.")
    p.add_argument("--probe-groups", type=int, default=8,
                   help="Random SO(3,3) elements for the equivariance probe.")
    p.add_argument("--probe-rapidity", type=float, default=0.5,
                   help="Scale of the probe's group elements.")
    p.add_argument("--no-save-samples", dest="save_samples", action="store_false",
                   help="Do not write the per-sample npz files.")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args(argv)

    if args.quick:
        args.n, args.epochs = 1_000, 5

    experiment = experiment_tag(
        "boost_ood",
        {"n": args.n, "epochs": args.epochs, "train_rap": args.train_rapidity,
         "ood_rap": args.ood_rapidity, "cap": args.connection_cap},
        {"n": 20_000, "epochs": 40, "train_rap": 0.6, "ood_rap": 2.5, "cap": "auto"})
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    meta = run_metadata(argv)

    summary = []
    for seed in parse_seeds(args.seeds, default=args.seed):
        summary.extend(run_seed(args, seed, names, experiment, results_dir, meta))

    print(f"\n[{experiment}] ID vs OOD AUC (smaller gap = better generalisation)")
    print(f"{'model':<30} {'seed':>4} {'params':>7} {'id_auc':>7} {'ood_auc':>8} {'gap':>7}")
    print("-" * 68)
    for r in sorted(summary, key=lambda r: (-(r['ood_auc'] if r['ood_auc'] == r['ood_auc'] else -1))):
        ood = f"{r['ood_auc']:.3f}" if r['ood_auc'] == r['ood_auc'] else "  nan"
        idv = f"{r['id_auc']:.3f}" if r['id_auc'] == r['id_auc'] else "  nan"
        gap = f"{r['auc_gap']:+.3f}" if r['auc_gap'] == r['auc_gap'] else "  nan"
        print(f"{r['model']:<30} {r['seed']:>4} {r['n_params']:>7} {idv:>7} {ood:>8} {gap:>7}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
