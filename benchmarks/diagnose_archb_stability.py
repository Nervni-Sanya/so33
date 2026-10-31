"""
benchmarks.diagnose_archb_stability
-----------------------------------
Pre-flight stability check for Architecture B (``so33_equivariant*``) on
per-constituent jets. Run it before a multi-hour top-tagging run.

Why Architecture B can diverge: the flow dv/dt = -omega_eff(v, v) satisfies

    d||v||/dt <= alpha ||v||^2,   alpha = 4/(3 sqrt 6) ||omega_eff||_F,

so an input of Euclidean norm r can blow up before T once
q = alpha * r * T >= 1. Lifted massless constituents lie on the null cone,
where neither the eta rescaling nor any other equivariant input map limits
their norm, and after global RMS scaling the leading constituents have
r of order 10-50. With the legacy cap (||omega_eff||_F < 1) q can reach
~5, so RK4 (and, earlier, the adjoint pass) overflows and the NaN reaches
every parameter. The cap ``safe_connection_cap(r_max, T)`` keeps q <= 0.5
for every input up to the largest training norm r_max and every value of
the coefficients.

The script
  1. loads the real constituents (``--cache-dir``) or, if none are found,
     toy massless jets (``benchmarks.synthetic.massless_jet_split``);
  2. prints the constituent-norm distribution, the auto cap and the
     coefficient norm at which the legacy cap stops being provably safe;
  3. trains the chosen Architecture-B variant for ``--steps`` steps with
     the connection cap off (legacy) and auto, logging the worst-case
     growth parameter q(t) = alpha(t) * r_max * T and the first
     non-finite loss or gradient.

Run:
    python -m benchmarks.diagnose_archb_stability                      # toy jets
    python -m benchmarks.diagnose_archb_stability --cache-dir data \\
        --canonical-splits --max-train-samples 50000
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
import torch.nn as nn

from so33.activation import BOOST_GROWTH_CONST, safe_connection_cap
from benchmarks.models import SET_MODELS, build_model, max_constituent_norm
from benchmarks.provenance import run_metadata
from benchmarks.synthetic import massless_jet_split
from benchmarks.train import so33_activations


def _load(args):
    if args.cache_dir:
        from benchmarks.datasets import load_top_tagging_constituents
        try:
            # Same loader as the runners (npz, parquet or h5 files).
            return load_top_tagging_constituents(
                args.cache_dir, max_samples=args.max_samples,
                n_constituents=args.n_constituents, seed=args.seed, normalize="global",
                use_canonical_splits=args.canonical_splits,
                max_train_samples=args.max_train_samples,
            ), "real"
        except FileNotFoundError as e:
            print(f"[stability] no top-tagging data under {args.cache_dir} ({e}); "
                  f"using toy massless jets")
    return massless_jet_split(args.toy_jets, args.n_constituents, seed=args.seed), "toy"


def _norm_report(X: torch.Tensor) -> dict:
    real = X[..., 4] > 0
    p4 = X[..., :4][real].double()
    r = p4.norm(dim=-1)
    m2 = p4[:, 0] ** 2 - p4[:, 1:].pow(2).sum(-1)
    q = torch.quantile(r[:200_000], torch.tensor([0.5, 0.99, 0.999], dtype=r.dtype))
    return {
        "n_constituents": int(len(r)),
        "median": float(q[0]), "p99": float(q[1]), "p999": float(q[2]),
        "max": float(r.max()),
        "median_abs_m2_over_E2": float((m2.abs() / p4[:, 0].pow(2).clamp_min(1e-30)).median()),
    }


EUCLIDEAN_BOUND_MODELS = ("so33_equivariant", "so33_equivariant_frozen")


def _legacy_threshold(r_max: float, T: float) -> float | None:
    """||c|| above which the legacy cap (kappa = 1) is no longer provably safe."""
    w = 1.0 / (BOOST_GROWTH_CONST * r_max * T)      # ||omega_eff||_F at q = 1
    if w >= 1.0:
        return None                                 # never: ||omega_eff|| < 1
    return w / (math.sqrt(2.0) * (1.0 - w))         # ||omega_eff|| = s/(1+s), s = sqrt2 ||c||


def _train_probe(split, name: str, cap_mode: str, r_max: float, r_in: float,
                 args) -> dict:
    """Short training run; r_in is the largest norm the activations see."""
    torch.manual_seed(args.seed)
    kw = {"input_norm_bound": r_max} if cap_mode == "auto" else {}
    model = build_model(name, in_features=4, out_features=2,
                        representation="constituents", T=args.T, **kw)
    acts = so33_activations(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    ce = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(args.seed)
    n = len(split.X_train)
    q_max, first_bad, log = 0.0, None, []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), generator=gen)
        opt.zero_grad()
        loss = ce(model(split.X_train[idx]), split.y_train[idx]) + model.regularization_loss()
        bad = None
        if not torch.isfinite(loss):
            bad = "non-finite loss"
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            if not torch.isfinite(gn):
                bad = "non-finite gradient"
        w_eff = max(a.effective_connection_norm() for a in acts)
        q = BOOST_GROWTH_CONST * w_eff * r_in * args.T
        q_max = max(q_max, q)
        if bad is not None:
            first_bad = {"step": step, "reason": bad, "q": q,
                         "max_coeff_norm": max(a.coeff_norm() for a in acts)}
            break
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            log.append({"step": step, "loss": loss.item(), "q": q,
                        "max_coeff_norm": max(a.coeff_norm() for a in acts)})
            print(f"   [{cap_mode:>4}] step {step:5d}  loss {loss.item():.3f}  "
                  f"max||c|| {log[-1]['max_coeff_norm']:.3f}  q {q:.2f}")
    return {"cap_mode": cap_mode,
            "connection_cap": acts[0].connection_cap,
            "steps_run": step, "q_max": q_max, "first_nonfinite": first_bad,
            "log": log}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Directory with top_tagging_*.npz (default: toy jets).")
    p.add_argument("--canonical-splits", action="store_true")
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--n-constituents", type=int, default=32)
    p.add_argument("--toy-jets", type=int, default=20_000)
    p.add_argument("--model", default="so33_equivariant_eta_bounded",
                   choices=[n for n in SET_MODELS if n.startswith("so33_equivariant")],
                   help="Architecture-B variant to probe.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--T", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args(argv)

    split, source = _load(args)
    norms = _norm_report(split.X_train)
    r_max = max_constituent_norm(split.X_train)
    # The Euclidean-bound variants map every input into the unit ball, so
    # their activations only ever see norms < 1 (EquivariantSO33Classifier).
    r_in = 1.0 if args.model in EUCLIDEAN_BOUND_MODELS else r_max
    cap = min(1.0, safe_connection_cap(r_in, args.T))
    thr = _legacy_threshold(r_in, args.T)
    print(f"[stability] data: {source}  {split.summary()}")
    print(f"[stability] train-constituent norms: median {norms['median']:.2f}, "
          f"p99 {norms['p99']:.2f}, p99.9 {norms['p999']:.2f}, max {norms['max']:.2f}; "
          f"median |m^2|/E^2 = {norms['median_abs_m2_over_E2']:.1e}")
    print(f"[stability] largest activation-input norm r = {r_in:.2f} "
          f"({'Euclidean bound' if r_in != r_max else 'raw lifted constituents'}); "
          f"auto connection cap for T = {args.T}: kappa = {cap:.4f} (q <= 0.5 for every c)")
    if thr is None:
        print("[stability] the legacy cap (kappa = 1) is provably safe for this data")
    else:
        print(f"[stability] legacy cap (kappa = 1): blow-up possible once ||c|| > {thr:.3f} "
              f"(init ||c|| ~ 0.04)")

    runs = []
    for mode in ("off", "auto"):
        print(f"[stability] training {args.model} for {args.steps} steps, cap {mode}")
        res = _train_probe(split, args.model, mode, r_max, r_in, args)
        runs.append(res)
        fb = res["first_nonfinite"]
        if fb is None:
            print(f"[stability] cap {mode}: no non-finite step in {res['steps_run']} "
                  f"steps (max q = {res['q_max']:.2f})")
        else:
            print(f"[stability] cap {mode}: FIRST NON-FINITE at step {fb['step']} "
                  f"({fb['reason']}), q = {fb['q']:.2f}, max||c|| = {fb['max_coeff_norm']:.3f}")

    out = pathlib.Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"diagnose_archb_stability__{source}__seed{args.seed}.json"
    path.write_text(json.dumps({
        "experiment": "diagnose_archb_stability", "data": source,
        "dataset": split.name, "model": args.model, "T": args.T,
        "constituent_norms": norms, "activation_input_norm": r_in,
        "auto_connection_cap": cap,
        "legacy_unsafe_above_coeff_norm": thr, "runs": runs,
        "provenance": run_metadata(argv),
    }, indent=2))
    print(f"[stability] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
