"""
benchmarks.run_efp_baseline
---------------------------
Linear energy-flow-polynomial (EFP) baseline for top tagging: Fisher's linear
discriminant on the EFPs of Komiske, Metodiev, Thaler, "Energy flow
polynomials: a complete linear basis for jet substructure", JHEP 04 (2018)
013, arXiv:1712.07124.

Inputs are those of the neural models in ``benchmarks.run_top_tagging``: the
leading ``--n-constituents`` (32) constituents of every jet, from the same
split (the published Kasieczka train / val / test with ``--canonical-splits``),
read in raw GeV (``normalize="none"``; the EFPs are invariant under a global
rescaling anyway). The real constituents (mask > 0) of each jet are converted
to hadronic coordinates (pT, y, phi) and every EFP with 1 <= d <= ``--dmax``
is computed with the ``energyflow`` package (``EFPSet.batch_compute``) in the
paper's hadronic measure (its Eq. 2.4, energyflow ``measure="hadr"``):

    EFP_G = sum_{i_1..i_N} z_{i_1} ... z_{i_N} prod_{(k,l) in G} theta_{i_k i_l},
    z_i = pT_i / sum_j pT_j,    theta_ij = (Delta y_ij^2 + Delta phi_ij^2)^(beta/2),

with the sums over all ordered N-tuples of constituents and Delta phi wrapped
into [0, pi]. The d = 0 EFP is identically 1 with normalised z and would only
duplicate the intercept, so it is left out. The model then has n_EFP + 1
parameters: 13 / 102 / 1000 for d <= 3 / 5 / 7, which are the EFP counts the
paper quotes (they include d = 0).

The EFPs are standardised with train statistics and a Fisher linear
discriminant (scikit-learn ``LinearDiscriminantAnalysis``, as in the paper) is
fitted on train. Its decision function is the jet score (test AUC, 1/eps_B at
eps_S = 0.3 / 0.5); the accuracy threshold is the one that maximises val
accuracy. Solver: scikit-learn's default ``svd``, without shrinkage. The EFP
basis is extremely collinear (on 4k toy jets the standardised within-class
covariance of the d <= 5 EFPs has condition number ~1e14), and the
discriminating information sits in the near-degenerate directions:
Ledoit-Wolf shrinkage (``--lda-solver lsqr --lda-shrinkage auto``) lowered the
toy test AUC from 0.995 to 0.982. The svd solver holds ~5 copies of the
(n_train, n_EFP) matrix in memory. For d = 7 on a large train set use
``--lda-solver lsqr``, which needs only the per-class p x p covariances.

Departures from the paper: d <= 5 by default (the paper uses d <= 7; the EFPs
of one jet cost ~5 ms per core at d <= 5 and ~90 ms at d <= 7, i.e. ~3 and
~50 core-hours for the 2.0M canonical jets, hence ``--n-jobs`` and
``--cache-efps``); only the leading-K constituents enter (the paper uses all
particles of the jet); the data are the Kasieczka jets (pT 550-650 GeV,
|eta| < 2, R = 0.8), not the paper's Pythia samples. beta = 0.5 is the paper's
choice for top tagging.

Writes ``<experiment>__efp_linear_d<dmax>__seed<N>.json`` (family
``jet_observable``; the experiment name is the one ``run_top_tagging`` gives
the same data options, so ``benchmarks.aggregate`` renders the row next to the
neural models) and, with ``--save-scores``, the per-jet test scores and labels
in ``...__scores.npz``. A non-default beta or LDA setting is appended to the
model name. ``energyflow`` is an optional dependency: ``pip install
energyflow``.

Run:
    # Canonical protocol (d <= 5, beta = 0.5):
    python -m benchmarks.run_efp_baseline --canonical-splits --save-scores

    # The paper's d <= 7, four processes, EFPs cached for later runs (a cache
    # computed at d <= 7 also serves every smaller --dmax):
    python -m benchmarks.run_efp_baseline --canonical-splits --dmax 7 --n-jobs 4 \\
        --lda-solver lsqr --max-train-samples 300000 \\
        --cache-efps data/efps_canonical_d7.npz --save-scores

    # Smoke test on toy jets (no data needed):
    python -m benchmarks.run_efp_baseline --toy-jets 2000 --dmax 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

from benchmarks.datasets import DatasetSplit, load_top_tagging_constituents
from benchmarks.provenance import experiment_tag, run_metadata
from benchmarks.synthetic import generate_massless_jets
from benchmarks.tabular_runner import binary_metrics

MEASURE = "hadr"           # z_i = pT_i / sum pT, theta_ij = Delta R_ij^beta
KAPPA = 1                  # energy weighting; only kappa = 1 is IRC safe
COORDS = "ptyphim"         # energyflow input rows (pT, y, phi)
DEFAULT_DMAX = 5
DEFAULT_BETA = 0.5         # the paper's choice for top tagging (its Fig. 10)
DEFAULT_SOLVER = "svd"
CHUNK_JETS = 10_000        # jets per batch_compute call (bounds memory)
CACHE_FORMAT = 1
SPEC_COLS = ("n", "e", "d", "v", "k", "c", "p", "h")   # energyflow EFPSet.cols
D_COL, P_COL = SPEC_COLS.index("d"), SPEC_COLS.index("p")


def import_energyflow():
    """The optional ``energyflow`` package, or ImportError with the install hint."""
    try:
        import energyflow
    except ImportError as exc:
        raise ImportError(
            "the linear-EFP baseline needs the optional 'energyflow' package "
            "(Komiske, Metodiev, Thaler, arXiv:1712.07124). Install it with:  "
            "pip install energyflow"
        ) from exc
    return energyflow


# ─────────────────────────────────────────────────────────────────────────
# Constituents -> hadronic coordinates -> EFPs
# ─────────────────────────────────────────────────────────────────────────

def p4_to_ptyphi(p4: np.ndarray) -> np.ndarray:
    """(M, 4) four-momenta (E, px, py, pz) with pT > 0 -> (M, 3) (pT, y, phi).

    The rapidity is y = asinh(pz / mT), mT^2 = max(E^2 - pz^2, pT^2). This
    equals energyflow's arctanh(pz / E) for any particle with m^2 >= 0, and it
    stays finite for the float32-rounded massless constituents of the data,
    whose E can come out a hair below |p|: their slightly negative m^2 is
    clamped to the massless value, which gives y = eta. phi = arctan2(py, px)
    is mapped to [0, 2 pi), as in energyflow (its 'hadr' measure wraps
    Delta phi, so the branch does not matter).
    """
    p4 = np.asarray(p4, dtype=np.float64)
    E, px, py, pz = p4[:, 0], p4[:, 1], p4[:, 2], p4[:, 3]
    pt = np.hypot(px, py)
    mt = np.sqrt(np.maximum(E * E - pz * pz, pt * pt))
    rap = np.arcsinh(pz / mt)
    phi = np.arctan2(py, px)
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)
    return np.stack([pt, rap, phi], axis=1)


def jets_to_events(X: np.ndarray) -> list[np.ndarray]:
    """Packed jets (n, K, 5) = (E, px, py, pz, mask) -> n arrays (M_i, 3) of (pT, y, phi).

    Keeps the real constituents (mask > 0) with pT > 0, in their stored order.
    A pT = 0 particle has z = 0 and drops out of every EFP (kappa = 1), but its
    rapidity is infinite, so it is removed rather than converted.
    """
    X = np.asarray(X)
    if len(X) == 0:
        return []
    p4 = X[..., :4].astype(np.float64)
    pt = np.hypot(p4[..., 1], p4[..., 2])
    keep = (X[..., 4] > 0) & (pt > 0)
    hadronic = p4_to_ptyphi(p4[keep])
    counts = keep.sum(axis=1)
    return np.split(hadronic, np.cumsum(counts)[:-1])


def make_efpset(dmax: int, beta: float):
    """energyflow EFPSet of every EFP with 1 <= d <= dmax (connected, then composite)."""
    ef = import_energyflow()
    return ef.EFPSet(("d<=", dmax), ("d>=", 1), measure=MEASURE, beta=beta,
                     kappa=KAPPA, normed=True, coords=COORDS)


def compute_efps(X: np.ndarray, efpset, n_jobs: int = 1,
                 chunk: int = CHUNK_JETS, label: str = "") -> np.ndarray:
    """(n, n_EFP) float64 EFP values of packed jets X (n, K, 5), in chunks of ``chunk`` jets.

    Column j is the EFP with ``efpset.specs[j]``. ``n_jobs`` is passed to
    energyflow's ``batch_compute`` (1 = no process pool).
    """
    n = len(X)
    out = np.empty((n, len(efpset.specs)), dtype=np.float64)
    t0 = last = time.time()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        vals = efpset.batch_compute(jets_to_events(X[start:stop]), n_jobs=n_jobs)
        # batch_compute squeezes a one-event batch to 1-D.
        out[start:stop] = np.asarray(vals, dtype=np.float64).reshape(stop - start, -1)
        now = time.time()
        if stop == n or now - last > 30.0:
            rate = stop / max(now - t0, 1e-9)
            print(f"   EFPs {label}: {stop}/{n} jets ({rate:.0f} jets/s, "
                  f"ETA {(n - stop) / rate:.0f}s)", flush=True)
            last = now
    return out


# ─────────────────────────────────────────────────────────────────────────
# Feature cache
# ─────────────────────────────────────────────────────────────────────────

def data_fingerprint(split: DatasetSplit) -> str:
    """sha256 of the packed jets and labels of all three splits."""
    h = hashlib.sha256()
    for t in (split.X_train, split.y_train, split.X_val, split.y_val,
              split.X_test, split.y_test):
        a = np.ascontiguousarray(t.numpy() if isinstance(t, torch.Tensor) else t)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.reshape(-1).view(np.uint8))
    return h.hexdigest()


def _cache_key(split: DatasetSplit, beta: float, fingerprint: str) -> dict[str, Any]:
    """Everything an EFP cache must match, apart from dmax (a larger one is subset)."""
    return {"format": CACHE_FORMAT, "measure": MEASURE, "beta": float(beta),
            "kappa": KAPPA, "normed": True, "coords": COORDS,
            "n_constituents": int(split.X_train.shape[1]),
            "sizes": [len(split.X_train), len(split.X_val), len(split.X_test)],
            "fingerprint": fingerprint}


def save_efp_cache(path: pathlib.Path, feats: list[np.ndarray], specs: np.ndarray,
                   meta: dict[str, Any]) -> None:
    """Write the EFPs atomically (a temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:   # a file object: np.savez must not append ".npz"
        np.savez(fh, train=feats[0], val=feats[1], test=feats[2],
                 specs=np.asarray(specs, dtype=np.int64),
                 meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_efp_cache(path: pathlib.Path, key: dict[str, Any],
                   dmax: int) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """EFPs with d <= dmax from a cache written for the same data and measure.

    Raises ValueError (and leaves the file alone) if the cache was computed for
    other data, another measure or a smaller dmax. Selecting the d <= dmax
    columns of a larger-dmax cache gives exactly the columns, in the same
    order, that ``make_efpset(dmax, ...)`` computes.
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["meta"].item())
        bad = sorted(k for k in key if meta.get(k) != key[k])
        if bad:
            raise ValueError(
                f"EFP cache {path} was computed for a different setting "
                f"(mismatch in {', '.join(bad)}); delete it or pass another "
                f"--cache-efps path.")
        if int(meta["dmax"]) < dmax:
            raise ValueError(
                f"EFP cache {path} holds d <= {meta['dmax']} only, but --dmax is "
                f"{dmax}; delete it or pass another --cache-efps path.")
        specs = z["specs"]
        cols = specs[:, D_COL] <= dmax
        feats = [np.ascontiguousarray(z[name][:, cols])
                 for name in ("train", "val", "test")]
    return feats, specs[cols], meta


def efp_features(split: DatasetSplit, dmax: int, beta: float, n_jobs: int = 1,
                 cache_path: str | None = None,
                 label: str = "") -> tuple[list[np.ndarray], dict[str, Any]]:
    """[train, val, test] EFP matrices (1 <= d <= dmax) and a summary of how they were made."""
    t0 = time.time()
    path = pathlib.Path(cache_path) if cache_path else None
    key = _cache_key(split, beta, data_fingerprint(split)) if path else None
    cache_hit = False
    if path is not None and path.is_file():
        feats, specs, meta = load_efp_cache(path, key, dmax)
        cache_hit = True
        print(f"[{label}] EFPs (d <= {dmax}) loaded from {path} "
              f"(computed with d <= {meta['dmax']})")
    else:
        efpset = make_efpset(dmax, beta)
        specs = np.asarray(efpset.specs, dtype=np.int64)
        print(f"[{label}] computing {len(specs)} EFPs (1 <= d <= {dmax}, "
              f"beta = {beta:g}) per jet with energyflow, n_jobs = {n_jobs}")
        feats = [compute_efps(X.numpy(), efpset, n_jobs=n_jobs, label=name)
                 for name, X in (("train", split.X_train), ("val", split.X_val),
                                 ("test", split.X_test))]
        if path is not None:
            version = import_energyflow().__version__
            save_efp_cache(path, feats, specs,
                           {**key, "dmax": int(dmax), "energyflow": version})
            print(f"[{label}] EFPs cached in {path}")
    for name, F in zip(("train", "val", "test"), feats):
        n_bad = int((~np.isfinite(F)).any(axis=1).sum())
        if n_bad:
            raise ValueError(f"{n_bad} {name} jets have non-finite EFPs; check "
                             f"the constituents (pT > 0, finite four-momenta).")
    n_empty = sum(int(((X[..., 4] > 0).sum(dim=-1) == 0).sum())
                  for X in (split.X_train, split.X_val, split.X_test))
    if n_empty:
        print(f"[{label}] warning: {n_empty} jets have no constituents; "
              f"their EFPs are all zero.")
    degrees, counts = np.unique(specs[:, D_COL], return_counts=True)
    info = {
        "n_efps": int(len(specs)),
        "n_prime": int((specs[:, P_COL] == 1).sum()),
        "per_degree": {str(int(d)): int(c) for d, c in zip(degrees, counts)},
        "n_empty_jets": n_empty,
        "cache": cache_path,
        "cache_hit": cache_hit,
        "efp_sec": time.time() - t0,
    }
    return feats, info


# ─────────────────────────────────────────────────────────────────────────
# Linear classifier
# ─────────────────────────────────────────────────────────────────────────

def fit_linear_discriminant(F_train: np.ndarray, y_train: np.ndarray,
                            solver: str = DEFAULT_SOLVER,
                            shrinkage: str | float | None = None):
    """Train-standardised Fisher linear discriminant: (scaler, lda)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(F_train)
    lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    lda.fit(scaler.transform(F_train), y_train)
    return scaler, lda


def lda_scores(scaler, lda, F: np.ndarray) -> np.ndarray:
    """Signal score = LDA decision function (larger = more top-like)."""
    return np.asarray(lda.decision_function(scaler.transform(F)), dtype=np.float64)


def best_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Cut t that maximises the accuracy of ``score > t``: returns (t, accuracy).

    Cuts are placed midway between consecutive distinct scores (or at +-inf);
    among equally good cuts the lowest is taken.
    """
    s = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y).astype(np.int64)
    order = np.argsort(s, kind="mergesort")
    s, yy = s[order], yy[order]
    n = len(s)
    # A cut below sorted position k calls the k lowest scores background.
    neg_below = np.concatenate([[0], np.cumsum(yy == 0)])
    pos_below = np.concatenate([[0], np.cumsum(yy == 1)])
    correct = neg_below + (pos_below[-1] - pos_below)
    allowed = np.ones(n + 1, dtype=bool)
    allowed[1:n] = s[1:] > s[:-1]
    k = int(np.argmax(np.where(allowed, correct, -1)))
    if k == 0:
        t = -np.inf
    elif k == n:
        t = np.inf
    else:
        t = 0.5 * (s[k - 1] + s[k])
    return float(t), float(correct[k]) / n


# ─────────────────────────────────────────────────────────────────────────
# Data, naming
# ─────────────────────────────────────────────────────────────────────────

def toy_jet_split(n_jets: int, n_constituents: int = 32, seed: int = 0) -> DatasetSplit:
    """Toy massless jets (``generate_massless_jets``) in raw GeV, packed like
    the loader, split 70/15/15 as in ``massless_jet_split``."""
    X, y = generate_massless_jets(n_jets, n_constituents, seed=seed)
    mask = (X.abs().sum(dim=-1) > 0).to(X.dtype)
    packed = torch.cat([X * mask.unsqueeze(-1), mask.unsqueeze(-1)], dim=-1)
    n_tr, n_va = int(0.7 * n_jets), int(0.15 * n_jets)
    return DatasetSplit(
        X_train=packed[:n_tr], y_train=y[:n_tr],
        X_val=packed[n_tr:n_tr + n_va], y_val=y[n_tr:n_tr + n_va],
        X_test=packed[n_tr + n_va:], y_test=y[n_tr + n_va:],
        name=f"toy_massless_jets(n={n_jets},k={n_constituents},norm=none)",
        n_features=4, n_classes=2,
    )


def load_split(args: argparse.Namespace) -> DatasetSplit:
    """The jets of the experiment, in raw GeV."""
    if args.toy_jets:
        return toy_jet_split(args.toy_jets, args.n_constituents, seed=args.seed)
    return load_top_tagging_constituents(
        cache_dir=args.cache_dir, max_samples=args.max_samples,
        n_constituents=args.n_constituents, seed=args.seed, standardise=False,
        normalize="none", use_canonical_splits=args.canonical_splits,
        max_train_samples=args.max_train_samples,
    )


def experiment_name(args: argparse.Namespace) -> str:
    """The experiment name ``run_top_tagging`` gives the same data options."""
    options: dict[str, Any] = {"k": args.n_constituents}
    defaults: dict[str, Any] = {"k": 32}
    if args.toy_jets:
        return experiment_tag("toy_jets", options, defaults)
    if args.canonical_splits:
        options["ntrain"], defaults["ntrain"] = args.max_train_samples, None
        return experiment_tag("top_tagging_canonical", options, defaults)
    options["n"], defaults["n"] = args.max_samples, 100_000
    return experiment_tag("top_tagging_constituents", options, defaults)


def model_name(dmax: int, beta: float, solver: str,
               shrinkage: str | float | None) -> str:
    """``efp_linear_d<dmax>``, plus any non-default beta / LDA setting."""
    return experiment_tag(
        f"efp_linear_d{dmax}",
        {"beta": beta, "lda": solver, "shrink": shrinkage},
        {"beta": DEFAULT_BETA, "lda": DEFAULT_SOLVER, "shrink": None})


def _shrinkage_arg(text: str) -> str | float | None:
    if text in ("none", "None"):
        return None
    if text == "auto":
        return text
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("shrinkage must be 'auto', 'none' or in [0, 1]")
    return value


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=str, default="data",
                   help="Directory containing top_tagging_*.npz.")
    p.add_argument("--canonical-splits", action="store_true",
                   help="Published Kasieczka train/val/test files instead of a "
                        "random 70/15/15 re-split.")
    p.add_argument("--max-samples", type=int, default=100_000,
                   help="Jets read for the internal 70/15/15 split.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap on the canonical train jets (val/test stay full).")
    p.add_argument("--n-constituents", type=int, default=32,
                   help="Leading constituents per jet that enter the EFPs.")
    p.add_argument("--seed", type=int, default=0,
                   help="Internal / toy split seed (the LDA fit is deterministic).")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-scores", action="store_true",
                   help="Also write per-jet test scores and labels to "
                        "<experiment>__<model>__seed<N>__scores.npz.")
    p.add_argument("--toy-jets", type=int, default=None, metavar="N",
                   help="Use N toy massless jets (benchmarks.synthetic) instead "
                        "of the top-tagging files.")
    p.add_argument("--dmax", type=int, default=DEFAULT_DMAX,
                   help="Maximum EFP degree (number of edges); the paper uses 7.")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA,
                   help="Angular exponent: theta_ij = Delta R_ij^beta.")
    p.add_argument("--lda-solver", choices=["svd", "lsqr"], default=DEFAULT_SOLVER,
                   help="LinearDiscriminantAnalysis solver (svd: scikit-learn "
                        "default; lsqr: p x p covariances only, for d = 7).")
    p.add_argument("--lda-shrinkage", type=_shrinkage_arg, default=None,
                   help="Covariance shrinkage for lsqr: 'auto' (Ledoit-Wolf), "
                        "a number in [0, 1], or none (default).")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Processes for energyflow's batch_compute (1: no pool).")
    p.add_argument("--cache-efps", type=str, default=None, metavar="PATH.npz",
                   help="Load the EFPs from this file if it holds them for the "
                        "same data and measure (any dmax >= --dmax), else "
                        "compute and save them there.")
    args = p.parse_args(argv)

    if not 1 <= args.dmax <= 10:
        p.error("--dmax must be between 1 and 10 (energyflow ships the d <= 10 graphs)")
    if args.beta <= 0.0:
        p.error("--beta must be positive")
    if args.n_jobs < 1:
        p.error("--n-jobs must be >= 1")
    if args.lda_solver == "svd" and args.lda_shrinkage is not None:
        p.error("--lda-shrinkage needs --lda-solver lsqr")
    if args.toy_jets is not None and args.toy_jets < 20:
        p.error("--toy-jets needs at least 20 jets")

    try:
        ef = import_energyflow()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    experiment = experiment_name(args)
    model = model_name(args.dmax, args.beta, args.lda_solver, args.lda_shrinkage)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(argv)

    split = load_split(args)
    print(f"[{experiment}] {split.summary()}")
    dataset = split.name
    y_train, y_val, y_test = (t.numpy().astype(np.int64)
                              for t in (split.y_train, split.y_val, split.y_test))
    (F_train, F_val, F_test), info = efp_features(
        split, args.dmax, args.beta, n_jobs=args.n_jobs,
        cache_path=args.cache_efps, label=experiment)
    del split                      # the constituents are no longer needed

    t_fit = time.time()
    scaler, lda = fit_linear_discriminant(F_train, y_train, args.lda_solver,
                                          args.lda_shrinkage)
    s_train, s_val, s_test = (lda_scores(scaler, lda, F)
                              for F in (F_train, F_val, F_test))
    threshold, val_acc = best_threshold(s_val, y_val)
    test_acc = float(((s_test > threshold).astype(np.int64) == y_test).mean())
    fit_sec = time.time() - t_fit

    from sklearn.metrics import roc_auc_score
    n_efps = F_train.shape[1]
    record: dict[str, Any] = {
        "experiment":   experiment,
        "dataset":      dataset,
        "model":        model,
        "family":       "jet_observable",
        "seed":         args.seed,
        "n_params":     n_efps + 1,
        "model_size":   f"{n_efps} EFPs (d<={args.dmax}) + bias",
        "n_train":      len(y_train),
        "n_val":        len(y_val),
        "n_test":       len(y_test),
        "n_features":   n_efps,
        "walltime_sec": info["efp_sec"] + fit_sec,
        "diverged":     False,
        "train_metrics": {
            "final_val_acc": val_acc,
            "val_auc":       float(roc_auc_score(y_val, s_val)),
        },
        "test_metrics": {"test_acc": test_acc, **binary_metrics(s_test, y_test)},
        "config": {
            "dmax": args.dmax, "beta": args.beta, "measure": MEASURE,
            "kappa": KAPPA, "normed": True, "coords": COORDS,
            "efp_degrees": [1, args.dmax],
            "standardise": "train mean/std",
            "lda_solver": args.lda_solver, "lda_shrinkage": args.lda_shrinkage,
            "threshold": "max val accuracy",
            "n_constituents": args.n_constituents, "normalize": "none",
            "canonical_splits": args.canonical_splits,
            "max_samples": args.max_samples,
            "max_train_samples": args.max_train_samples,
            "toy_jets": args.toy_jets, "cache_dir": args.cache_dir,
            "cache_efps": args.cache_efps, "n_jobs": args.n_jobs,
            "energyflow_version": ef.__version__,
        },
        "efp": {**info, "fit_sec": fit_sec, "val_threshold": threshold,
                "train_auc": float(roc_auc_score(y_train, s_train))},
        "provenance": meta,
    }
    stem = f"{experiment}__{model}__seed{args.seed}"
    (results_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
    if args.save_scores:
        np.savez_compressed(results_dir / f"{stem}__scores.npz",
                            scores=s_test.astype(np.float32),
                            labels=y_test.astype(np.int8))

    tm = record["test_metrics"]
    print(f"[{experiment}] {model} ({n_efps + 1} params): val_acc={val_acc:.4f} "
          f"test_acc={test_acc:.4f} test_auc={tm['test_auc']:.4f} "
          f"1/eB@0.3={tm['bg_rej_30']:.1f} 1/eB@0.5={tm['bg_rej_50']:.1f}  "
          f"({record['walltime_sec']:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
