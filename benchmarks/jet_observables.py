"""
benchmarks.jet_observables
--------------------------
Jet-level observables for the cut / BDT baselines of the top-tagging study
(``benchmarks.run_jet_baselines``): the jet mass and pT, and the N-subjettiness
ratios tau21 = tau2 / tau1 and tau32 = tau3 / tau2 of Thaler and Van Tilburg,
"Identifying boosted objects with N-subjettiness", JHEP 03 (2011) 015,
arXiv:1011.2268, and "Maximizing boosted top identification by minimizing
N-subjettiness", JHEP 02 (2012) 093, arXiv:1108.2701.

Input: four-momenta p4 (n, K, 4) = (E, px, py, pz) in GeV and a mask (n, K),
> 0 for a real constituent, e.g. the leading-K constituents returned by
``load_top_tagging_constituents(..., normalize="none")``. Padding may sit in
any slot. Everything is numpy, float64, vectorised over the jets and processed
in blocks of jets so that the memory stays bounded for the ~2M canonical jets.

Kinematics. pT = hypot(px, py), phi = atan2(py, px) in (-pi, pi] and the
rapidity y = 0.5 log((E + pz) / (E - pz)), evaluated as in FastJet's PseudoJet:
y = sign(pz) log((E + |pz|) / mT) with mT^2 = pT^2 + max(m^2, 0). That is the
same number whenever E > |pz|, and it stays finite for the float32-rounded
massless constituents of the data, whose E can come out at or below |pz|
(their slightly negative m^2 is clamped to 0, which gives y = eta). Padding,
and a particle with pT = 0 along the beam, gets y = 0 and never enters a
distance. The jet four-momentum is the sum over the real constituents and
M = sqrt(max(E^2 - |p|^2, 0)).

N-subjettiness, normalised measure with beta = 1 and R0 = 0.8:

    tau_N = sum_i pT_i min_k DeltaR_ik^beta / (sum_i pT_i R0^beta),
    DeltaR_ik^2 = (y_i - y_k)^2 + (phi_i - phi_k)^2,  Delta phi wrapped to (-pi, pi],

with i over the real constituents with pT > 0 (a pT = 0 particle has zero
weight and no rapidity) and k over N axes. The axes are EXCLUSIVE-kT axes, as
FastJet's ExclusiveJetAxes (KT_Axes of the Nsubjettiness contrib): the
constituents are clustered with the kT distance d_ij = min(pT_i^2, pT_j^2)
DeltaR_ij^2, without a beam distance (FastJet's kt algorithm at R =
max_allowable_R = 1000 never uses one), with E-scheme recombination (the
four-vector sum), until N pseudojets remain; those N pseudojets are the axes.
tau_1, tau_2 and tau_3 come from one clustering pass that records the
pseudojets when 3, 2 and 1 of them remain. These are exclusive-kT axes with
NO one-pass minimisation (not OnePass_KT_Axes) and no winner-take-all
recombination. A jet with at most N such constituents has tau_N = 0 (every
constituent is its own axis; FastJet would pad the axes with zero vectors and
warn that the result is unpredictable). The ratios are set to 1 when their
denominator vanishes ("the extra axis does not help"): tau_{N-1} = 0 only
happens for a jet with fewer than N constituents (or exactly collinear ones),
and then tau_N = 0 as well.

The clustering keeps, for every pseudojet, its nearest neighbour in the kT
distance (the bookkeeping of FastJet's NNH strategy): a merge recomputes the
merged pseudojet's row and the rows that pointed at either merged pseudojet,
and only compares the others with the merged pseudojet, so a jet of K
constituents costs O(K^2) instead of the naive O(K^3). The jets of a block
are clustered in lock step, sorted by constituent count so that the jets
still clustering always form a prefix of the block.

Pure numpy because FastJet (and fjcontrib) has no Windows wheels and the
study has to run there as well. Where the ``fastjet`` package is installed,
tests/test_jet_observables.py checks the clustering against
fastjet.ClusterSequence(kt_algorithm, R = 1000, E_scheme).exclusive_jets(N).

Run:
    # Throughput on toy jets and their observable AUCs (no data needed):
    python -m benchmarks.jet_observables --toy-jets 20000
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

BETA = 1.0                   # angular exponent of the N-subjettiness measure
R0 = 0.8                     # characteristic jet radius of the normalisation
N_MAX = 3                    # tau_1 .. tau_3
RATIO_IF_ZERO = 1.0          # tau_N / tau_{N-1} when tau_{N-1} = 0
AXES = "exclusive_kt"        # E-scheme kT to N pseudojets, no beam, no minimisation
CHUNK_JETS = 4096            # jets per block
MAX_PAIR_ELEMENTS = 1 << 21  # cap on jets * K^2 of a block's initial pair matrix
OBSERVABLES = ("mass", "pt", "n_constituents", "tau1", "tau2", "tau3",
               "tau21", "tau32")
TWO_PI = 2.0 * np.pi


# ─────────────────────────────────────────────────────────────────────────
# Kinematics
# ─────────────────────────────────────────────────────────────────────────

def _rap_phi(E: np.ndarray, px: np.ndarray, py: np.ndarray, pz: np.ndarray,
             pt2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FastJet's rapidity (sign(pz) log((E + |pz|) / mT)) and atan2 azimuth.

    mT^2 = pT^2 + max(m^2, 0); y = 0 where mT = 0 (padding, or pT = 0 along
    the beam).
    """
    mt2 = pt2 + np.maximum(E * E - pt2 - pz * pz, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rap = np.log((E + np.abs(pz)) / np.sqrt(mt2))
    rap = np.where(mt2 > 0.0, np.where(pz > 0.0, rap, -rap), 0.0)
    return rap, np.arctan2(py, px)


def constituent_kinematics(p4: np.ndarray, mask: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pT, y, phi) of every slot of p4 (..., 4) = (E, px, py, pz); 0 for padding."""
    p4 = np.asarray(p4, dtype=np.float64)
    real = np.asarray(mask) > 0
    E, px, py, pz = (np.where(real, p4[..., c], 0.0) for c in range(4))
    pt2 = px * px + py * py
    rap, phi = _rap_phi(E, px, py, pz, pt2)
    return np.sqrt(pt2), rap, phi


def delta_r2(rap1: np.ndarray, phi1: np.ndarray, rap2: np.ndarray,
             phi2: np.ndarray) -> np.ndarray:
    """Delta y^2 + Delta phi^2 (broadcasting), Delta phi wrapped to (-pi, pi]."""
    dphi = np.subtract(phi1, phi2)
    np.abs(dphi, out=dphi)
    np.minimum(dphi, TWO_PI - dphi, out=dphi)
    np.square(dphi, out=dphi)
    drap = np.subtract(rap1, rap2)
    np.square(drap, out=drap)
    drap += dphi
    return drap


def jet_four_momentum(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(n, 4) sum of the real constituents' (E, px, py, pz), in float64."""
    p4 = np.asarray(p4, dtype=np.float64)
    real = np.asarray(mask) > 0
    return np.where(real[..., None], p4, 0.0).sum(axis=-2)


def jet_mass_pt(p4: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jet mass sqrt(max(E^2 - |p|^2, 0)) and jet pT of (n, K, 4) constituents."""
    P = jet_four_momentum(p4, mask)
    m2 = P[:, 0] ** 2 - P[:, 1] ** 2 - P[:, 2] ** 2 - P[:, 3] ** 2
    return np.sqrt(np.maximum(m2, 0.0)), np.hypot(P[:, 1], P[:, 2])


def tau_ratio(num: np.ndarray, den: np.ndarray,
              if_zero: float = RATIO_IF_ZERO) -> np.ndarray:
    """num / den, and ``if_zero`` (default 1) where den = 0."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, float(if_zero))
    np.divide(num, den, out=out, where=den > 0.0)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Exclusive-kT clustering and N-subjettiness
# ─────────────────────────────────────────────────────────────────────────

def block_size(n_constituents: int, chunk: int = CHUNK_JETS) -> int:
    """Jets per block: at most ``chunk`` and jets * K^2 <= MAX_PAIR_ELEMENTS."""
    per_jet = max(1, n_constituents * n_constituents)
    return int(max(1, min(chunk, MAX_PAIR_ELEMENTS // per_jet)))


def _nearest_neighbours(pt2: np.ndarray, rap: np.ndarray, phi: np.ndarray,
                        active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pseudojet: kT distance to, and slot of, its nearest active neighbour.

    (B, K) inputs. Inactive slots get distance inf and neighbour -1; an active
    slot without a neighbour gets inf.
    """
    K = pt2.shape[1]
    d = np.minimum(pt2[:, :, None], pt2[:, None, :])
    d *= delta_r2(rap[:, :, None], phi[:, :, None], rap[:, None, :], phi[:, None, :])
    ok = active[:, :, None] & active[:, None, :]
    ok[:, np.arange(K), np.arange(K)] = False
    np.putmask(d, ~ok, np.inf)
    idx = d.argmin(axis=2)
    dist = np.take_along_axis(d, idx[..., None], axis=2)[..., 0]
    idx[~active] = -1
    return dist, idx


def _tau_with_axes(c_pt: np.ndarray, c_rap: np.ndarray, c_phi: np.ndarray,
                   den: np.ndarray, ax_rap: np.ndarray, ax_phi: np.ndarray,
                   beta: float) -> np.ndarray:
    """tau = sum_i pT_i min_k DeltaR_ik^beta / den; constituents (S, K), axes (S, N)."""
    d2 = delta_r2(c_rap[:, :, None], c_phi[:, :, None],
                  ax_rap[:, None, :], ax_phi[:, None, :]).min(axis=2)
    w = np.sqrt(d2) if beta == 1.0 else d2 ** (0.5 * beta)
    return (c_pt * w).sum(axis=1) / den


def _exclusive_kt_taus(p4: np.ndarray, use: np.ndarray, n_max: int,
                       beta: float, r0: float) -> np.ndarray:
    """(B, n_max) tau_1..tau_n_max of one block; p4 (B, K, 4) float64, use (B, K) bool.

    ``use`` marks the constituents that are clustered and summed over (real,
    pT > 0). Jets with at most N of them keep tau_N = 0.
    """
    B, K = use.shape
    tau = np.zeros((B, n_max))
    count = use.sum(axis=1)
    order = np.argsort(-count, kind="stable")
    m = int(np.count_nonzero(count > 1))
    if m == 0:
        return tau
    # Jets with >= 2 constituents, by descending count. Every merge step
    # removes one pseudojet from each jet that still has two or more, so the
    # order is kept and the jets still clustering are a prefix [:alive].
    jets = order[:m]
    count = count[jets]
    used = use[jets]
    P = np.where(used[..., None], p4[jets], 0.0)                  # (m, K, 4) pseudojets
    pt2 = P[..., 1] ** 2 + P[..., 2] ** 2
    rap, phi = _rap_phi(P[..., 0], P[..., 1], P[..., 2], P[..., 3], pt2)
    c_pt, c_rap, c_phi = np.sqrt(pt2), rap.copy(), phi.copy()   # the particles
    den = c_pt.sum(axis=1) * r0 ** beta
    active = used.copy()
    nn_dist, nn_idx = _nearest_neighbours(pt2, rap, phi, active)
    out = np.zeros((m, n_max))

    alive = m
    while True:
        alive = int(np.count_nonzero(count[:alive] > 1))
        if alive == 0:
            break
        r = np.arange(alive)
        Pv, pt2v, rapv, phiv = P[:alive], pt2[:alive], rap[:alive], phi[:alive]
        actv, ndv, niv = active[:alive], nn_dist[:alive], nn_idx[:alive]

        # The closest pair of every jet; merge slot c into slot a (E-scheme).
        i = ndv.argmin(axis=1)
        j = niv[r, i]
        a, c = np.minimum(i, j), np.maximum(i, j)
        pa = Pv[r, a] + Pv[r, c]                                  # (alive, 4)
        Pv[r, a] = pa
        Pv[r, c] = 0.0
        actv[r, c] = False
        count[:alive] -= 1
        pa2 = pa[:, 1] ** 2 + pa[:, 2] ** 2
        ra, fa = _rap_phi(pa[:, 0], pa[:, 1], pa[:, 2], pa[:, 3], pa2)
        pt2v[r, a], rapv[r, a], phiv[r, a] = pa2, ra, fa
        pt2v[r, c] = 0.0

        # kT distance of the merged pseudojet to every pseudojet of its jet.
        dA = np.minimum(pt2v, pa2[:, None])
        dA *= delta_r2(rapv, phiv, ra[:, None], fa[:, None])
        np.putmask(dA, ~actv, np.inf)
        dA[r, a] = np.inf
        # Rows whose neighbour was a or c are recomputed below (inactive rows
        # hold the neighbour -1, so they never match); every other row keeps
        # its neighbour unless the merged pseudojet is now closer.
        stale = (niv == a[:, None]) | (niv == c[:, None])
        stale[r, a] = False
        stale[r, c] = False
        closer = dA < ndv                     # dA = inf on a and on inactive slots
        np.copyto(ndv, dA, where=closer)
        np.copyto(niv, a[:, None], where=closer)
        best = dA.argmin(axis=1)
        niv[r, a], ndv[r, a] = best, dA[r, best]
        niv[r, c], ndv[r, c] = -1, np.inf
        rb, rk = np.nonzero(stale)
        if rb.size:
            d = np.minimum(pt2v[rb], pt2v[rb, rk][:, None])
            d *= delta_r2(rapv[rb], phiv[rb], rapv[rb, rk][:, None], phiv[rb, rk][:, None])
            np.putmask(d, ~actv[rb], np.inf)
            q = np.arange(rb.size)
            d[q, rk] = np.inf
            best = d.argmin(axis=1)
            niv[rb, rk], ndv[rb, rk] = best, d[q, best]

        # Jets now made of exactly N pseudojets: those are their N axes.
        cnt = count[:alive]
        for n in range(1, n_max + 1):
            sel = np.flatnonzero(cnt == n)
            if sel.size == 0:
                continue
            cols = np.nonzero(actv[sel])[1].reshape(sel.size, n)
            out[sel, n - 1] = _tau_with_axes(
                c_pt[sel], c_rap[sel], c_phi[sel], den[sel],
                np.take_along_axis(rapv[sel], cols, axis=1),
                np.take_along_axis(phiv[sel], cols, axis=1), beta)

    tau[jets] = out
    return tau


def _blocks(p4: np.ndarray, mask: np.ndarray, chunk: int, label: str = ""):
    """Yield (start, stop, p4 block as float64, real mask, clustered mask) per block.

    The clustered constituents are the real ones with pT > 0. Raises ValueError
    if a real constituent has a non-finite component (padding is not read).
    """
    real = np.asarray(mask) > 0
    n, K = real.shape
    step = block_size(K, chunk)
    for start in range(0, n, step):
        stop = min(start + step, n)
        P = np.asarray(p4[start:stop], dtype=np.float64)
        rb = real[start:stop]
        if not np.isfinite(P[rb]).all():
            raise ValueError(f"non-finite four-momenta among the real constituents "
                             f"of jets {start}..{stop - 1} {label}".rstrip())
        with np.errstate(invalid="ignore"):
            use = rb & (np.hypot(P[..., 1], P[..., 2]) > 0.0)
        yield start, stop, P, rb, use


def nsubjettiness(p4: np.ndarray, mask: np.ndarray, n_max: int = N_MAX,
                  beta: float = BETA, r0: float = R0,
                  chunk: int = CHUNK_JETS) -> np.ndarray:
    """(n, n_max) tau_1..tau_n_max of jets p4 (n, K, 4) GeV with mask (n, K).

    Exclusive-kT axes (E-scheme, no beam distance, no minimisation); see the
    module docstring. Jets are processed in blocks of ``block_size(K, chunk)``.
    """
    p4 = np.asarray(p4)
    tau = np.zeros((len(p4), n_max))
    for start, stop, P, _, use in _blocks(p4, mask, chunk):
        tau[start:stop] = _exclusive_kt_taus(P, use, n_max, beta, r0)
    return tau


def compute_jet_observables(p4: np.ndarray, mask: np.ndarray, beta: float = BETA,
                            r0: float = R0, chunk: int = CHUNK_JETS,
                            label: str = "", verbose: bool = True
                            ) -> dict[str, np.ndarray]:
    """Every name in OBSERVABLES for jets p4 (n, K, 4) = (E, px, py, pz) GeV, mask (n, K).

    Returns float64 arrays (n,) (``n_constituents``: int64, the real slots);
    jets are converted to float64 and clustered block by block.
    """
    p4 = np.asarray(p4)
    real = np.asarray(mask) > 0
    n = len(real)
    out = {name: np.zeros(n) for name in OBSERVABLES}
    out["n_constituents"] = real.sum(axis=1).astype(np.int64)
    t0 = last = time.time()
    for start, stop, P, rb, use in _blocks(p4, real, chunk, label):
        out["mass"][start:stop], out["pt"][start:stop] = jet_mass_pt(P, rb)
        tau = _exclusive_kt_taus(P, use, N_MAX, beta, r0)
        for k in range(N_MAX):
            out[f"tau{k + 1}"][start:stop] = tau[:, k]
        now = time.time()
        if verbose and (stop == n or now - last > 30.0):
            rate = stop / max(now - t0, 1e-9)
            print(f"   jet observables {label}: {stop}/{n} jets ({rate:.0f} jets/s, "
                  f"ETA {(n - stop) / rate:.0f}s)", flush=True)
            last = now
    out["tau21"] = tau_ratio(out["tau2"], out["tau1"])
    out["tau32"] = tau_ratio(out["tau3"], out["tau2"])
    return out


def observables_from_packed(X: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
    """``compute_jet_observables`` of packed jets X (n, K, 5) = (E, px, py, pz, mask)."""
    X = np.asarray(X)
    return compute_jet_observables(X[..., :4], X[..., 4], **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Timing on toy jets
# ─────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--toy-jets", type=int, default=20_000,
                   help="Number of toy massless jets (benchmarks.synthetic).")
    p.add_argument("--n-constituents", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-jets", type=int, default=CHUNK_JETS,
                   help="Jets per block (also capped by jets * K^2 <= 2^21).")
    args = p.parse_args(argv)
    if args.toy_jets < 2 or args.n_constituents < 1 or args.chunk_jets < 1:
        p.error("--toy-jets must be >= 2, --n-constituents and --chunk-jets >= 1")

    from sklearn.metrics import roc_auc_score
    from benchmarks.synthetic import generate_massless_jets

    t0 = time.time()
    X, y = generate_massless_jets(args.toy_jets, args.n_constituents, seed=args.seed)
    X, y = X.numpy(), y.numpy()
    mask = (np.abs(X).sum(axis=-1) > 0).astype(np.float64)
    t_gen = time.time() - t0
    t0 = time.time()
    obs = compute_jet_observables(X, mask, chunk=args.chunk_jets, verbose=False)
    t_obs = time.time() - t0
    per_jet = t_obs / args.toy_jets
    print(f"{args.toy_jets} toy jets, K = {args.n_constituents} "
          f"(block {block_size(args.n_constituents, args.chunk_jets)} jets): "
          f"generated in {t_gen:.1f}s, observables in {t_obs:.2f}s "
          f"({1e6 * per_jet:.0f} us/jet, {1.0 / per_jet:.0f} jets/s; "
          f"2.0M jets ~ {2.0e6 * per_jet / 60.0:.1f} min on one core)")
    if len(np.unique(y)) == 2:
        for name, score in (("M_jet", obs["mass"]),
                            ("-|M_jet - 172.5|", -np.abs(obs["mass"] - 172.5)),
                            ("-tau32", -obs["tau32"]), ("-tau21", -obs["tau21"])):
            print(f"   AUC({name}) = {roc_auc_score(y, score):.3f}")
        for name in ("mass", "tau21", "tau32"):
            print(f"   median {name}: signal {np.median(obs[name][y == 1]):.3f}, "
                  f"background {np.median(obs[name][y == 0]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
