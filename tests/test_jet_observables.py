"""
Jet-observable baselines (benchmarks/jet_observables.py, benchmarks/run_jet_baselines.py).

Verifies:
  • tau_1..tau_3 (beta = 1, R0 = 0.8, exclusive-kT axes: pairwise merging with
    d_ij = min(pT_i^2, pT_j^2) DeltaR_ij^2, E-scheme, no beam distance) equal a
    pure-Python brute-force reference (naive merging loop + the tau formula) on
    27 random small jets (K <= 8, padding in random slots, massless and massive
    constituents, jets straddling phi = +-pi), for any block size and any extra
    padding. A vectorised reference that recomputes the full pairwise matrix
    at every merge (no nearest-neighbour bookkeeping; itself checked against
    the brute force) agrees on 3000 toy jets with K = 32, enough to catch a
    missed neighbour update, which changes only a few jets in a thousand.
    Where the optional ``fastjet`` package is installed, the taus also equal
    those from FastJet's kt_algorithm (R = 1000, E_scheme) exclusive_jets(N).
  • tau_N = 0 for a jet with exactly N (or fewer) constituents, and the ratio
    convention: tau21 / tau32 = 1 when the denominator vanishes.
  • The two-particle jet mass is sqrt(2 E1 E2 (1 - cos theta)).
  • M, pT and the taus are invariant under a common longitudinal boost and a
    common rotation about the beam, also across phi = +-pi (and do change under
    a transverse boost).
  • The rapidity equals 0.5 log((E + pz) / (E - pz)) and stays finite (= eta)
    when float32 rounding puts a massless constituent's E below |pz|; padding
    never produces NaN; a pT = 0 particle along the beam enters the mass but
    not the taus; non-finite real constituents raise.
  • A toy 3-prong jet has a smaller tau32 than a 1-prong jet (and a 2-prong
    jet a smaller tau21).
  • The runner, end to end on --toy-jets 300 and on tiny canonical
    top_tagging_{train,val,test}.npz files: the three JSONs (jet_mass,
    mass_window, bdt_mass_tau32) with the score-based schema, n_params 0 / 0 /
    null, finite AUCs, test accuracy at the val-chosen cut, per-jet scores
    aligned with the test labels, the masses of the leading-K raw-GeV
    constituents, run_top_tagging's experiment-name rule, model names with
    non-default settings, and rows that benchmarks.aggregate renders.
  • --cache-observables reproduces a run exactly and refuses a cache made for
    other jets (leaving the file alone).

Run:
    python -m pytest tests/test_jet_observables.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import math

import numpy as np
import pytest

from benchmarks import jet_observables as jo
from benchmarks import run_jet_baselines as rjb
from benchmarks.datasets import select_leading_constituents
from benchmarks.provenance import experiment_tag
from benchmarks.run_efp_baseline import toy_jet_split
from benchmarks.synthetic import generate_massless_jets

MODELS = ("jet_mass", "mass_window", "bdt_mass_tau32")
SCHEMA_KEYS = {"experiment", "dataset", "model", "family", "seed", "n_params",
               "model_size", "n_train", "n_val", "n_test", "n_features",
               "walltime_sec", "diverged", "train_metrics", "test_metrics",
               "config", "provenance"}
TAU_NAMES = ("tau1", "tau2", "tau3")


def _massless(pt, rap, phi):
    """(n, 4) massless four-momenta (E, px, py, pz) from (pT, y, phi)."""
    pt, rap, phi = (np.asarray(v, dtype=np.float64) for v in (pt, rap, phi))
    return np.column_stack([pt * np.cosh(rap), pt * np.cos(phi),
                            pt * np.sin(phi), pt * np.sinh(rap)])


# ─────────────────────────────────────────────────────────────────────────
# Brute-force reference: naive exclusive-kT merging + the tau formula
# ─────────────────────────────────────────────────────────────────────────

def _rap(p):
    E, _, _, pz = p
    return 0.5 * math.log((E + pz) / (E - pz))


def _phi(p):
    return math.atan2(p[2], p[1])


def _pt(p):
    return math.hypot(p[1], p[2])


def _dr2(p, q):
    dphi = abs(_phi(p) - _phi(q))
    dphi = min(dphi, 2.0 * math.pi - dphi)
    return (_rap(p) - _rap(q)) ** 2 + dphi ** 2


def brute_force_taus(particles, n_max=3, beta=1.0, r0=0.8):
    """tau_1..tau_n_max: merge the pair with the smallest
    min(pT_i^2, pT_j^2) DeltaR_ij^2 (four-vector sum) until one pseudojet is
    left; the pseudojets present when N remain are the N axes."""
    pseudo = [tuple(p) for p in particles]
    axes = {len(pseudo): list(pseudo)}
    while len(pseudo) > 1:
        _, i, j = min((min(_pt(pseudo[i]) ** 2, _pt(pseudo[j]) ** 2)
                       * _dr2(pseudo[i], pseudo[j]), i, j)
                      for i in range(len(pseudo)) for j in range(i + 1, len(pseudo)))
        merged = tuple(x + y for x, y in zip(pseudo[i], pseudo[j]))
        pseudo = [q for k, q in enumerate(pseudo) if k not in (i, j)] + [merged]
        axes[len(pseudo)] = list(pseudo)
    den = sum(_pt(p) for p in particles) * r0 ** beta
    taus = []
    for n in range(1, n_max + 1):
        if len(particles) <= n:
            taus.append(0.0)
            continue
        num = sum(_pt(p) * min(math.sqrt(_dr2(p, a)) for a in axes[n]) ** beta
                  for p in particles)
        taus.append(num / den)
    return taus


def _random_small_jets(k=8, seed=0):
    """27 jets with 0..8 constituents (three of each count) in random slots;
    every third jet straddles phi = +-pi, half of the constituents are massive."""
    rng = np.random.default_rng(seed)
    n_jets = 3 * (k + 1)
    p4, mask = np.zeros((n_jets, k, 4)), np.zeros((n_jets, k))
    for b in range(n_jets):
        n = b % (k + 1)
        slots = rng.permutation(k)[:n]
        rap0 = rng.uniform(-2.0, 2.0)
        phi0 = (math.pi - 0.05) * rng.choice([-1.0, 1.0]) if b % 3 == 0 \
            else rng.uniform(-math.pi, math.pi)
        pt = np.exp(rng.uniform(0.0, 5.0, n))
        rap = rap0 + rng.normal(0.0, 0.4, n)
        phi = phi0 + rng.normal(0.0, 0.4, n)
        mass = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.1, 3.0, n))
        mt = np.sqrt(pt ** 2 + mass ** 2)
        p4[b, slots] = np.column_stack([mt * np.cosh(rap), pt * np.cos(phi),
                                        pt * np.sin(phi), mt * np.sinh(rap)])
        mask[b, slots] = 1.0
    return p4, mask


def test_tau_matches_brute_force_exclusive_kt():
    p4, mask = _random_small_jets()
    tau = jo.nsubjettiness(p4, mask)
    assert tau.shape == (27, 3) and np.isfinite(tau).all()
    for b in range(len(p4)):
        particles = [tuple(p4[b, s]) for s in range(p4.shape[1]) if mask[b, s] > 0]
        np.testing.assert_allclose(tau[b], brute_force_taus(particles),
                                   rtol=1e-10, atol=1e-14)
    n = mask.sum(axis=1)
    assert (tau[n > 3] > 0).all()                       # not trivially zero
    # Blocks of 5 jets, or three extra padding slots, change nothing.
    np.testing.assert_array_equal(jo.nsubjettiness(p4, mask, chunk=5), tau)
    wide = np.concatenate([p4, np.zeros((27, 3, 4))], axis=1)
    np.testing.assert_allclose(
        jo.nsubjettiness(wide, np.concatenate([mask, np.zeros((27, 3))], axis=1)),
        tau, rtol=1e-12, atol=1e-15)
    obs = jo.compute_jet_observables(p4, mask, chunk=4, verbose=False)
    np.testing.assert_allclose(np.column_stack([obs[t] for t in TAU_NAMES]), tau,
                               rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(obs["tau32"], jo.tau_ratio(tau[:, 2], tau[:, 1]))


def full_matrix_taus(p4, mask, n_max=3, beta=1.0, r0=0.8):
    """Vectorised reference WITHOUT nearest-neighbour bookkeeping: every step
    recomputes the whole pairwise kT matrix of every jet and merges its
    smallest entry (O(K^3) per jet, fine for tests)."""
    use = (np.asarray(mask) > 0) & (np.hypot(p4[..., 1], p4[..., 2]) > 0)
    P = np.where(use[..., None], p4, 0.0)
    B, K = use.shape
    rows = np.arange(B)

    def kinematics(Q, live):
        E, px, py, pz = (Q[..., c] for c in range(4))
        with np.errstate(divide="ignore", invalid="ignore"):
            rap = np.where(live, 0.5 * np.log((E + pz) / (E - pz)), 0.0)
        return px * px + py * py, rap, np.arctan2(py, px)

    def dr2(rap1, phi1, rap2, phi2):
        dphi = np.mod(phi1 - phi2 + np.pi, 2.0 * np.pi) - np.pi
        return (rap1 - rap2) ** 2 + dphi ** 2

    pt2, c_rap, c_phi = kinematics(P, use)
    c_pt = np.sqrt(pt2)
    den = c_pt.sum(axis=1) * r0 ** beta
    active, count = use.copy(), use.sum(axis=1)
    tau = np.zeros((B, n_max))
    off_diag = ~np.eye(K, dtype=bool)
    for _ in range(K - 1):
        go = count > 1
        if not go.any():
            break
        pt2, rap, phi = kinematics(P, active)
        d = (np.minimum(pt2[:, :, None], pt2[:, None, :])
             * dr2(rap[:, :, None], phi[:, :, None], rap[:, None, :], phi[:, None, :]))
        d = np.where(active[:, :, None] & active[:, None, :] & off_diag, d, np.inf)
        i, j = np.divmod(d.reshape(B, -1).argmin(axis=1), K)
        r, i, j = rows[go], i[go], j[go]
        P[r, i] += P[r, j]
        P[r, j] = 0.0
        active[r, j] = False
        count[r] -= 1
        pt2, rap, phi = kinematics(P, active)
        for n in range(1, n_max + 1):
            sel = r[count[r] == n]
            if sel.size:
                d2 = dr2(c_rap[sel][:, :, None], c_phi[sel][:, :, None],
                         rap[sel][:, None, :], phi[sel][:, None, :])
                d2 = np.where(active[sel][:, None, :], d2, np.inf).min(axis=2)
                tau[sel, n - 1] = (c_pt[sel] * d2 ** (0.5 * beta)).sum(axis=1) / den[sel]
    return tau


def test_nearest_neighbour_bookkeeping_matches_full_matrix_reference():
    # The full-matrix reference agrees with the brute force on the small jets ...
    p4, mask = _random_small_jets(seed=1)
    ref_small = full_matrix_taus(p4, mask)
    for b in range(len(p4)):
        particles = [tuple(p4[b, s]) for s in range(p4.shape[1]) if mask[b, s] > 0]
        np.testing.assert_allclose(ref_small[b], brute_force_taus(particles),
                                   rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(jo.nsubjettiness(p4, mask), ref_small,
                               rtol=1e-10, atol=1e-14)
    # ... and checks the nearest-neighbour updates on 3000 toy jets (K = 32),
    # where a missed update changes only a few jets in a thousand.
    X, _ = generate_massless_jets(3000, 32, seed=21)
    X = X.numpy()
    mask = np.abs(X).sum(axis=-1) > 0
    np.testing.assert_allclose(jo.nsubjettiness(X, mask), full_matrix_taus(X, mask),
                               rtol=1e-10, atol=1e-14)


def test_taus_match_fastjet_exclusive_kt_axes():
    fj = pytest.importorskip("fastjet")
    X, _ = generate_massless_jets(40, 32, seed=7)
    X = X.numpy()
    mask = np.abs(X).sum(axis=-1) > 0
    tau = jo.nsubjettiness(X, mask)
    jet_def = fj.JetDefinition(fj.kt_algorithm, 1000.0, fj.E_scheme)
    for b in range(len(X)):
        # fastjet.PseudoJet takes (px, py, pz, E)
        parts = [fj.PseudoJet(float(p[1]), float(p[2]), float(p[3]), float(p[0]))
                 for p in X[b][mask[b]]]
        cs = fj.ClusterSequence(parts, jet_def)
        den = 0.8 * sum(p.pt() for p in parts)
        for n in (1, 2, 3):
            axes = cs.exclusive_jets(n)
            ref = sum(p.pt() * min(math.sqrt(p.squared_distance(a)) for a in axes)
                      for p in parts) / den
            assert tau[b, n - 1] == pytest.approx(ref, rel=1e-10)


def test_tau_n_vanishes_with_n_or_fewer_constituents():
    rng = np.random.default_rng(3)
    counts = (0, 1, 2, 3, 4)
    p4, mask = np.zeros((5, 6, 4)), np.zeros((5, 6))
    for b, n in enumerate(counts):
        slots = rng.permutation(6)[:n]
        p4[b, slots] = _massless(rng.uniform(20.0, 200.0, n), rng.normal(0.0, 0.3, n),
                                 rng.normal(1.0, 0.3, n))
        mask[b, slots] = 1.0
    obs = jo.compute_jet_observables(p4, mask, verbose=False)
    assert obs["n_constituents"].tolist() == list(counts)
    for b, n in enumerate(counts):
        for N, name in enumerate(TAU_NAMES, start=1):
            if n <= N:
                assert obs[name][b] == 0.0, (n, name)
            else:
                assert obs[name][b] > 0.0, (n, name)
    # tau_N / tau_{N-1} = 1 when tau_{N-1} = 0, i.e. fewer than N constituents
    assert obs["tau21"].tolist()[:2] == [1.0, 1.0] and obs["tau21"][2] == 0.0
    assert obs["tau32"].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
    assert 0.0 < obs["tau21"][4] and 0.0 < obs["tau32"][4]
    assert obs["mass"][0] == 0.0 and obs["mass"][1] == pytest.approx(0.0, abs=1e-4)


def test_two_particle_mass_is_sqrt_2_e1_e2_one_minus_cos_theta():
    rng = np.random.default_rng(5)
    n = 50
    E1, E2 = rng.uniform(5.0, 500.0, n), rng.uniform(5.0, 500.0, n)
    u1, u2 = (u / np.linalg.norm(u, axis=1, keepdims=True)
              for u in (rng.normal(size=(n, 3)), rng.normal(size=(n, 3))))
    p4, mask = np.zeros((n, 3, 4)), np.zeros((n, 3))
    p4[:, 0] = np.column_stack([E1, E1[:, None] * u1])
    p4[:, 2] = np.column_stack([E2, E2[:, None] * u2])      # slot 1 is padding
    mask[:, [0, 2]] = 1.0
    mass, pt = jo.jet_mass_pt(p4, mask)
    cos_theta = (u1 * u2).sum(axis=1)
    np.testing.assert_allclose(mass, np.sqrt(2.0 * E1 * E2 * (1.0 - cos_theta)),
                               rtol=1e-9)
    total = E1[:, None] * u1 + E2[:, None] * u2
    np.testing.assert_allclose(pt, np.hypot(total[:, 0], total[:, 1]), rtol=1e-12)


def _boost(P, rapidity, axis):
    out = P.copy()
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    out[..., 0] = ch * P[..., 0] + sh * P[..., axis]
    out[..., axis] = sh * P[..., 0] + ch * P[..., axis]
    return out


def _rotate_z(P, angle):
    out = P.copy()
    c, s = math.cos(angle), math.sin(angle)
    out[..., 1] = c * P[..., 1] - s * P[..., 2]
    out[..., 2] = s * P[..., 1] + c * P[..., 2]
    return out


def test_observables_invariant_under_longitudinal_boosts_and_rotations():
    X, _ = generate_massless_jets(30, 32, seed=11)
    rng = np.random.default_rng(2)
    straddle = np.zeros((1, 32, 4))          # a jet across phi = +-pi
    straddle[0, :20] = _massless(rng.uniform(5.0, 80.0, 20), rng.normal(0.4, 0.3, 20),
                                 math.pi - 0.02 + rng.normal(0.0, 0.3, 20))
    p4 = np.concatenate([X.numpy(), straddle])
    mask = np.abs(p4).sum(axis=-1) > 0
    phi = np.arctan2(p4[-1, :20, 2], p4[-1, :20, 1])
    assert phi.max() > 2.8 and phi.min() < -2.8           # really straddles
    ref = jo.compute_jet_observables(p4, mask, verbose=False)
    names = ("mass", "pt") + TAU_NAMES + ("tau21", "tau32")
    for transform in (lambda P: _boost(P, 1.3, 3),
                      lambda P: _rotate_z(P, 2.5),          # many jets across +-pi
                      lambda P: _rotate_z(P, math.pi),      # the straddler to phi ~ 0
                      lambda P: _boost(_rotate_z(P, -1.9), -0.8, 3)):
        got = jo.compute_jet_observables(np.where(mask[..., None], transform(p4), 0.0),
                                         mask, verbose=False)
        for name in names:
            np.testing.assert_allclose(got[name], ref[name], rtol=1e-9, atol=1e-12,
                                       err_msg=name)
    moved = jo.compute_jet_observables(_boost(p4, 0.6, 1) * mask[..., None], mask,
                                       verbose=False)
    assert np.max(np.abs(moved["tau2"] / ref["tau2"] - 1.0)) > 1e-2
    np.testing.assert_allclose(moved["mass"], ref["mass"], rtol=1e-9)   # M is Lorentz invariant


def test_rapidity_guards_and_degenerate_constituents():
    # E a hair below |pz| (float32 rounding of a massless particle): y = eta.
    p4 = np.array([[[10.0, 1e-3, 0.0, 10.0 + 1e-6], [0.0, 0.0, 0.0, 0.0],
                    [50.0, 10.0, 5.0, -30.0]]])
    pt, rap, phi = jo.constituent_kinematics(p4, np.array([[1.0, 0.0, 1.0]]))
    assert rap[0, 0] == pytest.approx(np.arcsinh((10.0 + 1e-6) / 1e-3))
    assert (pt[0, 1], rap[0, 1], phi[0, 1]) == (0.0, 0.0, 0.0)      # padding
    assert rap[0, 2] == pytest.approx(0.5 * math.log((50.0 - 30.0) / (50.0 + 30.0)),
                                      rel=1e-14)
    assert phi[0, 2] == pytest.approx(math.atan2(5.0, 10.0))
    # A pT = 0 particle along the beam enters the jet mass, not the taus.
    rng = np.random.default_rng(4)
    base = np.zeros((1, 6, 4))
    base[0, :4] = _massless(rng.uniform(20.0, 90.0, 4), rng.normal(0.0, 0.3, 4),
                            rng.normal(0.0, 0.3, 4))
    beam = base.copy()
    beam[0, 5] = [7.0, 0.0, 0.0, 7.0]
    o_base = jo.observables_from_packed(np.concatenate(
        [base, (np.abs(base).sum(-1, keepdims=True) > 0) * 1.0], axis=-1), verbose=False)
    o_beam = jo.compute_jet_observables(beam, np.abs(beam).sum(-1) > 0, verbose=False)
    for name in TAU_NAMES:
        assert o_beam[name][0] == pytest.approx(o_base[name][0], rel=1e-14)
    assert o_beam["n_constituents"][0] == 5 and o_base["n_constituents"][0] == 4
    assert o_beam["mass"][0] > o_base["mass"][0]
    # NaN in a padding slot is ignored; in a real constituent it raises.
    padded_nan = base.copy()
    padded_nan[0, 4] = np.nan
    mask = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    o = jo.compute_jet_observables(padded_nan, mask, verbose=False)
    assert all(np.isfinite(o[k]).all() for k in jo.OBSERVABLES)
    with pytest.raises(ValueError, match="non-finite"):
        jo.compute_jet_observables(padded_nan, np.ones((1, 6)), verbose=False)


def _prong_jet(prongs, n_per, spread, rng, k=32, rap0=0.3, phi0=2.9):
    """One jet (k, 4) of massless particles around (rap0 + dy, phi0 + dphi) per prong."""
    parts = [_massless(share * rng.uniform(5.0, 60.0, n_per),
                       rap0 + dy + rng.normal(0.0, spread, n_per),
                       phi0 + dphi + rng.normal(0.0, spread, n_per))
             for share, dy, dphi in prongs]
    out = np.zeros((k, 4))
    P = np.concatenate(parts)
    out[:len(P)] = P
    return out


def test_three_prong_jet_has_smaller_tau32_than_one_prong_jet():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        p4 = np.stack([
            _prong_jet([(1.0, 0.0, 0.0), (0.8, 0.45, 0.25), (0.6, -0.2, 0.5)], 8, 0.02, rng),
            _prong_jet([(1.0, 0.0, 0.0), (0.7, 0.5, 0.2)], 12, 0.02, rng),
            _prong_jet([(1.0, 0.0, 0.0)], 24, 0.15, rng)])
        o = jo.compute_jet_observables(p4, np.abs(p4).sum(-1) > 0, verbose=False)
        three, two, one = 0, 1, 2
        assert o["tau32"][three] < 0.25 < 0.5 < o["tau32"][one]
        assert o["tau21"][two] < 0.25 < 0.5 < o["tau21"][one]
        assert o["tau32"][three] < o["tau32"][two]


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

def _records(results_dir, experiment, seed=0):
    return {m: json.loads((results_dir / f"{experiment}__{m}__seed{seed}.json").read_text())
            for m in MODELS}


def _check_records(recs, experiment, sizes, min_auc=0.75):
    for model, r in recs.items():
        assert SCHEMA_KEYS <= set(r)
        assert r["experiment"] == experiment and r["model"] == model
        assert r["family"] == "jet_observable" and r["diverged"] is False
        assert (r["n_train"], r["n_val"], r["n_test"]) == sizes
        assert set(r["train_metrics"]) == {"final_val_acc", "val_auc"}
        assert set(r["test_metrics"]) == {"test_acc", "test_auc", "bg_rej_30", "bg_rej_50"}
        tm = r["test_metrics"]
        assert math.isfinite(tm["test_auc"]) and tm["test_auc"] > min_auc
        assert math.isfinite(r["train_metrics"]["val_auc"])
        assert 0.5 < tm["test_acc"] <= 1.0 and 0.5 < r["train_metrics"]["final_val_acc"] <= 1.0
        assert tm["bg_rej_30"] >= tm["bg_rej_50"] >= 1.0
        assert isinstance(r["model_size"], str) and r["model_size"]
        assert r["config"]["normalize"] == "none" and r["provenance"]["argv"]
        assert r["walltime_sec"] >= 0.0
    assert [recs[m]["n_params"] for m in MODELS] == [0, 0, None]
    assert [recs[m]["n_features"] for m in MODELS] == [1, 1, 2]
    bdt = recs["bdt_mass_tau32"]["bdt"]
    assert 1 <= bdt["n_iter"] <= 500 and f"{bdt['n_iter']} trees" in \
        recs["bdt_mass_tau32"]["model_size"]


def _check_scores(results_dir, experiment, recs, y_test, seed=0):
    from sklearn.metrics import roc_auc_score
    for model, r in recs.items():
        z = np.load(results_dir / f"{experiment}__{model}__seed{seed}__scores.npz")
        assert z["scores"].dtype == np.float32 and z["labels"].dtype == np.int8
        assert np.array_equal(z["labels"], np.asarray(y_test))   # same jets, same order
        s = z["scores"].astype(np.float64)
        assert roc_auc_score(z["labels"], s) == pytest.approx(
            r["test_metrics"]["test_auc"], abs=1e-6)
        # test accuracy at the cut chosen on val, not re-optimised on test
        acc = ((s > r["score"]["val_threshold"]) == z["labels"]).mean()
        assert acc == pytest.approx(r["test_metrics"]["test_acc"])


def test_runner_end_to_end_on_toy_jets(tmp_path):
    assert rjb.main(["--toy-jets", "300", "--save-scores",
                     "--results-dir", str(tmp_path)]) == 0
    recs = _records(tmp_path, "toy_jets")
    _check_records(recs, "toy_jets", (210, 45, 45))
    split = toy_jet_split(300, 32, seed=0)
    _check_scores(tmp_path, "toy_jets", recs, split.y_test)
    # The zero-parameter scores are M and -|M - 172.5| of the raw-GeV test jets.
    Xt = split.X_test.numpy()
    mass, _ = jo.jet_mass_pt(Xt[..., :4], Xt[..., 4])
    z = np.load(tmp_path / "toy_jets__jet_mass__seed0__scores.npz")
    np.testing.assert_array_equal(z["scores"], mass.astype(np.float32))
    z = np.load(tmp_path / "toy_jets__mass_window__seed0__scores.npz")
    np.testing.assert_array_equal(z["scores"], (-np.abs(mass - 172.5)).astype(np.float32))
    assert recs["bdt_mass_tau32"]["config"]["bdt"]["random_state"] == 0

    from benchmarks.aggregate import group_by_experiment_model, render_tabular
    md = render_tabular(group_by_experiment_model(list(recs.values())))
    assert "jet observables (no constituent network)" in md
    rows = {line.split("|")[1].strip(): line for line in md.splitlines()
            if line.startswith("| ") and "seeds" not in line}
    assert set(rows) == set(MODELS)
    assert rows["jet_mass"].split("|")[3].strip() == "0"
    assert rows["bdt_mass_tau32"].split("|")[3].strip() == "—"


def _write_kasieczka_file(path, n, seed):
    """Toy massless jets in the layout of top_tagging_<split>.npz."""
    X, y = generate_massless_jets(n, n_constituents=64, seed=seed)
    cons = np.zeros((n, 200, 4), dtype=np.float32)
    cons[:, :64] = X.numpy()
    np.savez(path, constituents=cons, labels=y.numpy())


def test_runner_on_tiny_canonical_files_and_observable_cache(tmp_path):
    data, res, res2 = tmp_path / "data", tmp_path / "results", tmp_path / "again"
    data.mkdir()
    for name, n, seed in (("train", 240, 10), ("val", 80, 11), ("test", 120, 12)):
        _write_kasieczka_file(data / f"top_tagging_{name}.npz", n, seed)
    cache = tmp_path / "observables.npz"
    common = ["--cache-dir", str(data), "--results-dir", str(res)]

    assert rjb.main(common + ["--canonical-splits", "--save-scores",
                              "--cache-observables", str(cache)]) == 0
    recs = _records(res, "top_tagging_canonical")
    _check_records(recs, "top_tagging_canonical", (240, 80, 120))
    assert recs["jet_mass"]["dataset"] == \
        "top_tagging_constituents(k=32,norm=none,split=canonical)"
    test_file = np.load(data / "top_tagging_test.npz")
    _check_scores(res, "top_tagging_canonical", recs, test_file["labels"])
    assert recs["jet_mass"]["observables"]["cache_hit"] is False and cache.is_file()

    # The cached test masses are those of the leading 32 constituents in GeV.
    lead = np.stack([select_leading_constituents(c, 32)
                     for c in test_file["constituents"]]).astype(np.float64)
    P = lead.sum(axis=1)
    expected = np.sqrt(np.maximum(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(axis=1), 0.0))
    with np.load(cache) as z:
        np.testing.assert_allclose(z["test_mass"], expected, rtol=1e-10)

    # A second run reads the cache and reproduces every number.
    assert rjb.main(["--cache-dir", str(data), "--results-dir", str(res2),
                     "--canonical-splits", "--cache-observables", str(cache)]) == 0
    again = _records(res2, "top_tagging_canonical")
    for m in MODELS:
        assert again[m]["observables"]["cache_hit"] is True
        assert again[m]["test_metrics"] == recs[m]["test_metrics"]
        assert again[m]["train_metrics"] == recs[m]["train_metrics"]

    # A cache made for other jets is refused and left alone.
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="different setting.*fingerprint"):
        rjb.main(common + ["--canonical-splits", "--max-train-samples", "100",
                           "--cache-observables", str(cache)])
    assert cache.read_bytes() == before

    # Other data options land in run_top_tagging's experiments.
    assert rjb.main(common + ["--canonical-splits", "--max-train-samples", "100",
                              "--n-constituents", "16"]) == 0
    recs = _records(res, "top_tagging_canonical-k_16-ntrain_100")
    assert (recs["jet_mass"]["n_train"], recs["jet_mass"]["n_test"]) == (100, 120)
    assert rjb.main(common + ["--max-samples", "300", "--seed", "1"]) == 0
    recs = _records(res, "top_tagging_constituents-n_300", seed=1)
    _check_records(recs, "top_tagging_constituents-n_300", (210, 45, 45))


def _args(**kw):
    base = dict(toy_jets=None, canonical_splits=False, max_samples=100_000,
                max_train_samples=None, n_constituents=32, mass_centre=172.5,
                bdt_max_iter=500, bdt_learning_rate=0.1, bdt_max_leaf_nodes=31,
                bdt_min_samples_leaf=20, bdt_validation_fraction=0.1,
                bdt_n_iter_no_change=20)
    return argparse.Namespace(**{**base, **kw})


def test_experiment_and_model_names_follow_run_top_tagging():
    # run_top_tagging's options at their training defaults, plus the data options.
    train_opts = {"norm": "global", "pool": "mean", "cap": "auto",
                  "epochs": 30, "wd": 0.0, "es": None}

    def top_tagging_name(base, k, extra, extra_default):
        key = "ntrain" if base == "top_tagging_canonical" else "n"
        return experiment_tag(base, {**train_opts, "k": k, key: extra},
                              {**train_opts, "k": 32, key: extra_default})

    for k, ntrain in ((32, None), (16, None), (32, 300_000)):
        assert rjb.experiment_name(_args(canonical_splits=True, n_constituents=k,
                                         max_train_samples=ntrain)) == \
            top_tagging_name("top_tagging_canonical", k, ntrain, None)
    for k, n in ((32, 100_000), (24, 50_000)):
        assert rjb.experiment_name(_args(n_constituents=k, max_samples=n)) == \
            top_tagging_name("top_tagging_constituents", k, n, 100_000)
    assert rjb.experiment_name(_args(toy_jets=300)) == "toy_jets"
    assert rjb.experiment_name(_args(toy_jets=300, n_constituents=16)) == "toy_jets-k_16"

    assert rjb.model_names(_args()) == {"jet_mass": "jet_mass",
                                        "mass_window": "mass_window",
                                        "bdt": "bdt_mass_tau32"}
    names = rjb.model_names(_args(mass_centre=175.0, bdt_max_iter=100,
                                  bdt_max_leaf_nodes=15))
    assert names["mass_window"] == "mass_window-centre_175.0"
    assert names["bdt"] == "bdt_mass_tau32-iter_100-leaves_15"



def _leading_constituents(path, k=32):
    z = np.load(path)
    X = np.stack([select_leading_constituents(c, k) for c in z["constituents"]]).astype(np.float64)
    mask = (np.abs(X).sum(-1) > 0).astype(np.float64)
    return X, mask, z["labels"].astype(np.int64)


def test_runner_protocol_invariants(tmp_path, monkeypatch):
    """Pins the protocol: the accuracy cut is searched on VAL only; the BDT
    is the HistGradientBoosting fit on TRAIN (M, tau32) with random_state =
    --seed (an independent rebuild reproduces the saved test scores and the
    number of trees); val_auc is the val AUC of the same scores."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    data, res = tmp_path / "data", tmp_path / "res"
    data.mkdir()
    for name, n, seed in (("train", 400, 10), ("val", 150, 11), ("test", 200, 12)):
        _write_kasieczka_file(data / f"top_tagging_{name}.npz", n, seed)
    tr, va, te = (_leading_constituents(data / f"top_tagging_{s}.npz")
                  for s in ("train", "val", "test"))

    seen = []
    orig = rjb.best_threshold
    monkeypatch.setattr(rjb, "best_threshold",
                        lambda s, y: (seen.append(np.asarray(y).copy()), orig(s, y))[1])
    seed = 3
    assert rjb.main(["--cache-dir", str(data), "--results-dir", str(res),
                     "--canonical-splits", "--save-scores", "--seed", str(seed)]) == 0
    assert len(seen) == 3 and all(np.array_equal(y, va[2]) for y in seen)

    feats = []
    for X, mask, _ in (tr, va, te):
        o = jo.compute_jet_observables(X, mask, verbose=False)
        feats.append(np.column_stack([o["mass"], o["tau32"]]))
    clf = HistGradientBoostingClassifier(early_stopping=True, scoring="loss",
                                         random_state=seed, **rjb.BDT_DEFAULTS).fit(feats[0], tr[2])
    z = np.load(res / "top_tagging_canonical__bdt_mass_tau32__seed3__scores.npz")
    np.testing.assert_allclose(z["scores"], clf.decision_function(feats[2]).astype(np.float32),
                               rtol=1e-6, atol=1e-6)
    recs = _records(res, "top_tagging_canonical", seed=seed)
    bdt = recs["bdt_mass_tau32"]
    assert bdt["config"]["bdt"]["random_state"] == seed
    assert bdt["bdt"]["n_iter"] == clf.n_iter_
    assert bdt["train_metrics"]["val_auc"] == pytest.approx(
        roc_auc_score(va[2], clf.decision_function(feats[1])), abs=1e-9)


def test_experiment_names_match_the_neural_runner(tmp_path):
    """Jet-observable rows must land in the same table as the networks: the
    real run_top_tagging and run_jet_baselines name the same data options alike."""
    from benchmarks.run_top_tagging import main as top_main
    data = tmp_path / "data"
    data.mkdir()
    for name, n, seed in (("train", 200, 10), ("val", 60, 11), ("test", 80, 12)):
        _write_kasieczka_file(data / f"top_tagging_{name}.npz", n, seed)
    for extra in ([], ["--n-constituents", "16"], ["--max-train-samples", "100"]):
        nn_dir, jo_dir = tmp_path / ("nn" + "_".join(extra)), tmp_path / ("jo" + "_".join(extra))
        assert top_main(["--cache-dir", str(data), "--representation", "constituents",
                         "--canonical-splits", "--models", "relu_bottleneck",
                         "--results-dir", str(nn_dir)] + extra) == 0
        assert rjb.main(["--cache-dir", str(data), "--canonical-splits",
                         "--results-dir", str(jo_dir)] + extra) == 0
        nn_exp = {json.loads(q.read_text())["experiment"] for q in nn_dir.glob("*.json")}
        jo_exp = {json.loads(q.read_text())["experiment"] for q in jo_dir.glob("*.json")}
        assert nn_exp == jo_exp, (extra, nn_exp, jo_exp)
