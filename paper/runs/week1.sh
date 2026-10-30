#!/usr/bin/env bash
# Reproduction of every experiment in the paper, on the corrected code
# (gradient through the adaptive scale, Architecture-B connection cap,
# divergence guards). Run from the repository root.
#
# Every run writes one JSON per (model, seed) into results/, with the git
# commit that produced it; tag that commit for the tables. Non-default
# options are part of the experiment name, so runs never overwrite each other.
#
# Wall-time estimates (CPU, ~8 cores):
#   boost-OOD 3 seeds ............ ~2 h   (ODE models dominate: ~2-10 min each)
#   HIGGS 8 seeds ................ ~6 h
#   Adult 3 seeds + controls ..... ~2 h
#   top tagging, internal split .. ~1 h
#   top tagging, canonical:
#     eta_invariants ............. ~36 min / seed
#     Architecture B ............. ODE per constituent: ~1-2 days / seed on the
#                                  full 1.2M train jets; use --max-train-samples
#                                  300000 on a CPU (val/test stay full)
#     PFN / EFN .................. ~30 min / seed; LorentzNet several hours / seed
#     jet observables + BDT ...... ~5 min (tau_N: numpy kT clustering, cached)
#     linear EFPs (d <= 5) ........ ~3 core-hours on the 2M jets (use --n-jobs);
#                                   d <= 7 (the EFP paper's setting) ~50 core-hours
#
# Prerequisites:
#   - pip install -e ".[bench]"   (scikit-learn, scipy, matplotlib, download deps)
#   - pip install energyflow       (only for the linear-EFP baseline; it pulls in
#                                   wasserstein, which pins numpy<2 - re-install
#                                   numpy>=2 afterwards if pip downgraded it)
#   - HIGGS CSV cached under data/ (see benchmarks/run_higgs.py for layout).
#   - Adult fetched on-demand by run_neutral.py via fetch_openml.
#   - Top tagging: python -m benchmarks.download_top_tagging --cache-dir data
#     (~2M jets; the canonical run needs ~4 GB of RAM for the train split).

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== boost-OOD (3 seeds) + figures ==="
python -m benchmarks.run_boost_ood --seeds 0-2
python -m benchmarks.figure_boost_ood
python -m benchmarks.figure_equivariance

echo "=== HIGGS (8 seeds; same seed = same split, so differences are paired) ==="
python -m benchmarks.run_higgs --seeds 0-7
python -m benchmarks.stats --experiment higgs --baseline relu_bottleneck

echo "=== Adult (3 seeds) + parameter-matched and regularised controls ==="
python -m benchmarks.run_neutral --seeds 0-2
python -m benchmarks.run_neutral --seeds 0-2 --models tanh_mlp_w24
python -m benchmarks.run_neutral --seeds 0-2 --models relu_mlp,tanh_mlp,gelu_mlp \
    --weight-decay 1e-4 --early-stop-patience 5
python -m benchmarks.stats --experiment neutral --baseline tanh_mlp_w24

echo "=== Top tagging, internal-split ablation ==="
python -m benchmarks.run_top_tagging --representation constituents \
    --models eta_invariants,relu_bottleneck,so33_signature_only,so33_equivariant_eta_bounded \
    --max-samples 100000 --epochs 30 --seeds 0-2

echo "=== Top tagging, canonical Kasieczka protocol ==="
# Pre-flight: norm distribution, connection cap, short training with the cap off/auto.
python -m benchmarks.diagnose_archb_stability --cache-dir data --canonical-splits \
    --max-train-samples 50000
python -m benchmarks.run_top_tagging --representation constituents --canonical-splits \
    --models eta_invariants,relu_bottleneck,pfn,efn --epochs 30 --seeds 0-2 --save-scores
python -m benchmarks.run_top_tagging --representation constituents --canonical-splits \
    --models lorentznet --epochs 30 --seeds 0-2 --save-scores --max-train-samples 300000
python -m benchmarks.run_top_tagging --representation constituents --canonical-splits \
    --models so33_equivariant_eta_bounded,so33_equivariant_unbounded,so33_equivariant \
    --epochs 30 --seeds 0-2 --save-scores --max-train-samples 300000
for s in 0 1 2; do   # observables are clustered once and cached; only the BDT depends on the seed
    python -m benchmarks.run_jet_baselines --canonical-splits --save-scores --seed "$s" \
        --cache-observables data/jet_observables_canonical.npz
done
python -m benchmarks.run_efp_baseline --canonical-splits --save-scores --n-jobs 4 \
    --cache-efps data/efps_canonical_d5.npz
python -m benchmarks.figure_toptag_mass --cache-dir data --canonical-splits \
    --scores "Readout A=results/top_tagging_canonical__eta_invariants__seed0__scores.npz"

echo
echo "Done. Aggregate with:"
echo "    python -m benchmarks.aggregate --out results/paper_tables.md"
