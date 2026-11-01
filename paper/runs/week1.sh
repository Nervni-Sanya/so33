#!/usr/bin/env bash
# Reproduction of every multi-seed experiment in the paper, on the corrected
# code (gradient through the adaptive scale, Architecture-B connection cap,
# divergence guards). Run from the repository root. Not included: the
# single-seed diagnostic `python -m benchmarks.diagnose_equivariant --seed 0`
# (paper Appendix B). The LorentzNet beam-frame sensitivity row
# (`lorentznet_lab`) is optional and commented out below.
#
# Every run writes one JSON per (model, seed) into results/, with the git
# commit that produced it; tag that commit for the tables. Non-default
# options are part of the experiment name, so runs never overwrite each other.
#
# Wall-time estimates for the commands as written (one after the other),
# extrapolated from per-step timings on a 4-core 2.8 GHz Xeon, PyTorch CPU:
#   boost-OOD 3 seeds ............ ~2 h   (ODE models dominate: ~2-10 min each)
#   HIGGS 8 seeds ................ ~23 h  (~2.9 h / seed; so33_multi alone ~1.7 h)
#   Adult 3 seeds + controls ..... ~2 h
#   top tagging, internal split .. ~7 h   (3 seeds; per seed Architecture B
#                                          ~1.8 h, so33_signature_only ~0.5 h)
#   top tagging, canonical:
#     eta_invariants ............. ~36 min / seed
#     Architecture B ............. ODE per constituent: ~1.5 days / seed on the
#                                  full 1.2M train jets, so the commands below
#                                  use --max-train-samples 300000 (val/test stay
#                                  full): ~10 h / seed, ~4 days for the three
#                                  variants x 3 seeds
#     PFN / EFN .................. ~30 min / seed
#     LorentzNet (5,088 params) .. ~2 h / epoch on one core for 1.2M jets
#                                  (~60 h / seed; ~24 h on 4 threads); at the
#                                  300000 train jets used below ~0.7 h / epoch,
#                                  ~20 h / seed on one core
#     jet observables + BDT ...... ~5 min (tau_N: numpy kT clustering, cached)
#     linear EFPs (d <= 5) ........ ~3 core-hours on the 2M jets (use --n-jobs);
#                                   d <= 7 (the EFP paper's setting) ~50 core-hours
#
# The SO33 models are small and mostly single-thread bound (HIGGS so33_multi:
# the same step time on 1 and 4 threads; Architecture B: 0.68 s vs 0.39 s per
# step), so independent runs finish much sooner as parallel single-thread
# processes, e.g. for HIGGS (~6 h on 4 cores instead of ~23 h):
#   for s in 0-1 2-3 4-5 6-7; do
#       OMP_NUM_THREADS=1 python -m benchmarks.run_higgs --seeds "$s" &
#   done; wait
# Different seeds write different JSONs. The same holds for the Architecture-B
# variants / seeds and for LorentzNet (three one-thread seeds in parallel), but
# every top-tagging process loads the data itself (~4-6 GB of RAM, see below).
#
# Prerequisites:
#   - pip install -e ".[bench]"   (scikit-learn, scipy, matplotlib, download deps)
#   - pip install energyflow       (only for the linear-EFP baseline; it pulls in
#                                   wasserstein, which pins numpy<2 - re-install
#                                   numpy>=2 afterwards if pip downgraded it)
#   - HIGGS CSV cached under data/ (see benchmarks/run_higgs.py for layout).
#   - Adult fetched on-demand by run_neutral.py via fetch_openml.
#   - Top tagging: python -m benchmarks.download_top_tagging --cache-dir data
#     (~2M jets; loading the 1.2M-jet train split peaks at ~6 GB of RAM, or
#     ~4 GB with --max-train-samples, which still reads the whole train
#     array).

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
# Optional: beam-frame sensitivity row, LorentzNet with the official lab-frame beams
# (another ~20 h / seed on one core).
# python -m benchmarks.run_top_tagging --representation constituents --canonical-splits \
#     --models lorentznet_lab --epochs 30 --seeds 0-2 --save-scores --max-train-samples 300000
python -m benchmarks.run_top_tagging --representation constituents --canonical-splits \
    --models so33_equivariant_eta_bounded,so33_equivariant_unbounded,so33_equivariant \
    --epochs 30 --seeds 0-2 --save-scores --max-train-samples 300000
for s in 0 1 2; do   # observables are clustered once and cached; only the BDT depends on the seed
    python -m benchmarks.run_jet_baselines --canonical-splits --save-scores --seed "$s" \
        --cache-observables data/jet_observables_canonical.npz
done
python -m benchmarks.run_efp_baseline --canonical-splits --save-scores --n-jobs 4 \
    --cache-efps data/efps_canonical_d5.npz          # ~8-9 GB peak RAM (svd LDA fit)
python -m benchmarks.figure_toptag_mass --cache-dir data --canonical-splits \
    --scores "Readout A=results/top_tagging_canonical__eta_invariants__seed0__scores.npz"

echo "=== Canonical protocol with 300k training jets: one table with Architecture B and LorentzNet ==="
# Architecture B and LorentzNet above are trained on 300k jets (experiment
# top_tagging_canonical-ntrain_300000); run the cheaper models there too, so the
# aggregate puts every model in the same table (val and test stay full).
python -m benchmarks.run_top_tagging --representation constituents --canonical-splits \
    --models eta_invariants,relu_bottleneck,pfn,efn --epochs 30 --seeds 0-2 --save-scores \
    --max-train-samples 300000
for s in 0 1 2; do
    python -m benchmarks.run_jet_baselines --canonical-splits --max-train-samples 300000 \
        --save-scores --seed "$s" --cache-observables data/jet_observables_canonical_300k.npz
done
python -m benchmarks.run_efp_baseline --canonical-splits --max-train-samples 300000 \
    --save-scores --n-jobs 4 --cache-efps data/efps_canonical_300k_d5.npz

echo
echo "Done. Aggregate with:"
echo "    python -m benchmarks.aggregate --out results/paper_tables.md"
