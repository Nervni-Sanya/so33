#!/usr/bin/env bash
# Reproduction of every multi-seed experiment in the paper: boost-OOD, HIGGS,
# Adult, and top tagging (internal-split ablation and the canonical Kasieczka
# protocol). Run from the repository root.
#
# Wall-time estimate (CPU): the first three experiments ~2-3 hours total; top
# tagging adds ~30 minutes for the internal split (eta_invariants ~1 min/seed;
# the so33_signature_only baseline is ODE-bound, ~25 min) and ~2 hours for the
# canonical protocol (~36 min per eta_invariants seed, ~14 min for
# relu_bottleneck). Run sequentially in tmux/screen, or split across machines.
# Each invocation writes its own per-seed JSON into results/, which
# `aggregate.py` picks up automatically.
#
# Prerequisites:
#   - pip install -e ".[bench]"   (scikit-learn for AUC / background rejection,
#                                  plus the top-tagging download dependencies)
#   - HIGGS CSV cached under data/ (see benchmarks/run_higgs.py for layout).
#   - Adult fetched on-demand by run_neutral.py via fetch_openml.
#   - Top tagging: python -m benchmarks.download_top_tagging --cache-dir data
#     (~2M jets; the canonical run needs ~4 GB of RAM for the train split).

set -euo pipefail

cd "$(dirname "$0")/../.."

SEEDS=(0 1 2)

echo "=== boost_ood (3 seeds) ==="
for s in "${SEEDS[@]}"; do
    python -m benchmarks.run_boost_ood --seed "$s"
done

echo "=== HIGGS matched-bottleneck (3 seeds, 30 epochs each) ==="
for s in "${SEEDS[@]}"; do
    python -m benchmarks.run_higgs --seed "$s"
done

echo "=== Adult / neutral (3 seeds) ==="
for s in "${SEEDS[@]}"; do
    python -m benchmarks.run_neutral --seed "$s"
done

echo "=== Top tagging, internal-split ablation (3 seeds + baselines) ==="
for s in "${SEEDS[@]}"; do
    python -m benchmarks.run_top_tagging --representation constituents \
        --models eta_invariants --max-samples 100000 --epochs 30 --seed "$s"
done
python -m benchmarks.run_top_tagging --representation constituents \
    --models relu_bottleneck,so33_signature_only --max-samples 100000 \
    --epochs 30 --seed 0

echo "=== Top tagging, canonical Kasieczka protocol ==="
for s in "${SEEDS[@]}"; do
    python -m benchmarks.run_top_tagging --representation constituents \
        --canonical-splits --models eta_invariants --epochs 30 --seed "$s"
done
python -m benchmarks.run_top_tagging --representation constituents \
    --canonical-splits --models relu_bottleneck --epochs 30 --seed 0

echo
echo "Done. Aggregate with:"
echo "    python -m benchmarks.aggregate"
