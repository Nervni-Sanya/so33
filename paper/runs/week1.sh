#!/usr/bin/env bash
# Week-1 reproduction: seeds x 3 for the headline boost-OOD experiment and the
# HIGGS matched-bottleneck setting. Run from the repository root.
#
# Wall-time estimate (CPU): ~2-3 hours total. Run sequentially in tmux/screen,
# or split across machines. Each invocation writes its own per-seed JSON into
# results/, which paper/notes/equivariant_diagnosis.md and `aggregate.py`
# pick up automatically.
#
# Prerequisites:
#   - HIGGS CSV cached under data/ (see benchmarks/run_higgs.py for layout).
#   - Adult fetched on-demand by run_neutral.py via fetch_openml.

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

echo
echo "Done. Aggregate with:"
echo "    python -m benchmarks.aggregate"
