# Top-tagging diagnosis (after week-2)

## Smoking-gun result

On a 7k-sample slice of Kasieczka top-tagging constituents (default
loader, normalize=global, K=32 leading constituents):

| model              | params | val_acc | test_acc | test_auc | wall  |
|--------------------|--------|---------|----------|----------|-------|
| **eta_invariants** | 4802   | 0.903   | 0.901    | 0.926    | 2.1s  |

For comparison, the old REPORT.md numbers on the **same path** at 100k
samples / 30 epochs were:
- pointwise relu:        0.749
- so33_signature_only:   0.738
- so33_frozen:           0.720
- so33:                  NaN (boost divergence, since fixed)

The eta-invariants gap is ~ +0.15 absolute AUC. That is too large to
be initialization noise. Mechanism: eta_invariants includes
**pairwise eta-inner products** s_ij = p_i.eta.p_j in its readout (mean,
mean^2, max), and pairwise structure is exactly what discriminates
multi-prong top jets from 1-2-prong QCD jets. The other architectures
(pointwise, so33 in DeepSetsClassifier wrapping) only see per-particle
features and a mean pool -- no pairwise term.

## Why Arch B (so33_equivariant_*) also fails on top-tagging

The week-2 run of so33_equivariant_unbounded AND
so33_equivariant_eta_bounded both got test_acc=0.504 (chance level)
on this benchmark. Cause is the same: EquivariantSO33Classifier's
readout is per-channel (mean m^2, mean |m^2|, jet_inv) -- 3 features
per channel, 4 channels = 12 features. NO pairwise term. Same
expressivity ceiling as the pointwise baselines, except routed
through an ODE that adds no signal because the per-particle invariants
don't discriminate.

The bound-vs-unbounded comparison is uninformative here because both
variants run on the same expressivity-limited readout: |x.eta.x| is
small on globally-normalised constituents (typical m/E ~ 1e-2), so
the eta-bound divisor 1+sqrt(small) ~ 1 doesn't actually bound
anything -- it reduces to the unbounded case. Both architectures hit
the same readout-feature wall.

## What this means for the paper

### Real-data headline strengthens
eta_invariants becomes a real-data positive result, not just a synthetic
boost-OOD demonstration. Subject to confirming on full 100k data,
multi-seed:

    python -m benchmarks.run_top_tagging --representation constituents \
        --models eta_invariants --max-samples 100000 --epochs 30 --seed 0
    # repeat with seeds 1, 2

Expected runtime: ~5-10 min each (no ODE). If AUC stays ~0.90-0.92,
we have a paper-grade win for Arch A on a standard benchmark.

### Arch B real-data story is honestly negative but understood

We can write:

> "Arch B (so33_equivariant) achieves perfect OOD generalisation on
> the synthetic boost-OOD task once the eta-invariant bound is used
> (Section 4.2). On real top-tagging it fails because its per-channel
> readout (mean m^2, jet_inv) lacks the pairwise eta-inner products
> that Arch A uses to capture multi-prong substructure. Enriching the
> Arch B readout with pairwise features -- combining the eta-bounded
> activation with Arch A-style invariants -- is a natural extension
> that we leave to future work."

### Comparison to published top-tagging SOTA

LorentzNet, PELICAN get ~0.94 AUC on the **canonical Kasieczka test
split** (all ~1.6M jets, train on canonical train, evaluate on
canonical test). Our 0.926 is on a 7k slice that is **internally split
70/15/15 of the first 10k Kasieczka train jets**, which is NOT
directly comparable. For an honest preprint claim:

1. Either implement canonical-split loading (a few hours of work).
2. Or report on the within-train-slice methodology and state
   explicitly that the number is not comparable to published
   top-tagging benchmarks, but is comparable to the in-house
   baselines run on the same setup (the +0.15 gap over so33_-
   signature_only).

Option 2 is honest and sufficient for a preprint that is methodological
in scope ("we show the eta-invariant readout strongly outperforms
non-invariant baselines on this benchmark slice").

## Decision rule for week-3 entry

- If 100k multi-seed eta_invariants holds ~0.90+: real-data headline
  upgrade. Update abstract, add Section 4.5 (top tagging) with this
  as the centrepiece.
- If 100k multi-seed eta_invariants drops to ~0.75 (matching the old
  pointwise baseline): the 7k number was the small-data regime. Walk
  it back, report what the full-data number is, no abstract change.

Either way, the methodological contribution (the eta-bound for Arch B,
the boost-OOD synthetic story) is unaffected.
