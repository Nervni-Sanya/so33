# Top-tagging diagnosis (after week-2)

## Smoking-gun result -- CONFIRMED at full scale

On 100k Kasieczka top-tagging constituents (70k train / 15k val /
15k test, normalize=global, K=32 leading constituents, 30 epochs):

| model              | seed | val_acc | test_acc | test_auc | wall  |
|--------------------|------|---------|----------|----------|-------|
| eta_invariants     | 0    | 0.895   | 0.897    | 0.944    | 56.1s |
| eta_invariants     | 1    | 0.900   | 0.896    | 0.945    | 56.9s |
| eta_invariants     | 2    | 0.898   | 0.896    | 0.944    | 56.8s |
| **mean +/- std**   |      | 0.898 +/- 0.002 | 0.896 +/- 0.001 | **0.944 +/- 0.0005** | ~57s |

The 7k-slice number (0.926) was if anything an *underestimate*; at
full scale eta_invariants reaches AUC 0.944 with std 0.0005 across
seeds. This is a rock-solid real-data positive result.

For comparison, the old REPORT.md numbers on the **same path** at 100k
samples / 30 epochs were:
- pointwise relu:        0.749
- so33_signature_only:   0.738
- so33_frozen:           0.720
- so33:                  NaN (boost divergence, since fixed)

The eta-invariants gap is ~ +0.19 absolute AUC over the best
non-invariant baseline (0.944 vs 0.749). Mechanism: eta_invariants
includes **pairwise eta-inner products** s_ij = p_i.eta.p_j in its
readout (mean, mean^2, max), and pairwise structure is exactly what
discriminates multi-prong top jets from 1-2-prong QCD jets. The other
architectures (pointwise, so33 in DeepSetsClassifier wrapping) only
see per-particle features and a mean pool -- no pairwise term.

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

### Real-data headline CONFIRMED
eta_invariants is a real-data positive result, not just a synthetic
boost-OOD demonstration: AUC 0.944 +/- 0.0005 across 3 seeds on the
full 100k benchmark. This is the paper's second real-data win
(alongside the Adult parameter-efficiency result), and the strongest
one numerically.

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

### Comparison to published top-tagging SOTA -- HANDLE WITH CARE

This is the one place we must NOT overclaim. Published numbers:
- LorentzNet, PELICAN report **AUC ~0.94** but those are usually
  quoted as accuracy ~0.94 / AUC ~0.985+ on the **canonical Kasieczka
  test split** (all ~1.2M train / 400k test jets, evaluated on the
  held-out canonical test set).
- Our 0.944 AUC is on a **70/15/15 internal split of the first 100k
  Kasieczka jets**, with only K=32 leading constituents and a tiny
  4802-param MLP on 7 pooled invariants.

These are NOT comparable, and our AUC 0.944 vs their AUC ~0.985 means
**we are well below SOTA on the real metric**. The fact that our number
*looks* close is an artifact of different split + metric reporting.

Two things to do before the preprint:
1. **Strongly suspect a too-easy split or a label/feature leak.**
   An 4802-param model reaching 0.944 AUC is plausible for this task
   (jet mass alone gets ~0.90), but verify: (a) the internal split is
   stratified and non-overlapping; (b) no constituent ordering leaks
   the label; (c) compare against the pointwise baseline ON THE SAME
   100k split (the old 0.749 was also 100k but pre-split-refactor --
   re-run relu_bottleneck and so33_signature_only on this exact
   loader to get an apples-to-apples baseline gap).
2. **Report honestly**: "On a 100k-jet subset with an internal split,
   the eta-invariant readout reaches AUC 0.944 +/- 0.0005, far above
   non-invariant baselines on the same subset (relu 0.749). We do not
   evaluate on the canonical test split and do not claim to match
   dedicated Lorentz-equivariant taggers (LorentzNet, PELICAN), which
   operate at higher accuracy on the full benchmark."

### IMMEDIATE next check (apples-to-apples baseline on same loader)

    python -m benchmarks.run_top_tagging --representation constituents \
        --models relu_bottleneck,so33_signature_only,eta_invariants \
        --max-samples 100000 --epochs 30 --seed 0

This puts all three on the EXACT same 100k split so the +0.19 gap claim
is defensible. If relu_bottleneck also jumps to ~0.90 on this split,
the gap shrinks and the story changes -- so this is the single most
important number to get before writing Section 4.5.

## Decision rule for week-3 entry

- If 100k multi-seed eta_invariants holds ~0.90+: real-data headline
  upgrade. Update abstract, add Section 4.5 (top tagging) with this
  as the centrepiece.
- If 100k multi-seed eta_invariants drops to ~0.75 (matching the old
  pointwise baseline): the 7k number was the small-data regime. Walk
  it back, report what the full-data number is, no abstract change.

Either way, the methodological contribution (the eta-bound for Arch B,
the boost-OOD synthetic story) is unaffected.
