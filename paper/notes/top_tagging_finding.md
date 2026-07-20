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

For comparison, **apples-to-apples** baselines on the EXACT same 100k
loader/split (seed 0, 30 epochs) confirm the gap is real:

| model               | params | test_acc | test_auc |
|---------------------|--------|----------|----------|
| **eta_invariants**  | 4802   | 0.897    | **0.944** |
| relu_bottleneck     | 44     | 0.707    | 0.759    |
| so33_signature_only | 50     | 0.691    | 0.750    |

Gap: **+0.185 AUC** over relu_bottleneck, **+0.194** over
so33_signature_only, all on the identical split. The old REPORT.md
numbers (0.749 / 0.738) reproduce within noise on the current loader,
so the split refactor did NOT make the task easier -- the gap stands.

The eta-invariants advantage comes from the **pairwise eta-inner
products** s_ij = p_i.eta.p_j in its readout (mean, mean^2, max), which
capture the multi-prong substructure that discriminates top jets from
QCD jets. The matched-bottleneck baselines (relu, so33_signature_only)
only see per-particle features through a mean pool -- no pairwise term --
so they plateau at ~0.75 regardless of the activation.

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

### Canonical-split results (FINAL)

The val split turned out to exist in the dl4phys mirror under the name
`validation.parquet` (the alternate-names fix found it), so all three
canonical splits are the published ones: train 1,211,000 / val 403,000
/ test 404,000. No train carve-out was needed. Results on the FULL
canonical test set (K=32, 30 epochs):

| model           | seeds | test_acc | test_auc | 1/eB@0.3 | wall/seed |
|-----------------|-------|----------|----------|----------|-----------|
| eta_invariants  | 0,1,2 | 0.901    | 0.948    | 50/49/50 | ~36 min   |
| relu_bottleneck | 0     | 0.704    | 0.755    | 8        | ~14 min   |

- Canonical AUC (0.948) is slightly ABOVE the internal-split 0.944 --
  17x more training data.
- The +0.193 AUC / 6x rejection gap over the non-invariant baseline
  survives on the canonical protocol. This is now a direct,
  same-protocol comparison, quotable next to published numbers.
- Honest positioning vs SOTA: LorentzNet/PELICAN are at AUC ~0.987 and
  rejection ~2000+; we are far below and say so plainly in Section 4.5.
  The paper's claim is the within-weight-class comparison, not SOTA.
- Published-row numbers in tab:topt-canonical are from memory and
  marked %verify in main.tex -- MUST be checked against the original
  papers before submission.

### Apples-to-apples baseline -- DONE (gap confirmed)

Ran all three on the exact same 100k split (seed 0, 30 epochs):
relu_bottleneck 0.759, so33_signature_only 0.750, eta_invariants
0.944. The +0.185 gap over relu is defensible and not a split
artifact. Section 4.5 can be written.

Optional polish (not blocking): seeds 1,2 for the two baselines to put
error bars on the gap row. Single-seed baselines are fine for a
preprint given the ~0.19 margin dwarfs any plausible seed variance.

## Decision rule for week-3 entry

- If 100k multi-seed eta_invariants holds ~0.90+: real-data headline
  upgrade. Update abstract, add Section 4.5 (top tagging) with this
  as the centrepiece.
- If 100k multi-seed eta_invariants drops to ~0.75 (matching the old
  pointwise baseline): the 7k number was the small-data regime. Walk
  it back, report what the full-data number is, no abstract change.

Either way, the methodological contribution (the eta-bound for Arch B,
the boost-OOD synthetic story) is unaffected.
