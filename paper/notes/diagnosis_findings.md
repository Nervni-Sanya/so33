# Week-2 findings -- diagnosis of `so33_equivariant` OOD failure

Three measurements run; bound_input hypothesis CONFIRMED via the
direct ablation, REFINED by the diagnose-script results.

## Headline results (seed 0)

| variant                          | params | id_auc | ood_auc | gap     |
|----------------------------------|--------|--------|---------|---------|
| so33_equivariant (default)       | 542    | 0.999  | 0.663   | +0.337  |
| so33_equivariant_frozen          | 482    | 0.999  | 0.662   | +0.337  |
| **so33_equivariant_unbounded**   | 542    | 1.000 +/- 0.000 | **1.000 +/- 0.000** | +0.000 |
| eta_invariants (reference)       | 4802   | 1.000  | 1.000   | +0.000  |

Disabling `bound_input` is the entire fix. Architecture B now matches
Architecture A on the headline OOD task. The default's failure was
purely architectural, not a fundamental limitation of the equivariant
lift + ODE approach.

## Diagnose-script numbers (seed 0)

```
mean ||coeffs||                 : 0.0000  (coeffs collapsed to identity)
mean Euclidean ||x||  ID        : 2.589
mean Euclidean ||x||  OOD       : 8.450   (3.26x larger)
rel-eq err on ID inputs         : mean 0.766  max 1.627
rel-eq err on OOD inputs        : mean 0.719  max 1.422
OOD/ID ratio of mean err        : 0.94x
```

## Refined mechanism (the diagnose-script prediction was wrong)

I predicted OOD/ID ratio of activation-level equivariance error would
be `>> 1`. The actual ratio is ~1. The activation-level error is
large at BOTH distributions, not selectively at OOD. So the
mechanism is more subtle:

1. During training, the connection coefficients `c_k` collapse to
   ~0 (mean ||coeffs|| = 0.0000). The activation degenerates to the
   bound_input map alone: `act(x) ≈ x / (1 + ||x||_2)`. The model
   does not use the geodesic flow at all -- it relies on the
   η-invariant readout downstream.
2. The η-invariant readout computes `m² = (act(x)·η·act(x))` per
   particle. With the degenerate activation, this expands to:
   `m² = (x·η·x) / (1 + ||x||_2)²`. The numerator is the genuine
   SO(3,3) invariant -- exactly what a Lorentz-equivariant classifier
   needs. The denominator is NOT SO(3,3)-invariant: `||x||_2` is a
   Euclidean norm, which under a boost g scales like `cosh(rapidity)`.
3. At training rapidity (~0.6), `||x||_2 ~ 2.6`, denominator ~13.
   The model learns a decision boundary in the feature space
   `m² ∈ (invariant_x) / 13ish`. This generalises within the training
   rapidity distribution.
4. At OOD rapidity (~2.5), `||x||_2 ~ 8.5`, denominator ~90 -- a
   ~7x feature-distribution shift. The boundary learned at training
   rapidity does not apply, and OOD classification fails.

So the mechanism is **feature-distribution shift induced by a
non-invariant scalar in the denominator of the readout feature**,
not activation-level equivariance violation. The unbounded variant
removes the denominator: `act(x) = identity` (since coeffs ≈ 0), the
readout sees `m² = (x·η·x)` exactly, which is genuinely invariant,
and OOD generalises perfectly.

## What this means for the paper

### Abstract & headline

Change the framing from "open question about learned Γ drift" to
"identified architectural flaw and provided a one-line fix that
restores OOD generalization to 1.000". This is a positive scientific
contribution, not a negative result.

### Table 1 (boost-OOD headline)

Add the `so33_equivariant_unbounded` row (currently 1.000 OOD on
seed 0; needs seeds 1, 2 to confirm error bar). Keep the
`so33_equivariant` and `so33_equivariant_frozen` rows as a story:
they show that the failure mode is real and reproducible, and that
the single-line fix resolves it.

### Section 5 (Discussion)

The discussion of the failure becomes a positive mechanistic
explanation:
- Why bound_input was originally added (geodesic divergence under
  indefinite metric).
- Why it silently breaks SO(3,3) invariance (Euclidean norm not
  invariant under boosts).
- How it propagates into the η-invariant readout (denominator of
  the invariant feature).
- Why removing it works on the controlled synthetic task (inputs
  are bounded anyway at train rapidity 0.6).
- Open follow-up: design an η-invariant bound such as
  `1 + sqrt(|x·η·x|)` for real-data settings where unbounded
  geodesics diverge. Test on top-tagging constituents (where the
  original divergence motivated bound_input in the first place).

### Section 4.3 (HIGGS)

The flat-tabular path uses `max_input_norm` (soft cap) instead of
`bound_input` (everywhere scaling), so it does not have the same
non-invariance issue. The HIGGS results are unaffected.

## Multi-seed confirmed (3/3 seeds 1.000)

Seeds 0, 1, 2 all returned OOD AUC = 1.000 for
`so33_equivariant_unbounded`. The new headline row has std = 0.000
to printed precision -- as solid as `eta_invariants`.

## Pending: η-invariant bound prototype (optional)

If we want to take the paper from "fixed on synthetic, open on real
data" to "fixed everywhere", a small follow-up experiment:

1. Add a `bound_input_mode` flag to SO33Activation: `"euclidean"`
   (current default), `"eta"` (new). The eta mode divides by
   `1 + sqrt(|x·η·x| + eps)`, which IS SO(3,3) invariant.
2. Verify it does not diverge on top-tagging constituents (the
   original motivation for bound_input).
3. Add to Table 1 as `so33_equivariant_eta_bounded`. If it works
   on both synthetic AND keeps top-tagging stable, that becomes
   the recommended default.

This is roughly 1-2 hours of code + 30 minutes of training. Worth
doing for the paper but not blocking.
