# Week-2 task: diagnose why `so33_equivariant` loses OOD generalization

The activation is approximately equivariant by construction (measured
relative error $\sim 1.3\times 10^{-3}$ on initialization). After training,
`so33_equivariant` collapses to chance on the $5\times$-boost OOD test
while `eta_invariants` -- which uses the same activation -- generalizes
perfectly. The activation is shared, so the regression must come from
the architecture wrapping it.

## Working hypothesis

The $\Gamma_\theta$ network is conditioned on the *full* hidden vector,
not just invariants of it. During training it learns to depend on
covariant components, which destroys the precondition for equivariance
of the activation. The $\eta$-invariant readout (used by
`eta_invariants` but not `so33_equivariant`) accidentally protects the
$\Gamma$ input from co-evolving in a covariant direction.

## Measurements to run (one seed each, ~30 min total)

### 1. Post-training equivariance error
Load the trained `so33_equivariant` from
`results/boost_ood__so33_equivariant__seed0.json` (and the corresponding
state dict -- check `benchmarks/train.py` for save path; add a save call
if it does not already write weights). Re-run the equivariance-error
measurement over a held-out validation batch, sampled at $\eta_{train}$
and at $\eta_{ood}$ separately.

Expected: error at $\eta_{train}$ small (good), error at $\eta_{ood}$
much larger -- showing that the activation has become input-distribution-
dependent rather than truly equivariant.

### 2. Frozen-$\Gamma$ ablation
Re-run `boost_ood` with `freeze_coeffs=True` on the
`so33_equivariant` architecture (already supported by the activation;
check `models.py` for the equivariant model's constructor argument).

Expected: OOD AUC improves significantly, possibly approaching
`eta_invariants`. If it does, the diagnosis is confirmed.

### 3. $\Gamma$-input ablation
Modify the `so33_equivariant` architecture so $\Gamma_\theta$ receives
only the $\eta$-norms of the hidden vector instead of the full vector.
Re-run boost_ood. This is the principled fix.

Expected: OOD AUC matches or exceeds `eta_invariants`. If it does,
this becomes a positive paper result rather than an open question.

## What to report in the paper

- Section 5 (Discussion): the hypothesis above, with measurement (1)
  numbers as evidence.
- Optional Section 4.7 if measurements (2) or (3) work: a fix that
  recovers OOD generalization, with a one-row addition to Table 1.
- If neither (2) nor (3) helps: keep as open question, do not hide it.

## Decision rule

After week-2 work, decide:
- **Both (2) and (3) fail to recover OOD AUC:** keep `so33_equivariant`
  as a documented negative result in Section 5, do not promote.
- **(2) or (3) recovers OOD AUC:** add a "fixed equivariant" row to
  Table 1 and a short paragraph in Section 4.
- **Both work:** prefer (3) for the paper since it is principled.
