# Week-2 task: diagnose why `so33_equivariant` loses OOD generalization

Multi-seed (3) shows the failure is rock-solid reproducible:
OOD AUC = 0.663 +/- 0.001, while `eta_invariants` (same lift, no
SO33Activation) reaches 1.000 +/- 0.000. So the failure must come from
the activation or its wrapper, not from initialisation noise.

## Corrected hypothesis (after re-reading the code)

`SO33Activation` does **not** have a learnable Γ-network -- the connection
is parameterised by 15 fixed coefficients, with no input dependence. So
"Γ drift on covariant features" was the wrong story.

The actual non-equivariance comes from one place: the `bound_input=True`
flag, which divides each input by `1 + ||x||_2` where `||.||_2` is the
**Euclidean** norm. Under an SO(3,3) boost `g`, the Euclidean norm is
**not** invariant -- it scales roughly like `cosh(rapidity)`. So a
boosted input gets a different normalisation factor than the original,
and the activation becomes input-rapidity-dependent.

`EquivariantSO33Classifier` defaults to `bound_input=True` because
the geodesic ODE under the indefinite metric blows up on unbounded
inputs. So at construction time, the architecture quietly trades exact
equivariance for numerical stability. The trade is invisible at training
rapidity (~0.6, modest renormalisation) but ruinous at OOD rapidity
(~2.5, ~6x renormalisation).

`eta_invariants` does not use `SO33Activation` at all -- it builds
invariants directly from the lifted vector. That is why it generalises
perfectly.

## What to measure (week-2 work, in priority order)

### 1. `benchmarks/diagnose_equivariant.py` -- the smoking gun
Trains `so33_equivariant` on the standard boost_ood setup, then measures
the empirical relative equivariance error of the trained activation on
freshly sampled ID-rapidity inputs vs OOD-rapidity inputs:

    err = ||act(g x) - g act(x)||_2 / ||act(x)||_2

**Prediction:** ratio of mean OOD err / mean ID err is `>> 1`
(probably 5-20x). If so, the bound_input hypothesis is confirmed.

Run:
    python -m benchmarks.diagnose_equivariant --seed 0

### 2. `so33_equivariant_unbounded` -- the principled fix
Same architecture but `bound_input=False`. If the geodesic does not
diverge on this small synthetic dataset, the model should train, and
OOD AUC should recover toward 1.000.

Run:
    python -m benchmarks.run_boost_ood --models so33_equivariant_unbounded --seed 0

**Risk:** the integrator may NaN. If so, that itself is a data point
worth reporting: the indefinite-metric ODE genuinely needs bounding,
and the current bound is incompatible with equivariance -- which sets
up a research direction (use an `eta`-invariant bound such as
`1 + |x^T eta x|^(1/2)` instead of `1 + ||x||_2`).

### 3. `so33_equivariant_frozen` -- the control
Same architecture, `freeze_coeffs=True`. This isolates whether **any**
training of the connection is needed -- if frozen still gets 0.663,
then the failure is purely the bound_input renormalisation, not any
training-time interaction.

Run:
    python -m benchmarks.run_boost_ood --models so33_equivariant_frozen --seed 0

## Decision rule

Cross-reference the three new numbers with the existing 0.663 baseline:

| measurement                       | reading                                             | implication                          |
|-----------------------------------|-----------------------------------------------------|--------------------------------------|
| diagnose OOD/ID err ratio >> 1   | yes (predicted)                                     | bound_input is the cause             |
| `unbounded` trains, OOD ~ 1.000  | yes                                                 | principled fix; promote to Table 1   |
| `unbounded` NaNs at training     | yes (also possible)                                 | open question -> Section 5            |
| `frozen` OOD ~ 0.663             | yes (predicted)                                     | learning Γ irrelevant to the failure |
| `frozen` OOD significantly higher | unexpected                                          | re-think; some training effect helps |

If `unbounded` works:
- Add row to Table 1: `so33_equivariant_unbounded` OOD ≈ 1.000.
- Frame in Section 4: "The default `bound_input` flag protects the
  indefinite-metric ODE from divergence at the cost of breaking
  equivariance under non-compact boosts. Disabling it on the controlled
  OOD setup recovers structural equivariance and restores generalisation
  to 1.000. Whether an `eta`-invariant bound can be designed that
  preserves both is an open question we revisit in Section 5."

If `unbounded` NaNs:
- Report all four numbers honestly in Table 1 plus a one-paragraph
  diagnosis with the diagnose-script ratio as evidence.
- The "open question" in Section 5 is then the eta-invariant bound idea.

Either way, the paper has a positive scientific finding from the
diagnostic work: a clean mechanistic explanation for why an "obviously
equivariant" architecture loses generalisation, plus an evidence-based
direction for fixing it.
