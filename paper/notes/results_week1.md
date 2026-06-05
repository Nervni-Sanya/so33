# Week-1 results — aggregated multi-seed numbers

3 seeds per row (seeds 0, 1, 2). Means and standard deviations computed by
hand from per-seed runs on the author's Windows box; per-seed JSON files
are in `results/` on that box and should be copied into the repo's
`results/` directory before running `python -m benchmarks.aggregate`.

## boost_ood (rapidity 0.6 -> 2.5)

| model                          | params | id_auc          | ood_auc             | gap     |
|--------------------------------|--------|-----------------|---------------------|---------|
| eta_invariants                 | 4802   | 1.000 +/- 0.000 | **1.000 +/- 0.000** | +0.000  |
| **so33_equivariant_unbounded** | 542    | 1.000 +/- 0.000 | **1.000 +/- 0.000** | +0.000  |
| **so33_equivariant_eta_bounded** | 542  | 1.000 +/- 0.000 | **1.000 +/- 0.000** | +0.000  |
| relu_mlp                       | 2306   | 1.000 +/- 0.000 | 0.944 +/- 0.004     | +0.056  |
| so33_multi                     | 278    | 1.000 +/- 0.000 | 0.888 +/- 0.004     | +0.112  |
| so33                           | 71     | 1.000 +/- 0.000 | 0.880 +/- 0.005     | +0.120  |
| so33_signature_only            | 62     | 1.000 +/- 0.000 | 0.880 +/- 0.006     | +0.120  |
| relu_bottleneck                | 56     | 1.000 +/- 0.000 | 0.876 +/- 0.004     | +0.124  |
| so33_equivariant (default)     | 542    | 0.999 +/- 0.001 | 0.663 +/- 0.001     | +0.336  |
| so33_equivariant_frozen        | 482    | 0.999 (n=1)     | 0.662 (n=1)         | +0.337  |

Notes for the paper:
- **Honest framing:** relu_mlp degrades to 0.944, not chance. The headline
  is *the gap*: 0.000 (eta_invariants) vs 0.056 (relu_mlp). Do not write
  "MLP collapses".
- `so33_equivariant` (default) failure is rock-solid reproducible across
  seeds (std = 0.001). Week-2 diagnosis (see `diagnosis_findings.md`)
  traced it to the non-invariant `bound_input` Euclidean norm.
- `so33_equivariant_unbounded` = the same Arch B with `bound_input=False`.
  Recovers OOD 1.000 +/- 0.000 across 3 seeds, matching `eta_invariants`.
  The +0.337 gap was an architectural flaw (Euclidean-norm bound breaks
  SO(3,3) invariance via the readout denominator), not a limitation of
  the equivariant lift + ODE approach.
- `so33_equivariant_eta_bounded` = the principled fix: same Arch B with
  `bound_input="eta"`, dividing by `1 + sqrt(|x . eta . x|)`. This bound
  is itself SO(3,3)-invariant (x . eta . x is the indefinite-metric
  invariant). OOD 1.000 +/- 0.000 across 3 seeds; readout-feature
  invariance verified at machine precision (1.4e-15). Compared to
  `unbounded`, this variant ALSO bounds the input so the indefinite-
  metric ODE stays stable -- pending top-tagging confirmation that
  this is enough to prevent the geodesic divergence that motivated
  bound_input in the first place.
- `so33_equivariant_frozen` (freeze coeffs): OOD 0.662, identical to the
  default -- confirms the failure is unrelated to *training* the
  connection (the coeffs collapse to ~0 anyway).

## HIGGS, matched bottleneck (hidden=6)

| model               | params | val_acc         | test_acc        | test_auc          |
|---------------------|--------|-----------------|-----------------|-------------------|
| tanh_bottleneck     | 188    | 0.701 +/- 0.002 | 0.701 +/- 0.002 | **0.770 +/- 0.002** |
| so33                | 203    | 0.694 +/- 0.003 | 0.694 +/- 0.001 | 0.763 +/- 0.002   |
| so33_signature_only | 194    | 0.694 +/- 0.000 | 0.693 +/- 0.001 | 0.761 +/- 0.001   |
| gelu_bottleneck     | 188    | 0.690 +/- 0.005 | 0.689 +/- 0.008 | 0.756 +/- 0.008   |
| relu_bottleneck     | 188    | 0.688 +/- 0.005 | 0.689 +/- 0.009 | 0.755 +/- 0.008   |
| so33_frozen         | 188    | 0.675 +/- 0.007 | 0.676 +/- 0.005 | 0.737 +/- 0.007   |

Notes for the paper:
- so33 (0.763 +/- 0.002) > relu_bottleneck (0.755 +/- 0.008) by ~1σ -- a
  small but consistent win at matched parameters. Tanh wins overall (0.770).
- so33_frozen (0.737) vs so33 (0.763): **+0.026 from learning Γ**, std ~0.005
  on both sides. Solid ablation evidence that the learnable connection
  matters, not just the algebraic structure.

## HIGGS, natural-width MLPs

| model      | params | test_auc        |
|------------|--------|-----------------|
| relu_mlp   | 7938   | **0.805 +/- 0.001** |
| tanh_mlp   | 7938   | 0.804 +/- 0.001 |
| gelu_mlp   | 7938   | 0.803 +/- 0.001 |
| so33_multi | 806    | 0.783 +/- 0.003 |

Notes for the paper:
- Honest negative result: at full width, MLPs win by 0.022 AUC (~7-15σ).
- so33_multi uses ~10x fewer parameters but does not recover the gap. Do
  not spin this as a win.

## Adult (neutral), matched bottleneck (hidden=6)

| model               | params | test_auc        |
|---------------------|--------|-----------------|
| so33                | 665    | 0.914 +/- 0.003 |
| so33_signature_only | 656    | 0.913 +/- 0.002 |
| gelu_bottleneck     | 650    | 0.913 +/- 0.001 |
| relu_bottleneck     | 650    | 0.911 +/- 0.002 |
| tanh_bottleneck     | 650    | 0.910 +/- 0.000 |
| so33_frozen         | 650    | 0.909 +/- 0.002 |

Notes for the paper:
- All within ~0.005, essentially tied. **Not a story** -- mention in
  passing or relegate to appendix.

## Adult (neutral), natural width

| model      | params | test_auc        |
|------------|--------|-----------------|
| so33_multi | **2654** | **0.912 +/- 0.002** |
| tanh_mlp   | 27650  | 0.905 +/- 0.001 |
| gelu_mlp   | 27650  | 0.900 +/- 0.001 |
| relu_mlp   | 27650  | 0.899 +/- 0.000 |

Notes for the paper:
- **The real Adult story:** so33_multi beats best MLP by +0.007 AUC
  (~3σ at std ~0.002) with ~10x fewer parameters. This is a paper-grade
  parameter-efficiency claim.
- This contradicts the earlier matched-bottleneck framing of the Adult
  story. The win lives at the *natural-width vs natural-width* comparison,
  not matched-parameter.

## What this changes in the plan

- Abstract has been updated to reflect honest headline numbers
  (commit alongside this file).
- Section 4.2 (boost-OOD): include the per-seed std column.
- Section 4.3 (HIGGS): add `so33 > relu_bottleneck` observation;
  highlight the frozen-vs-learned ablation as ablation table.
- Section 4.4 (Adult): reframe as natural-width comparison, not matched
  bottleneck.
- Section 5 (Discussion): the so33_equivariant failure is reproducible
  (std=0.001), so the diagnosis hypothesis in
  `equivariant_diagnosis.md` must be tested -- not optional.
