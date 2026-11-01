# Week-1 results — aggregated multi-seed numbers

> **Status (after the referee audit, October 2026).** These are the week-1 numbers,
> written before the corrections listed in the README ("Corrections"). Three readings
> below are superseded, each marked "[Superseded]" where it occurs:
> 1. Architecture B on top tagging did not read out "at chance": its records have test
>    AUC NaN, the ODE diverged (see the correction in `top_tagging_finding.md`).
> 2. The eta rescaling `1/(1+sqrt|x.eta.x|)` is not a bound (identity on the null cone,
>    `so33/activation.py`); on top tagging the flow needs the connection cap
>    (`--connection-cap auto`).
> 3. The Adult "parameter-efficiency" reading of `so33_multi` is not evidence of an
>    inductive bias (README, "Key findings"; controls in `paper/runs/week1.sh`).

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
  SO(3,3) invariance via the readout denominator). With K = 1 the trained
  connection collapses to c ~ 0 and the readout cannot see c, so this tests the
  input map, not the flow.
- `so33_equivariant_eta_bounded` = the principled fix: same Arch B with
  `bound_input="eta"`, dividing by `1 + sqrt(|x . eta . x|)`. This bound
  is itself SO(3,3)-invariant (x . eta . x is the indefinite-metric
  invariant). OOD 1.000 +/- 0.000 across 3 seeds; readout-feature
  invariance verified at machine precision (1.4e-15). [Superseded: the
  rescaling is NOT a bound -- it is the identity on the null cone, so on top
  tagging (massless constituents) the flow diverges without the connection
  cap (`--connection-cap auto`); see the README, "Corrections".]
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
  (~3σ at std ~0.002) with ~10x fewer parameters. [Superseded: this is not
  evidence of an inductive bias. The width-6 models in the table above
  (650-665 parameters) already beat the unregularised width-256 MLPs; the
  controls `tanh_mlp_w24` (2,594 parameters, vs 2,654 for so33_multi) and the
  weight-decay / early-stopping MLPs in `paper/runs/week1.sh` test this.]
- This contradicts the earlier matched-bottleneck framing of the Adult
  story. The win lives at the *natural-width vs natural-width* comparison,
  not matched-parameter.

## Top tagging constituents (100k, K=32, normalize=global, 30 epochs)

| model          | params | test_acc        | test_auc          |
|----------------|--------|-----------------|-------------------|
| eta_invariants | 4802   | 0.896 +/- 0.001 | **0.944 +/- 0.0005** |

(3 seeds; see top_tagging_finding.md.) Old non-invariant baselines on
the same path were ~0.749 (relu) / 0.738 (so33_signature_only), but
those predate a split refactor; on the exact current loader (seed 0) they
are relu_bottleneck 0.759 and so33_signature_only 0.750 (see
top_tagging_finding.md). [Superseded:] so33_equivariant variants got test_acc
0.504 with test AUC NaN: the ODE diverged (non-finite logits; 0.50 is an argmax
over NaN). It was not a chance-level readout, and not a readout lacking
pairwise eta-products (see the correction in top_tagging_finding.md).

CRITICAL caveat: this is a 70/15/15 internal split of 100k jets (the first
100k in sorted file-name order, i.e. of top_tagging_test.npz when all three
files are present), NOT the canonical Kasieczka protocol. Do NOT compare
to LorentzNet / PELICAN published numbers -- they are far ahead on the
real benchmark. See top_tagging_finding.md for the honest framing. The
canonical-protocol run of eta_invariants has since been done (README, "Key
findings"); the baselines under the same protocol (jet observables, EFPs,
PFN/EFN, LorentzNet) are implemented and still have to be run
(`paper/runs/week1.sh`).

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
