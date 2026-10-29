# SO33 Activation — A Geodesic ODE Layer with an SO(3,3) Inductive Bias

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0--beta.3-orange.svg)]()
[![DOI](https://zenodo.org/badge/1220231738.svg)](https://doi.org/10.5281/zenodo.19763338)

A neural-network activation defined as the time-`T` solution of a **geodesic-like
ODE on the pseudo-Euclidean space ℝ³˒³** (signature +,+,+,−,−,−) with a learnable
`so(3,3)` connection. Unlike ReLU/Tanh and other elementwise functions, the layer
is a structured nonlinear flow that conserves the indefinite (Lorentz-like) norm
⟨v,v⟩_η, and it can be assembled into architectures whose readout is SO(3,3)
invariant.

This repository contains the activation, two end-to-end architectures built on it,
a benchmark harness, and the source of an accompanying preprint
(see [`paper/`](paper/)).

## What's here

- **`so33/`** — the core: `SO33Activation` (the geodesic ODE layer), the 15-generator
  `so(3,3)` basis, and convenience wrappers (`SO33Network`, `BottleneckClassifier`).
- **`benchmarks/`** — datasets, models, the training loop, and runnable experiments
  (synthetic boost-OOD, HIGGS, Adult, top-tagging constituents), the published-tagger
  and jet-observable baselines, statistics and figure scripts.
- **`paper/`** — the preprint (`main.tex`), figures, and the annotated result notes.
- **`tests/`** — 103 tests: the basis, gradients against finite differences at the
  experimental solver settings, conservation vs. equivariance, the input maps, RK4
  convergence order, the stability guarantee, the training/evaluation divergence
  guards, the published parameter counts, the data loaders, metrics, result records
  and statistics.

## Corrections (October 2026)

A referee audit found defects that change how earlier results must be read. They
are fixed in the code; the affected experiments have to be re-run.

- **Gradient bug.** The adaptive scale ω → ω/(1+‖ω‖_F) was evaluated without
  gradient, so training ignored its derivative (autograd vs. finite differences:
  0.3–4% error). It is now differentiable. The *forward* map is unchanged
  (the equivariance CSV reproduces byte for byte); every *trained* SO33 number
  can change slightly.
- **Architecture B on top tagging diverged** (test AUC NaN), it was not "at
  chance". Lifted massless constituents lie on the null cone, where neither the
  η rescaling nor any equivariant input map bounds their norm, and with the old
  cap the ODE overflows once ‖c‖ ≈ 0.1. A provable connection cap now prevents it
  (see *Stability* below).
- **What the top-tagging readout learns.** Constituents are massless, and a
  zero-parameter jet-mass window on the same 32 constituents reaches AUC 0.941;
  `eta_invariants` (0.948) approximately learns that window. The pairwise term is
  not the mechanism.
- **β_η is a rescaling, not a bound**, and the flow is SO(3,3)-equivariant only at
  c = 0. A readout of ⟨σ(x),σ(x)⟩_η cannot see the connection (the flow conserves
  it), so ε_σ — not ε_m — is the equivariance measure (`figure_equivariance`).
- **Result files** of runs that differed only in, e.g., the constituent
  normalisation overwrote each other. Non-default options are now part of the
  experiment name.

## Key findings (to be re-run on the corrected code)

| Result | Setting | Takeaway |
|--------|---------|----------|
| **Boost-OOD sanity check** | Synthetic single particles (K=1), train rapidity ≤0.6, test ≤2.5 | `eta_invariants` reaches OOD AUC 1.000 by construction (it reads m² directly); a relu MLP degrades to 0.944 ± 0.004. |
| **Non-invariant input bound** | Same task, architecture B | With the Euclidean bound ÷(1+‖x‖₂) the readout ⟨β(x),β(x)⟩_η depends on rapidity and OOD AUC falls to 0.663 ± 0.001; the η rescaling — or no bound — gives 1.000. With K=1 the connection gets no gradient (c → 0), so this tests the input map, not the flow. |
| **Top tagging, canonical protocol** | Full 404k test set, 32 leading constituents | `eta_invariants` (4.8k params) AUC 0.948, 1/ε_B ≈ 50; a zero-parameter mass window gives 0.941 / 40. Baselines under the identical protocol: jet mass / mass window / BDT on (M, τ₃₂), linear EFPs, PFN/EFN and LorentzNet at ~5k parameters. |
| **Tabular data** | HIGGS, Adult at width 6 | The activations are statistically indistinguishable on HIGGS at n = 3 (Welch p ≈ 0.27); on Adult the small width-6 models already beat the unregularised width-256 MLPs, so `so33_multi` (0.912) is not evidence of an inductive bias. |

> ⚠️ **No SOTA claim.** Dedicated Lorentz-equivariant taggers (LorentzNet, PELICAN)
> reach AUC ≈ 0.987 with background rejections in the thousands on the same test set.

## Quick start

```bash
git clone https://github.com/Nervni-Sanya/so33.git
cd so33
pip install -r requirements.txt
pip install -e .
```

To run the benchmarks, add the extras (`scikit-learn` and `scipy` are needed for AUC,
background rejection and statistics; the rest is for the top-tagging download):

```bash
pip install -e ".[bench]"
pip install energyflow      # only for the linear-EFP baseline
```

```python
import torch
from so33 import SO33Activation, SO33Network

# The geodesic ODE activation on R^{3,3}, with the integrator used in the experiments.
act = SO33Activation(T=0.3, method="rk4", solver_options={"step_size": 0.03})
x   = torch.randn(8, 6, dtype=torch.float64)
y   = act(x)                       # (8, 6)

# A small classifier built on top of it.
net    = SO33Network(in_features=6, out_features=2, T=0.3, method="rk4")
logits = net(x)                    # (8, 2)
```

During training, add gradient clipping and the activation's regularizer:

```python
loss = criterion(net(xb), yb) + net.regularization_loss()
loss.backward()
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
```

## `SO33Activation` parameters

| Parameter | Type | Default | Description |
|-----------|------|:-------:|-------------|
| `T` | `float` | `1.0` | ODE integration horizon (the experiments use 0.3). |
| `method` | `str` | `"dopri5"` | ODE solver: `"dopri5"`, `"rk4"`, `"euler"`. The experiments use fixed-step `"rk4"`; a fixed-grid method without a `step_size` uses T/10 (it used to take a single step). |
| `adjoint` | `bool` | `True` | Adjoint backprop (O(1) memory). `False` for direct autograd. Both agree with finite differences to ~1e-9. |
| `connection_cap` | `float` | `1.0` | κ in ω_eff = κω/(κ+‖ω‖_F), so ‖ω_eff‖_F < κ. Use `so33.safe_connection_cap(r, T)` for a no-blow-up guarantee on inputs with ‖x‖₂ ≤ r. |
| `bound_input` | `bool \| str` | `False` | Input map: `False`/`"none"`, `True`/`"euclidean"` (÷`1+‖x‖₂`: bounded, **not** SO(3,3)-invariant), or `"eta"` (÷`1+√\|xᵀηx\|`: commutes with SO(3,3) but is the identity on the null cone, i.e. **not** a bound). |
| `signature_only` | `bool` | `False` | Restrict to the 6 compact generators (so(3)⊕so(3) Euclidean ablation; preserves ‖v‖₂, never blows up). |
| `freeze_coeffs` | `bool` | `False` | Freeze the 15 connection coefficients at init. |
| `max_input_norm` | `float \| None` | `None` | Soft norm cap (rescales only outliers); used on the flat tabular paths. |
| `rtol` / `atol` | `float` | `1e-4` / `1e-5` | Adaptive-solver tolerances. |
| `reg_coef` | `float` | `1e-3` | Frobenius penalty on the raw connection (add `regularization_loss()` to the loss). |

**Stability.** Rotations preserve ‖v‖₂; boosts give d‖v‖/dt ≤ 4/(3√6)·‖ω_eff‖_F·‖v‖²
(sharp), so the flow can blow up before T once that factor times ‖x‖·T reaches 1.
`safe_connection_cap(r, T, margin=0.5)` keeps every input with ‖x‖₂ ≤ r below 2r for
every value of the coefficients — a cap on the parameters, so it adds no breaking of
equivariance.

## Architectures (in `benchmarks/models.py`)

- **`eta_invariants`** (Arch A) — parameter-free lift `(E,p)→(p,E,0,0)`, then seven
  pooled SO(3,3) invariants (per-particle η-norms, pairwise η-products, the jet
  total's η-norm) into a 7–64–64–2 ReLU MLP (4,802 parameters). No ODE.
- **`so33_equivariant`** (Arch B) — same lift, four SO33 activation channels per
  particle, then per channel the mean η-norm, mean |η-norm| and the jet total's
  η-norm (= −M²_jet at c = 0) into a 12–32–2 head (542 parameters). Variants:
  `_unbounded`, `_eta_bounded`, `_frozen`. On per-constituent data the runners give
  the β_none/β_η channels the provable connection cap (`--connection-cap auto`).
- **Baselines**: `pfn`, `efn`, `lorentznet` (~5k parameters, same training protocol);
  `benchmarks/run_jet_baselines.py` (jet mass, mass window, BDT on (M, τ₃₂));
  `benchmarks/run_efp_baseline.py` (linear EFPs).
- **`so33`, `so33_multi`, `so33_signature_only`, `so33_frozen`** — Deep Sets / bottleneck
  variants used in the tabular comparisons; `<relu|tanh|gelu>_mlp_w<H>` are MLPs of
  width H (e.g. `tanh_mlp_w24`, the Adult control at the `so33_multi` parameter count).

## Reproducing the experiments

All commands run from the repo root and write one JSON per (model, seed) to
`results/`, including the git commit, library versions and command line, the
training-guard outcome (`diverged`) and, for SO33 models, the trained coefficients.
Commit these JSONs with the code that produced them; per-sample `.npz` files
(`--save-scores`) are git-ignored and belong in the Zenodo archive.
`python -m benchmarks.aggregate` reduces them to tables and
`python -m benchmarks.stats` compares models across seeds (Welch test and paired
95% intervals). The full pipeline, with wall times, is
[`paper/runs/week1.sh`](paper/runs/week1.sh). Examples:

```bash
# Synthetic boost-OOD, 3 seeds, then the figures (AUC vs rapidity, readout distributions)
python -m benchmarks.run_boost_ood --seeds 0-2
python -m benchmarks.figure_boost_ood
python -m benchmarks.figure_equivariance

# Top tagging on the canonical Kasieczka split. Download first (~2M jets):
#   python -m benchmarks.download_top_tagging --cache-dir data
# Architecture B: check stability first (2 minutes), then train.
python -m benchmarks.diagnose_archb_stability --cache-dir data --canonical-splits \
    --max-train-samples 50000
python -m benchmarks.run_top_tagging --representation constituents --canonical-splits \
    --models eta_invariants,so33_equivariant_eta_bounded,pfn,efn --seeds 0-2 --save-scores
python -m benchmarks.run_jet_baselines --canonical-splits --save-scores

# HIGGS with 8 seeds and paired statistics
python -m benchmarks.run_higgs --seeds 0-7
python -m benchmarks.stats --experiment higgs --baseline relu_bottleneck
```

## Tests

```bash
python -m pytest tests/ -v     # 103 tests, ~15 s
```

## Paper

The accompanying preprint is in [`paper/`](paper/) (`paper/main.tex`). Build with:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

*(arXiv link to be added once posted.)*

## Citation

See [`CITATION.cff`](CITATION.cff).

## Acknowledgements

The architecture, benchmark harness, experiments, and preprint were developed with
the assistance of **Claude (Anthropic)**.
