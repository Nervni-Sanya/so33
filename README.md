# SO33 Activation — A Geodesic ODE Layer with an SO(3,3) Inductive Bias

> Branch note (feature/so3c-complexification): This branch introduces a related but distinct architecture called **SO3C** (a complexified variant). SO3C is a different architecture from SO33 in general — it is not a drop-in replacement. The README below primarily documents the SO33 activation and associated architectures; where SO3C changes behavior or interfaces, consult the branch-specific code and any branch notes in `paper/` or `benchmarks/`.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0--beta.3-orange.svg)]()
[![DOI](https://zenodo.org/badge/1220231738.svg)](https://doi.org/10.5281/zenodo.19763338)

A neural-network activation defined as the time-`T` solution of a **geodesic-like
ODE on the pseudo-Euclidean space ℝ³˒³** (signature +,+,+,−,−,−) with a learnable
`so(3,3)` connection. Unlike ReLU/Tanh and other elementwise functions, the layer
is a structured nonlinear flow that respects the indefinite (Lorentz-like) metric,
and it can be assembled into architectures that are exactly or approximately
invariant under SO(3,3) transformations.

This repository contains the activation, two end-to-end architectures built on it,
a benchmark harness, and the source of an accompanying preprint
(see [`paper/`](paper/)).

## What's here

- **`so33/`** — the core: `SO33Activation` (the geodesic ODE layer), the 15-generator
  `so(3,3)` basis, and convenience wrappers (`SO33Network`, `BottleneckClassifier`).
- **`benchmarks/`** — datasets, models, the training loop, and runnable experiments
  (synthetic boost-OOD, HIGGS, Adult, top-tagging constituents).
- **`paper/`** — the preprint (`main.tex`), figures, and the annotated result notes.
- **`tests/`** — 14 tests covering the basis, forward/adjoint consistency,
  regularization, training, and the ablation/dtype variants.

## Key findings

The project is a controlled study of *when* an SO(3,3) inductive bias helps. The
honest summary (full numbers and caveats in [`paper/`](paper/)):

| Result | Setting | Takeaway |
|--------|---------|----------|
| **Perfect OOD generalization** | Synthetic boost-OOD, train rapidity ≤0.6, test ≤2.5 | The structurally invariant `eta_invariants` reaches **OOD AUC 1.000 ± 0.000** (3 seeds), while a relu [...]
| **A diagnosed-and-fixed equivariance bug** | Same task, architecture B | The default `so33_equivariant` collapses to **0.663 ± 0.001** because a Euclidean-norm input bound (added for ODE stabil[...] 
| **Parameter efficiency** | Adult (natural width) | `so33_multi` reaches **AUC 0.912 ± 0.002** with ~2.7k params, above the best 10×-larger MLP (0.905 ± 0.001). |
| **Invariant readout vs baselines** | Top tagging, **canonical Kasieczka protocol** (full 404k test set) | `eta_invariants` (4.8k params, 32 leading constituents) reaches **AUC 0.948** (identical[...] 

> ⚠️ **No SOTA claim.** On the same canonical protocol, dedicated Lorentz-equivariant
> taggers (LorentzNet, PELICAN) reach AUC ≈ 0.987 with background rejections in the
> thousands — far ahead of us, and we say so plainly in the paper. The contribution
> is the within-weight-class comparison: a tiny invariant readout strongly
> outperforms a non-invariant baseline of the same setup, now measured directly on
> the published test split.

Honest negatives are reported too: on full-width HIGGS, standard MLPs win
(0.805 vs 0.783 AUC), and architecture B is at chance on top tagging because its
readout omits the pairwise η-inner-product term.

## Quick start

```bash
git clone https://github.com/Nervni-Sanya/so33.git
cd so33
pip install -r requirements.txt
pip install -e .
```

```python
import torch
from so33 import SO33Activation, SO33Network

# The geodesic ODE activation on R^{3,3}.
act = SO33Activation(T=0.5, adjoint=True)
x   = torch.randn(8, 6, dtype=torch.float64)
y   = act(x)                       # (8, 6)

# A small classifier built on top of it.
net    = SO33Network(in_features=6, out_features=2, T=0.5)
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
| `T` | `float` | `1.0` | ODE integration horizon. Smaller `T` → closer to the identity/linear regime. |
| `method` | `str` | `"dopri5"` | ODE solver: `"dopri5"`, `"rk4"`, `"euler"`. The benchmarks use fixed-step `"rk4"`. |
| `adjoint` | `bool` | `True` | Adjoint backprop (O(1) memory). `False` for direct autograd / debugging. |
| `bound_input` | `bool \| str` | `False` | Input bound for ODE stability: `False`/`"none"`, `True`/`"euclidean"` (÷`1+‖x‖₂`, **not** SO(3,3)-invariant), or `"eta"` (÷`1+√\|xᵀηx\|`, *[...]*) |
| `signature_only` | `bool` | `False` | Restrict to the 6 compact generators (so(3)⊕so(3) Euclidean ablation). |
| `freeze_coeffs` | `bool` | `False` | Freeze the 15 connection coefficients at init (isolates *learning* the connection). |
| `max_input_norm` | `float \| None` | `None` | Soft norm cap (rescales only outliers); used on the flat tabular paths. |
| `rtol` / `atol` | `float` | `1e-4` / `1e-5` | Adaptive-solver tolerances. |
| `reg_coef` | `float` | `1e-3` | Frobenius penalty on the connection tensor (add `regularization_loss()` to the loss). |

## Architectures (in `benchmarks/models.py`)

- **`eta_invariants`** (Arch A) — parameter-free lift `(E,p)→ℝ³˒³`, then a readout of
  *only* SO(3,3) invariants (per-particle η-norms and **pairwise** η-inner products).
  Exactly invariant by construction; no ODE activation.
- **`so33_equivariant`** (Arch B) — same lift, then the SO33 geodesic activation per
  particle, then an invariant readout. End-to-end invariance is *conditional* on the
  activation's equivariance — which is exactly what the input-bound choice controls.
  Variants: `so33_equivariant_unbounded`, `so33_equivariant_eta_bounded` (the fix),
  `so33_equivariant_frozen` (ablation).
- **`so33`, `so33_multi`, `so33_signature_only`, `so33_frozen`** — Deep Sets / bottleneck
  variants used in the tabular comparisons.

> Note: the SO3C complexified variant introduced on this branch differs from the SO33
> architectures listed above in parameterization and behavior. Treat SO3C as a
> separate architecture when running benchmarks or comparing results.

## Reproducing the experiments

All commands run from the repo root and write per-seed JSON to `results/`;
`python -m benchmarks.aggregate` reduces them to tables. Full pipeline and wall
times are in Appendix B of the paper. Examples:

```bash
# Synthetic boost-OOD (the headline), 3 seeds
for s in 0 1 2; do python -m benchmarks.run_boost_ood --seed $s; done

# Equivariance probe + Figure 1
python -m benchmarks.figure_equivariance

# Top tagging on the canonical Kasieczka split (downloads ~2M jets first:
# python -m benchmarks.download_top_tagging --cache-dir data)
python -m benchmarks.run_top_tagging --representation constituents \
    --canonical-splits --models eta_invariants --epochs 30 --seed 0
```

## Tests

```bash
python -m pytest tests/ -v     # 14 tests
```

Covers basis construction & the metric-connection condition, forward pass,
adjoint-vs-direct autograd consistency, Frobenius regularization, a minimal
training step, synthetic causal classification, and the ablation/dtype variants.

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
