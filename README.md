# SO33 and SO3C — geodesic-flow layers with Lorentz-group structure

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0--beta.3-orange.svg)]()
[![DOI](https://zenodo.org/badge/1220231738.svg)](https://doi.org/10.5281/zenodo.19763338)

This repository holds two related neural-network architectures and the
benchmark harness they share.

- **SO33** ([`so33/`](so33/)) — an activation defined as the time-`T` solution
  of a geodesic-like ODE on the pseudo-Euclidean space ℝ³˒³ (signature
  +,+,+,−,−,−) with a learnable `so(3,3)` connection, with the source of an
  accompanying preprint in [`paper/`](paper/).
- **SO3C** ([`so3c/`](so3c/), [`benchmarks/so3c_models.py`](benchmarks/so3c_models.py))
  — the complexified rotation algebra so(3) ⊕ i·so(3) = so(3,ℂ) ≅ so(3,1):
  flows on ℂ³ evaluated in closed form, and Lorentz-invariant jet taggers built
  on a bivector lift of the constituent 4-momenta. Developed on the
  `feature/so3c-complexification` branch; full documentation in
  [`docs/so3c/`](docs/so3c/).

The two share the η metric — Re(z·z) in SO3C is exactly the η-invariant of
SO33 — and the training harness, but they are separate architectures: an SO3C
model is not a drop-in replacement for an SO33 one.

## What's here

| Path | Contents |
|---|---|
| [`so33/`](so33/) | `SO33Activation` (the geodesic ODE layer), the 15-generator `so(3,3)` basis, `SO33Network`, `BottleneckClassifier` |
| [`so3c/`](so3c/) | the `so(3,ℂ)` algebra and closed-form exponential, `HermitianMetric`, `SO3CActivation`, `SO3CInteraction`, the bivector lift |
| [`benchmarks/`](benchmarks/) | data loaders, the `build_model` registry, the training loop, experiment runners, figure scripts, Kaggle tooling |
| [`tests/`](tests/) | 67 tests: 14 for SO33, 38 for SO3C, 15 for the training harness |
| [`docs/so3c/`](docs/so3c/) | SO3C theory, models, API reference, experiments and results |
| [`notebooks/`](notebooks/) | generated Kaggle notebooks for the GPU runs, and how to run them |
| [`paper/`](paper/) | the preprint source (`main.tex`) and figures |
| `results_*/` | committed result files (`results_boost`, `results_fixed`, `results_higgs_ablation`, `results_init`, `results_kappa`, `results_message`, `results_scaling`); `results/` itself is git-ignored |

## SO3C

A state is a complex 3-vector z ∈ ℂ³, stored as a real 6-vector. A learned
connection a ∈ ℂ³ ≅ so(3,ℂ) drives the flow ż = −[a]× z, which conserves the
complex bilinear invariant z·z = Σᵢ zᵢ² and is computed exactly with a complex
Rodrigues formula. Jet constituents enter ℂ³ as bivectors
z_a = (E_a P⃗ − E_P p⃗_a) + i (p⃗_a × P⃗) with the jet momentum P, and a Lorentz
transformation of the jet acts on them as z ↦ Qz with Q ∈ SO(3,ℂ).

Conserving z·z does not make a flow equivariant. A connection computed from
invariants does not transform with the input, so the models built that way
lose their invariance as soon as training moves the connection away from zero.
An equivariant flow needs a covariant connection assembled from several
particles, z_a × Σ_b φ(invariants) z_b
([theory](docs/so3c/theory.md#5-conservation-is-not-equivariance)).

| Model (`--models` name) | Params | Lorentz-invariant logits |
|---|---:|---|
| `so3c_invariant_set` | 5,122 | yes, by construction (no flow) |
| `so3c_covariant_set` | 9,078 | yes, exactly (one closed-form covariant step) |
| `so3c_message_set` | 13,862 | yes, exactly (covariant message passing with a scalar channel) |
| `so3c_interaction_set` | 5,652 | yes, to ODE tolerance |
| `so3c_equivariant_set` | 9,056 | **no**, once trained — kept to reproduce results measured on it |

### Results on the canonical top-tagging benchmark

Kasieczka et al. reference data set with its published split (1.21M training,
404k test jets); K is the number of leading constituents per jet and 1/ε_B the
background rejection at signal efficiency 0.3.

| Model | Params | K | Seeds | AUC | 1/ε_B |
|---|---:|---:|---:|---:|---:|
| `so3c_message_set` | 13,862 | 64 | 2 | 0.9807 ± 0.0001 | 850 ± 56 |
| `so3c_covariant_set` | 9,078 | 64 | 2 | 0.9772 ± <0.0001 | 638 ± 1 |
| `so3c_covariant_set` | 9,078 | 32 | 3 | 0.9746 ± 0.0001 | 312 ± 19 |
| `eta_invariants` (SO33) | 4,802 | 32 | 3 | 0.9478 ± <0.0001 | 49 ± 0 |
| generic Deep Sets MLP, parameter-matched | 9,053 | 32 | 1 | 0.7636 | 9 |
| PELICAN (published) | 208,000 | — | — | 0.9870 | 2250 ± 75 |
| PELICAN, its own size sweep (published) | 605 | — | — | 0.9823 | 901 ± 59 |

On test jets pushed through random Lorentz transformations (boost scale up to
3), the invariant taggers' AUC moves by at most 0.0002, while
`so3c_equivariant_set` falls from 0.9424 to 0.4414 ± 0.3284.

> ⚠️ **No state-of-the-art claim.** Published Lorentz-equivariant taggers are
> ahead: PELICAN with 605 parameters already beats the best SO3C model on both
> metrics. A capacity sweep of the message-passing model (up to 6.7× the
> parameters) moved AUC by at most +0.0001, so more size is not the missing
> ingredient. Every result, its source file and the known limitations are in
> [`docs/so3c/experiments.md`](docs/so3c/experiments.md).

### Quick start

```python
import torch
from benchmarks.models import build_model
from so3c import SO3CActivation

# An exactly Lorentz-invariant jet tagger. Input: (B, K, 5) = (E, px, py, pz, mask).
model = build_model("so3c_covariant_set", in_features=4, out_features=2,
                    representation="constituents")
jets = torch.zeros(4, 32, 5, dtype=torch.float64)
jets[..., 1:4] = torch.randn(4, 32, 3, dtype=torch.float64)
jets[..., 0] = jets[..., 1:4].norm(dim=-1)    # massless constituents
jets[..., 4] = 1.0                            # all 32 constituents are real
logits = model(jets)                          # (4, 2)

# The single-state flow on C^3 = R^6, closed form.
act = SO3CActivation()
y = act(torch.randn(8, 6, dtype=torch.float64))   # (8, 6)
```

### Reproducing

```bash
# The published top-tagging split, written to data/top_tagging_{train,val,test}.npz
python -m benchmarks.download_top_tagging --cache-dir data

# Smoke test: three SO3C taggers, one epoch on 1,400 jets
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --models so3c_invariant_set,so3c_covariant_set,so3c_message_set \
    --max-samples 2000 --epochs 1 --results-dir results_smoke

# The canonical split at K = 64 on a GPU
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --n-constituents 64 --epochs 30 --device cuda --dtype float32 \
    --batch-size 256 --eval-chunk-size 1024 --models so3c_covariant_set

# Invariance on boosted jets (float64 by default)
python -m benchmarks.run_boost_robustness --cache-dir data --n 20000 --epochs 8
```

Documentation: [theory](docs/so3c/theory.md) ·
[models](docs/so3c/models.md) · [API](docs/so3c/api.md) ·
[experiments and results](docs/so3c/experiments.md).

## SO33

A neural-network activation defined as the time-`T` solution of a
**geodesic-like ODE on the pseudo-Euclidean space ℝ³˒³** (signature
+,+,+,−,−,−) with a learnable `so(3,3)` connection. Unlike ReLU/Tanh and other
elementwise functions, the layer is a structured nonlinear flow that respects
the indefinite (Lorentz-like) metric, and it can be assembled into
architectures that are exactly or approximately invariant under SO(3,3)
transformations.

### Key findings

The project is a controlled study of *when* an SO(3,3) inductive bias helps. The
honest summary (full numbers and caveats in [`paper/`](paper/)):

| Result | Setting | Takeaway |
|--------|---------|----------|
| **Perfect OOD generalization** | Synthetic boost-OOD, train rapidity ≤0.6, test ≤2.5 | The structurally invariant `eta_invariants` reaches **OOD AUC 1.000 ± 0.000** (3 seeds), while a ReLU MLP reaches 0.944 ± 0.005 and the flat `so33` bottleneck 0.880 ± 0.007. |
| **A diagnosed-and-fixed equivariance bug** | Same task, architecture B | The default `so33_equivariant` collapses to **0.663 ± 0.001** because its Euclidean-norm input bound (added for ODE stability) is not SO(3,3)-invariant; bounding by the η-norm (`so33_equivariant_eta_bounded`) or not bounding at all restores **1.000 ± 0.000**. |
| **Parameter efficiency** | Adult (natural width) | `so33_multi` reaches **AUC 0.912 ± 0.002** with ~2.7k params, above the best 10×-larger MLP (0.905 ± 0.001). |
| **Invariant readout vs baselines** | Top tagging, **canonical Kasieczka protocol** (full 404k test set) | `eta_invariants` (4.8k params, 32 leading constituents) reaches **AUC 0.948** (3 seeds), against 0.755 for a non-invariant ReLU bottleneck trained the same way (1 seed). |

> ⚠️ **No SOTA claim.** On the same canonical protocol, dedicated Lorentz-equivariant
> taggers (LorentzNet, PELICAN) reach AUC ≈ 0.987 with background rejections in the
> thousands — far ahead of us, and we say so plainly in the paper. The contribution
> is the within-weight-class comparison: a tiny invariant readout strongly
> outperforms a non-invariant baseline of the same setup, now measured directly on
> the published test split.

Honest negatives are reported too: on full-width HIGGS, standard MLPs win
(0.805 vs 0.783 AUC), and architecture B is at chance on top tagging (test
accuracy 0.504 and 0.501 over two seeds) because its readout omits the
pairwise η-inner-product term.

### Quick start

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
criterion = torch.nn.CrossEntropyLoss()
yb = torch.randint(0, 2, (8,))
loss = criterion(net(x), yb) + net.regularization_loss()
loss.backward()
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
```

### `SO33Activation` parameters

| Parameter | Type | Default | Description |
|-----------|------|:-------:|-------------|
| `T` | `float` | `1.0` | ODE integration horizon. Smaller `T` → closer to the identity/linear regime. |
| `method` | `str` | `"dopri5"` | ODE solver: `"dopri5"`, `"rk4"`, `"euler"`. The benchmarks use fixed-step `"rk4"`. |
| `adjoint` | `bool` | `True` | Adjoint backprop (O(1) memory). `False` for direct autograd / debugging. |
| `bound_input` | `bool \| str` | `False` | Input bound for ODE stability: `False`/`"none"`, `True`/`"euclidean"` (÷`1+‖x‖₂`, **not** SO(3,3)-invariant), or `"eta"` (÷`1+√\|xᵀηx\|`, invariant by construction). |
| `signature_only` | `bool` | `False` | Restrict to the 6 compact generators (so(3)⊕so(3) Euclidean ablation). |
| `freeze_coeffs` | `bool` | `False` | Freeze the 15 connection coefficients at init (isolates *learning* the connection). |
| `max_input_norm` | `float \| None` | `None` | Soft norm cap (rescales only outliers); used on the flat tabular paths. |
| `rtol` / `atol` | `float` | `1e-4` / `1e-5` | Adaptive-solver tolerances. |
| `reg_coef` | `float` | `1e-3` | Frobenius penalty on the connection tensor (add `regularization_loss()` to the loss). |

### Architectures (in `benchmarks/models.py`)

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

### Reproducing the experiments

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
python -m pytest tests/ -q
```

67 tests. The 14 SO33 tests cover basis construction and the metric-connection
condition, the forward pass, adjoint-vs-direct autograd consistency, Frobenius
regularization, a minimal training step, synthetic causal classification, and
the ablation/dtype variants. The 38 SO3C tests cover the algebra, the
activation, the interaction layer, the lift, the models (including
equivariance with a non-zero connection) and float32 behaviour. The 15 harness
tests cover checkpointing and resume, reproducibility and the figure scripts.
On a CPU-only machine 65 pass and 2 CUDA-only tests are skipped.

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
