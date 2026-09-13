# SO3C — Lorentz-covariant jet tagging with complexified-SO(3) geodesic flows

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Scope.** This branch, `feature/so3c-complexification`, is the SO3C project,
> and this README covers SO3C only. The SO(3,3) activation (`so33/`), its
> preprint (`paper/main.tex`), `REPORT.md`, `examples/` and `CITATION.cff` belong
> to a separate project, whose README is on the
> [`main` branch](https://github.com/Nervni-Sanya/so33/tree/main). The two still
> share this repository and will move to separate ones.

SO3C builds jet taggers on the complexified rotation algebra
so(3) ⊕ i·so(3) = so(3,ℂ) ≅ so(3,1), which is the Lorentz algebra. Each
particle carries a complex 3-vector z ∈ ℂ³. A learned connection a ∈ ℂ³ moves
it along the flow ż = −[a]× z, which conserves the bilinear invariant
z·z = Σᵢ zᵢ² and is evaluated in closed form with a complex Rodrigues formula,
so no ODE solver is involved. Jet constituents enter ℂ³ through a bivector lift
on which a Lorentz transformation acts as z ↦ Qz with Q ∈ SO(3,ℂ).

## Goal and status

The goal is state of the art on the top-tagging reference dataset of
Kasieczka et al. (arXiv:1902.09914) with its published split: 1,211,000
training, 403,000 validation and 404,000 test jets. It has not been reached.

| Model | Parameters | AUC | 1/ε_B at ε_S = 0.3 |
|---|---:|---|---|
| **SO3C**, `so3c_message_set --beams --channels 8`, K = 64, mean of 2 seeds | 22,834 | **0.98333 ± 0.00010** | **1131 ± 22** |
| the same two seeds with their scores averaged | 2 × 22,834 | 0.98373 | 1174 |
| PELICAN | 208,000 | 0.9870 | 2250 ± 75 |
| LorentzNet | 224,000 | 0.9868 | 2195 ± 173 |
| PELICAN, its own size sweep | 11,000 | 0.9858 | 1879 ± 103 |
| PELICAN, its own size sweep | 1,000 | 0.9835 | 1145 ± 74 |

The gap to PELICAN is 0.0037 AUC and a factor of 2.0 in background rejection.
PELICAN's 1,000-parameter model already matches the 22,834-parameter SO3C
model, so what is missing is architecture, not size. The published values were
checked against the source tables
([`paper/figures/literature_reference.csv`](paper/figures/literature_reference.csv),
[`paper/figures/pelican_scaling.csv`](paper/figures/pelican_scaling.csv)). The
working status — candidates under test, questions closed by measurement, the
GPU budget — is in [`SO3C_STATUS.md`](SO3C_STATUS.md).

## How it works

1. **Lift.** A constituent with 4-momentum p_a in a jet with total momentum P
   becomes z_a = (E_a P⃗ − E_P p⃗_a) + i (p⃗_a × P⃗). Transforming every
   4-momentum by a Lorentz matrix Λ maps each z_a to Q z_a, with Q ∈ SO(3,ℂ)
   the matching group element.
2. **A covariant connection.** A connection computed from invariants alone
   conserves z·z but does not turn with the jet, so a model built on it stops
   being invariant as soon as training moves the connection away from zero
   ([theory §5](docs/so3c/theory.md#5-conservation-is-not-equivariance)). SO3C
   uses a_a = z_a × Σ_b φ(invariants) z_b, which transforms as a ↦ Qa, and
   updates z_a ↦ exp(−T [a_a]×) z_a.
3. **Message passing** (`so3c_message_set`): three such rounds over all pairs
   of particles, a real scalar state per particle updated by invariant edge
   messages, and complex channel mixing
   ([theory §10](docs/so3c/theory.md#10-covariant-message-passing-so3c_message_set)).
4. **Beams.** Two beam 4-vectors (1, 0, 0, ±1) join the particle set, as in
   LorentzNet and PELICAN, so the network can use each constituent's lab-frame
   energy and longitudinal momentum
   ([theory §11](docs/so3c/theory.md#11-beams-what-symmetry-survives)).

**Which symmetry the output has.** Without beams the taggers are exactly
Lorentz-invariant. With beams the network is exactly covariant when the jet and
the beams are transformed together; its output changes when only the jet is
boosted or tilted, and stays unchanged only under rotations about the beam
axis. That is the intended physics. Measured in float64 on the headline
configuration: the logits move by 2×10⁻¹⁶ when the beams move with the jet, and
by 3×10⁻² when they stay fixed.

## Models

| `--models` name | Parameters at the defaults | Output symmetry |
|---|---|---|
| `so3c_message_set` | 13,862; 15,038 with `--beams`; 22,834 with `--beams --channels 8` | Lorentz-invariant; with beams, covariant together with the beams and invariant under rotations about the beam axis |
| `so3c_covariant_set` | 9,078 | Lorentz-invariant, exactly |
| `so3c_invariant_set` | 5,122 | Lorentz-invariant by construction; no flow |
| `so3c_interaction_set` | 5,652 | Lorentz-invariant to ODE tolerance |
| `so3c_equivariant_set` | 9,056 | **not invariant once trained**; kept to reproduce the results measured on it |
| `so3c`, `so3c_static`, `so3c_multi` | depends on the input width | none: flat tabular models on a learned embedding |

`so3c_message_set` also has six switches, all off by default and none trained
on the full data yet: `--no-mass-input`, `--no-self-edges`, `--relnorm-edge`,
`--falpha N`, `--vector-channel` and `--pair-latent N`
([models](docs/so3c/models.md#switches-not-yet-trained)).

## Results on the published split

| Model | Parameters | K | Seeds | AUC | 1/ε_B at ε_S = 0.3 |
|---|---:|---:|---:|---|---|
| `eta_invariants`, the SO(3,3) invariant readout, as a baseline | 4,802 | 32 | 3 | 0.9478 | 49 |
| Deep Sets MLP with the same parameter count and no geometry | 9,053 | 32 | 1 | 0.7636 | 8.5 |
| `so3c_covariant_set` | 9,078 | 32 | 3 | 0.9746 ± 0.0001 | 312 ± 19 |
| `so3c_covariant_set` | 9,078 | 64 | 2 | 0.9772 ± <0.0001 | 638 ± 1 |
| `so3c_message_set` | 13,862 | 64 | 2 | 0.9807 ± 0.0001 | 850 ± 56 |
| `so3c_message_set --beams --channels 8` | 22,834 | 64 | 2 | 0.98333 ± 0.00010 | 1131 ± 22 |

K is the number of leading constituents per jet; ± is the standard deviation
over seeds. On test jets pushed through random Lorentz transformations (boost
scale up to 3, float64), the AUC of the invariant models moves by at most
0.0002, and so does that of the beams model when its beams move with the jet,
while `so3c_equivariant_set` falls from 0.9424 to 0.4414 ± 0.3284. Every result,
its source directory and the known limitations are in
[`docs/so3c/experiments.md`](docs/so3c/experiments.md).

## Quick start

```bash
git clone https://github.com/Nervni-Sanya/so33.git
cd so33
git checkout feature/so3c-complexification
pip install -r requirements.txt
pip install -e .
pip install scikit-learn
```

`pip install -e .` installs the `so3c` and `benchmarks` packages (and, until the
split, `so33`); scikit-learn computes AUC and background rejection.

```python
import torch
from benchmarks.models import build_model, count_parameters

# The headline configuration. Input: (B, K, 5) = (E, px, py, pz, mask).
model = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True, "channels": 8})
print(count_parameters(model))                # 22834

jets = torch.zeros(4, 32, 5, dtype=torch.float64)
jets[..., 1:4] = torch.randn(4, 32, 3, dtype=torch.float64)
jets[..., 0] = jets[..., 1:4].norm(dim=-1)    # massless constituents: E = |p|
jets[..., 4] = 1.0                            # all 32 constituents are real
logits = model(jets)
print(logits.shape)                           # torch.Size([4, 2])
```

## Reproducing

```bash
# The top-tagging data with its published split. K = 64 runs need files converted
# with at least 64 constituent slots; the default keeps all 200.
pip install huggingface_hub pandas pyarrow tables
python -m benchmarks.download_top_tagging --cache-dir data
```

```bash
# Smoke test: three taggers, one epoch on 1,400 jets, seconds on a CPU
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --models so3c_invariant_set,so3c_covariant_set,so3c_message_set \
    --max-samples 2000 --epochs 1 --results-dir results_smoke
```

```bash
# The headline configuration, one seed: 10.6 hours on a Kaggle P100 in float32.
# --max-seconds ends a session; rerunning the same command resumes from the checkpoint.
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --epochs 30 --normalize global --seed 0 \
    --device cuda --dtype float32 --batch-size 256 --n-constituents 64 \
    --models so3c_message_set --rounds 3 --beams --channels 8 --eval-chunk-size 1024 \
    --results-dir results_beams_k64 --ckpt-dir ckpt_beams_k64 --resume --max-seconds 30000

# Seed ensemble from the saved per-jet scores, no retraining
python -m benchmarks.ensemble_scores --results-dir results_message/beams_k64
```

```bash
# Symmetry on boosted test jets (float64 by default). With --beams the script also
# records the curve with the beams transformed along with the jet.
python -m benchmarks.run_boost_robustness --cache-dir data --models so3c_message_set \
    --beams --channels 8 --tag _beams_c8 --n 20000 --epochs 8
```

The GPU runs were executed as Kaggle notebooks generated by
`benchmarks/build_notebooks.py`, which embed the code. The workflow is in
[`notebooks/README.md`](notebooks/README.md); the pitfalls met along the way are
in the infrastructure table of [`SO3C_STATUS.md`](SO3C_STATUS.md).

## Documentation

| Page | Contents |
|---|---|
| [docs/so3c/theory.md](docs/so3c/theory.md) | The mathematics: realification, the closed-form exponential, why conservation is not equivariance, the covariant connection, the bivector lift, message passing, beams, the untrained switches, numerical precision |
| [docs/so3c/models.md](docs/so3c/models.md) | Every tagger: pipeline, parameter counts, switches, and how its symmetry was measured |
| [docs/so3c/api.md](docs/so3c/api.md) | The `so3c` package, `benchmarks.so3c_models` and the harness, with runnable examples |
| [docs/so3c/experiments.md](docs/so3c/experiments.md) | Data, protocols, commands, every result with its source, limitations |
| [SO3C_STATUS.md](SO3C_STATUS.md) | Working status: goal, current best, closed questions, open candidates, compute budget, infrastructure pitfalls |
| [so3c_notes/](so3c_notes/) | Research notes, such as the gap-to-SOTA survey of 2026-09-13 |

## Repository layout

| Path | Belongs to | Contents |
|---|---|---|
| [`so3c/`](so3c/) | SO3C | Algebra and closed-form exponential, `HermitianMetric`, `SO3CActivation`, `SO3CInteraction`, the bivector lift |
| [`benchmarks/so3c_models.py`](benchmarks/so3c_models.py) | SO3C | Taggers, flat models and synthetic-task heads |
| [`benchmarks/`](benchmarks/), the other files | shared harness | Data loaders, `build_model`, training loop, runners, `ensemble_scores`, figure scripts, Kaggle tooling; `benchmarks/models.py` imports `so33` for the SO(3,3) models it also registers |
| [`tests/`](tests/) | both | `test_so3c_*.py` (47 tests) and `test_harness.py` (17) for SO3C and the harness; the other files (14 tests) test `so33` |
| [`notebooks/`](notebooks/) | SO3C | Generated Kaggle notebooks with the code embedded |
| `results_message/`, `results_kappa/`, `results_fixed/`, `results_boost/`, `results_scaling/`, `results_init/`, `results_sweep/`, `results_60ep/`, `results_cleanup/`, `results_matched/`, `results_matched_canonical/` | SO3C | Committed results with per-jet test scores |
| `results_higgs_ablation/` | both | HIGGS feature-set ablation with SO3C and SO(3,3) rows |
| [`paper/figures/`](paper/figures/) | SO3C | Figures, their CSV twins and the published reference values, inside the SO(3,3) paper's directory for now |
| `so33/`, `paper/main.tex`, `REPORT.md`, `examples/`, `CITATION.cff` | SO(3,3) | Not described here |

`results/` and `data/` (the datasets, about 11 GB) are in `.gitignore`.

## Tests

```bash
python -m pytest tests/ -q
```

The suite has 78 tests. The 47 SO3C tests cover the algebra (5), the activation
(7), the interaction layer (5), the lift (4), the models (20) and float32
behaviour (6); the 17 harness tests cover checkpointing and resume,
reproducibility, the training schedules and the figure scripts; 14 test the
`so33` package. On a CPU-only machine 76 pass and two CUDA-only harness tests
are skipped. Tests of the flow models' equivariance set the zero-initialised
weight heads to random values first: on a freshly built model the flow is the
identity, and an invariance check would pass without testing the flow.

## License and citation

MIT, see [LICENSE](LICENSE). There is no SO3C paper or DOI yet; `CITATION.cff`
and the Zenodo DOI describe the SO(3,3) software.

## Acknowledgements

The SO3C construction, benchmark harness and experiments were developed with
the assistance of **Claude (Anthropic)**.
