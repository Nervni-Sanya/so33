# SO3C documentation

The overview, current results and quick start are in the
[repository README](../../README.md). The working status — candidates under
test, questions closed by measurement, compute budget, infrastructure
pitfalls — is in [`SO3C_STATUS.md`](../../SO3C_STATUS.md). These pages are the
reference.

## Pages

| Page | Read it for |
|---|---|
| [theory.md](theory.md) | The mathematics: realification, the closed-form exponential, what the flow conserves, why conservation is not equivariance, the covariant connection, the bivector lift and its structural zeros, message passing, beams and the symmetry that survives them, the untrained switches, readout features, numerical precision. |
| [models.md](models.md) | Every classifier: pipeline, feature sizes, parameter counts, switches, which symmetry each has and how that was measured. |
| [api.md](api.md) | Reference for the `so3c` package, `benchmarks.so3c_models` and the harness entry points, with runnable examples. |
| [experiments.md](experiments.md) | Data preparation, protocols, commands, every measured result with its provenance, and the known limitations. |

## Where the code is

| Path | Contents |
|---|---|
| [`so3c/algebra.py`](../../so3c/algebra.py) | Generators, realification $\mathbb C^3\cong\mathbb R^6$, the bilinear invariant $z\cdot z$, closed-form `expm_so3c`. |
| [`so3c/metric.py`](../../so3c/metric.py) | `HermitianMetric`: invariants $\to$ connection $a=\rho+i\beta$, with $g=I+i[\beta]_\times$. |
| [`so3c/activation.py`](../../so3c/activation.py) | `SO3CActivation`: single-state flow, closed form or torchdiffeq. |
| [`so3c/interaction.py`](../../so3c/interaction.py) | `SO3CInteraction`: equivariant multi-particle ODE layer. |
| [`so3c/lift.py`](../../so3c/lift.py) | Bivector lift of 4-momenta, Lorentz matrices matched to `expm_so3c`. |
| [`benchmarks/so3c_models.py`](../../benchmarks/so3c_models.py) | Set taggers, flat (tabular) models, synthetic-task heads. |
| [`benchmarks/models.py`](../../benchmarks/models.py) | `build_model` registry: the `--models` names. |
| [`benchmarks/train.py`](../../benchmarks/train.py), [`benchmarks/tabular_runner.py`](../../benchmarks/tabular_runner.py) | Training loop with checkpoint and exact resume; training, evaluation and result files. |
| [`benchmarks/run_top_tagging.py`](../../benchmarks/run_top_tagging.py) | Top-tagging runner (internal and canonical protocols). |
| [`benchmarks/ensemble_scores.py`](../../benchmarks/ensemble_scores.py) | Seed ensembles from the saved per-jet scores. |
| [`benchmarks/run_boost_robustness.py`](../../benchmarks/run_boost_robustness.py) | Re-evaluates a trained tagger on Lorentz-transformed test jets, with the beams fixed or moved along. |
| [`benchmarks/run_so3c_boost_ood.py`](../../benchmarks/run_so3c_boost_ood.py), [`benchmarks/so3c_synthetic.py`](../../benchmarks/so3c_synthetic.py) | Synthetic electromagnetic-invariant task. |
| [`benchmarks/build_notebooks.py`](../../benchmarks/build_notebooks.py), [`benchmarks/kaggle_client.py`](../../benchmarks/kaggle_client.py) | Kaggle notebooks with the code embedded, and a REST client to push and fetch them. |
| [`tests/`](../../tests/) | `test_so3c_*.py`: 47 tests for the algebra, layers, lift, models and float32 behaviour; `test_harness.py`: 17 for training, resume, schedules and the figure scripts. |
