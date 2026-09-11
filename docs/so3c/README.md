# SO3C — geodesic flows on the complexified rotation algebra

SO3C is a family of layers and jet taggers built on the complexified rotation
algebra

$$\mathfrak{so}(3)\oplus i\,\mathfrak{so}(3)=\mathfrak{so}(3,\mathbb C)\;\cong\;\mathfrak{so}(3,1)\;\cong\;\mathfrak{sl}(2,\mathbb C)_{\mathbb R},$$

which is the Lorentz algebra. A state is a complex 3-vector $z\in\mathbb C^3$
(stored as a real 6-vector), a learned connection $a\in\mathbb C^3$ drives the
flow $\dot z=-[a]_\times z$, and the flow is evaluated in closed form with a
complex Rodrigues formula rather than an ODE solver. Jet constituents enter
$\mathbb C^3$ through a bivector ("Riemann–Silberstein") lift that is exactly
Lorentz-covariant.

SO3C sits beside the parent SO33 architecture ([`so33/`](../../so33/)) in this
repository. The two share the benchmark harness and the $\eta$ metric
($\operatorname{Re}(z\cdot z)$ is the SO33 $\eta$-invariant), but they are
different architectures and are not interchangeable.

## Pages

| Page | Read it for |
|---|---|
| [theory.md](theory.md) | The mathematics: realification, the closed-form exponential, what the flow conserves, why conservation is not equivariance, the covariant connection, the bivector lift and its structural zeros, message passing. |
| [models.md](models.md) | Every classifier: pipeline, feature sizes, parameter counts, which ones are exactly Lorentz-invariant and how that was measured. |
| [api.md](api.md) | Reference for the `so3c` package, `benchmarks.so3c_models`, and the harness entry points, with runnable examples. |
| [experiments.md](experiments.md) | Data preparation, commands, protocols, every measured result with its provenance, and the known limitations. |

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
| [`benchmarks/run_top_tagging.py`](../../benchmarks/run_top_tagging.py) | Top-tagging runner (internal and canonical protocols). |
| [`benchmarks/run_boost_robustness.py`](../../benchmarks/run_boost_robustness.py) | Re-evaluates a trained tagger on Lorentz-boosted test jets. |
| [`benchmarks/run_so3c_boost_ood.py`](../../benchmarks/run_so3c_boost_ood.py), [`benchmarks/so3c_synthetic.py`](../../benchmarks/so3c_synthetic.py) | Synthetic electromagnetic-invariant task. |
| [`tests/test_so3c_*.py`](../../tests/) | 38 tests for the algebra, layers, lift, models and float32 behaviour. |

## Status in one screen

- **Exactly Lorentz-invariant taggers:** `so3c_invariant_set` (no flow),
  `so3c_covariant_set` and `so3c_message_set` (closed-form flows), and
  `so3c_interaction_set` (to ODE tolerance).
- **Not invariant once trained:** `so3c_equivariant_set` and the synthetic
  `SO3CFlowClassifier`. Their connection is computed from invariants, so it
  does not transform with the input. A freshly initialised model hides this,
  because its connection is zero and the flow is the identity
  ([theory.md §5](theory.md#5-conservation-is-not-equivariance)).
- **Best canonical top-tagging result:** `so3c_message_set` at $K=64$
  leading constituents, 13,862 parameters: AUC $0.9807\pm0.0001$, background
  rejection $1/\varepsilon_B=850\pm56$ at $\varepsilon_S=0.3$ (2 seeds,
  30 epochs).
- **Not state of the art.** PELICAN reaches AUC 0.9870 and rejection 2250 at
  208k parameters, and its own size sweep beats the model above on both
  metrics from 605 parameters up (0.9823 / 901).
- **Saturated at the probe protocol.** Six capacity changes to the
  message-passing model, up to 6.7× the parameters, moved AUC by at most
  +0.0001.

Details, provenance and the full list of caveats are in
[experiments.md](experiments.md).

## Tests

```bash
python -m pytest tests/ -q
```

The suite collects 67 tests; on a CPU-only machine 65 pass and 2 CUDA-only
tests in `tests/test_harness.py` are skipped. The SO3C tests are
`test_so3c_algebra.py` (5), `test_so3c_activation.py` (7),
`test_so3c_interaction.py` (5), `test_so3c_lift.py` (4),
`test_so3c_models.py` (11) and `test_so3c_float32.py` (6); the training
harness, checkpointing and figure scripts are covered by `test_harness.py` (15).
