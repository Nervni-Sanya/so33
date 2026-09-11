# SO3C models

All classes are in [`benchmarks/so3c_models.py`](../../benchmarks/so3c_models.py)
and are built by name through `benchmarks.models.build_model`, which is what
`--models` on the command line resolves to. Parameter counts below were
obtained by instantiating each model with its defaults.

## Input format

Set models take a `(B, K, 5)` tensor per batch: columns `0:4` are the
constituent 4-momentum $(E,p_x,p_y,p_z)$ and column `4` is a mask (1 for a
real constituent, 0 for padding). `benchmarks.datasets.load_top_tagging_constituents`
produces it: the $K$ leading constituents by $p_T$, zero-padded, divided by one
global scale (the RMS over real training constituents, which keeps
$E^2-\vec p^{\,2}$ up to a constant factor). Each model builds its own lift
internally, so the SO33 arguments of `build_model` (`T`, `bound_input`,
solver settings) do not apply to them.

Flat models take `(B, F)` feature vectors (HIGGS, Adult).

## Registry

| `--models` name | Class | Parameters | Flow | Lorentz-invariant logits |
|---|---|---:|---|---|
| `so3c_invariant_set` | `SO3CInvariantSetClassifier` | 5,122 | none | **yes**, by construction |
| `so3c_equivariant_set` | `SO3CEquivariantSetClassifier` | 9,056 | `SO3CActivation` per channel | **no**, once the connection is non-zero |
| `so3c_covariant_set` | `SO3CCovariantSetClassifier` | 9,078 | one closed-form covariant step | **yes**, exactly |
| `so3c_message_set` | `SO3CMessageSetClassifier` | 13,862 | 3 closed-form covariant rounds + scalar channel | **yes**, exactly |
| `so3c_interaction_set` | `SO3CInteractionSetClassifier` | 5,652 | `SO3CInteraction` ODE | **yes**, to solver tolerance |
| `so3c` | `SO3CBottleneck(mode="dynamic")` | $6F+170$ | exact flow on a learned 6-D embedding | no symmetry (learned lift) |
| `so3c_static` | `SO3CBottleneck(mode="static")` | $6F+26$ | same, 6 constant coefficients | no symmetry |
| `so3c_multi` | `MultiChannelSO3C` (4 blocks) | $24F+674$ | 4 parallel bottleneck flows | no symmetry |

$F$ is the number of input features (e.g. 338 / 194 / 1,346 parameters for
the 28 HIGGS features). Three heads used only by
[`run_so3c_boost_ood.py`](../../benchmarks/run_so3c_boost_ood.py) are not in the
registry: `SO3CInvariantsClassifier` (1,218), `SO3CFlowClassifier` (1,952; not
invariant once trained) and `EtaOnlyClassifier` (1,186).

## How invariance was established

"Exactly" means up to floating-point round-off. Freshly built models have a
zero connection, so their flows are the identity and cannot fail an invariance
check; the numbers below were measured in float64 after setting every
connection head to random non-zero values, on random jets transformed by
`so3c.lift.random_lorentz_pair`:

| Model | Relative logit change, boost scale 0.5 | 1.5 |
|---|---:|---:|
| `so3c_invariant_set` | $1.2\times10^{-15}$ | $1.1\times10^{-14}$ |
| `so3c_message_set` | $1.9\times10^{-15}$ | $4.2\times10^{-14}$ |
| `so3c_covariant_set` | $2.4\times10^{-10}$ | $3.3\times10^{-9}$ |
| `so3c_interaction_set` (`rtol=1e-5`) | $5.4\times10^{-7}$ | $4.7\times10^{-7}$ |
| `so3c_equivariant_set` | $6.7\times10^{-2}$ | $3.3\times10^{-1}$ |

The same distinction shows on real data: see the boost-robustness table in
[experiments.md](experiments.md#d-boost-robustness). The mechanism is in
[theory.md §5](theory.md#5-conservation-is-not-equivariance). In the test suite,
`test_trained_regime_invariance`, `test_covariant_set_is_equivariant_when_excited`,
`test_message_set_is_equivariant_when_excited`,
`test_message_set_neighbour_graph_is_equivariant` and
`test_interaction_set_is_equivariant_when_excited` exercise non-zero
connections. `test_set_lorentz_invariance` and `test_classifier_invariance`
use freshly built models, so for the flow models they check the readout only.

---

## `so3c_invariant_set`

```
jets ─► z_a = bivec(p_a, P) ─► 5 pooled invariants (simple=True) ─┐
    └──────────────────────────► 7 Minkowski statistics ──────────┴─► MLP 12→64→64→2
```

No flow. Parameters: $832+4{,}160+130=5{,}122$. `regularization_loss()` is 0.
Knob: `hidden` (`--hidden`); `channels`, `act_hidden` and `T` are dropped by
the factory.

Earlier revisions pooled all 11 bivector features (5,506 parameters). The six
removed features are identically zero on the raw lift
([theory.md §9](theory.md#9-lifting-4-momenta-the-bivector-map)); on the
internal protocol AUC was unchanged ($0.9629\pm0.0017$ before,
$0.9631\pm0.0015$ after, 3 seeds). The canonical result for this model in
[experiments.md](experiments.md) was measured with the 5,506-parameter version.

## `so3c_equivariant_set` — not invariant once trained

```
jets ─► z_a ─► z_a^(c) = w_c z_a  (C complex scalars)
      ─► SO3CActivation(dynamic, exact, bilinear bound) on every (c, a)
      ─► per channel: 11 pooled invariants
         + Re/Im of z_tot^(c) · z_tot^(d), c ≤ d
         + 7 Minkowski statistics ─► MLP (11C + C(C+1) + 7 = 71)→64→64→2
```

The connection of `SO3CActivation` is a function of invariants, so it does not
transform with the jet, and the pairwise and cross-channel products the readout
depends on are not invariant once training moves the connection away from zero
([theory.md §5](theory.md#5-conservation-is-not-equivariance)). A model trained
for 8 epochs on 20k jets falls from AUC 0.9424 in the lab frame to
$0.6052\pm0.1954$ at boost scale 2. It remains in the registry because several
recorded results (the canonical 0.9744, the scaling study, the $K$ sweep and
the binned analysis) were measured on it; use `so3c_covariant_set` instead.

Parameters: channel weights 8 + metric MLP 150 + readout 8,898 = 9,056.
Knobs: `channels`, `hidden`, `act_hidden` (metric width), `T` (`--flow-T`).
`regularization_loss()` is the activation's output-layer weight penalty
(`reg_coef=1e-3`).

## `so3c_covariant_set`

```
jets ─► z_a ─► φ(asinh Re/Im of s_aa, s_bb, s_ab) ∈ R^C   (6→act_hidden→C, zero-init)
      ─► a_a^(c) = z_a × (1/n) Σ_b φ_c(a,b) z_b
      ─► z_a^(c) = expm_so3c(a_a^(c), t=−T) z_a          (one group element each)
      ─► same readout as so3c_equivariant_set (71 features at C=4) ─► MLP ─► 2
```

Every channel starts from the same lifted state and differs through its own
weight $\varphi_c$. There is no input bound and no connection normalisation in
this model (unlike `so3c_message_set`). Parameters: $\varphi$ 180 + readout
8,898 = 9,078. Knobs: `channels`, `hidden`, `act_hidden`, `T`.
`regularization_loss()` is $10^{-3}\lVert\varphi_\text{out}\rVert^2$.

## `so3c_message_set`

The covariant step applied for several rounds, with a scalar state per
particle and complex channel mixing
([theory.md §10](theory.md#10-covariant-message-passing-so3c_message_set)).
The readout adds the masked mean and max of the scalar state, so its input is
$11C+C(C+1)+2D+7=87$ features at the defaults.

| Argument | Default | CLI flag | Meaning |
|---|---|---|---|
| `channels` | 4 | `--channels` | complex channels $C$ |
| `rounds` | 3 | `--rounds` | message-passing rounds $R$ |
| `hidden` | 64 | `--hidden` | readout MLP width |
| `act_hidden` | 16 | `--act-hidden` | edge-network width |
| `scalar_dim` | 8 | `--scalar-dim` | scalar state width $D$; 0 removes the scalar channel |
| `msg_dim` | 8 | `--msg-dim` | edge message width $M$ |
| `T` | 1.0 | `--flow-T` | flow time |
| `channel_mixing` | `True` | — | per-round complex mixing matrices |
| `neighbors` | `None` | `--neighbors` | $k$ strongest partners by $\lvert\operatorname{Re}z_a\cdot z_b\rvert$; `None` is dense |

Parameters for the variants that were run (`neighbors` does not change the count):

| Configuration | Parameters |
|---|---:|
| defaults ($R=3$, $C=4$, $D=M=8$) | 13,862 |
| `rounds=1` | 11,262 |
| `rounds=6` | 17,762 |
| `scalar_dim=0` | 10,406 |
| `scalar_dim=24, msg_dim=24` | 20,678 |
| `channels=8` | 21,658 |
| `channels=16` | 43,970 |
| `hidden=256` | 92,774 |

Note that `rounds=1` is not `so3c_covariant_set` plus a scalar channel: this
model also bounds the input lift and the connection, and mixes channels.
`regularization_loss()` is $10^{-3}$ times the squared weights of all weight
heads. A dense round costs $O(CK^2)$ per jet.

## `so3c_interaction_set`

```
jets ─► z_a ─► z_a / (1 + |z_a·z_a|^½) ─► SO3CInteraction (dopri5, rtol 1e-5, atol 1e-7, mask)
      ─► 11 pooled invariants + 7 Minkowski statistics ─► MLP 18→64→64→2
```

The only set model that integrates an ODE. Parameters: interaction MLP 146 +
readout 5,506 = 5,652. Knobs from the CLI: `hidden`, `T`; the interaction
MLP width and solver tolerances are constructor arguments only. On the
canonical protocol it took 4.92 GPU-hours per seed against 0.53 for
`so3c_covariant_set`, for a lower AUC.

## Flat models: `so3c`, `so3c_static`, `so3c_multi`

`SO3CBottleneck` is `Linear(F→6) → SO3CActivation(method="exact", bound_input="none") → Linear(6→2)`;
`MultiChannelSO3C` runs four such blocks in parallel and concatenates their
24 outputs before `Linear(24→2)`. The flow acts on a learned embedding, so
these models have no Lorentz symmetry: they test the flow as an activation
function. The closed-form flow is a bounded group element for any input, so
neither input bounding nor a norm cap is used on this path. `build_model`
forwards `so3c_kwargs` (for example `T`, `hidden_metric`, `bound_input`) to
`SO3CBottleneck` but not to `MultiChannelSO3C`, and it does not forward the
harness `T`. These names reject `representation="constituents"`.

## Synthetic-task heads

Used by [`run_so3c_boost_ood.py`](../../benchmarks/run_so3c_boost_ood.py) on
$(B,6)$ inputs $(\operatorname{Re}z,\operatorname{Im}z)$:

| Model name | Class | Pipeline | Invariant |
|---|---|---|---|
| `so3c_invariants` | `SO3CInvariantsClassifier` | $\operatorname{asinh}(\operatorname{Re}z\cdot z,\operatorname{Im}z\cdot z)$ → MLP 2→32→32→2 | yes |
| `eta_only` | `EtaOnlyClassifier` | $\operatorname{asinh}\operatorname{Re}z\cdot z$ → MLP 1→32→32→2 | yes, and blind to $\operatorname{Im}z\cdot z$ |
| `so3c_flow` | `SO3CFlowClassifier` | $z_c=w_cz$ ($C=4$) → shared `SO3CActivation` → Re/Im of $z_c(T)\cdot z_d(T)$, $c\le d$ (20) → MLP 20→32→32→2 | only while the connection is zero |

## Parameter-matched generic baseline

`--models relu_mlp,gelu_mlp --natural-hidden 1293` builds the Deep Sets model
`Linear(4→1293) → activation per constituent → masked mean → Linear(1293→2)`:
9,053 parameters, matched to the 9,056-parameter set models, with no Lorentz
structure. It is the "same capacity, no geometry" control.
