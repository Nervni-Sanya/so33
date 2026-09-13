# SO3C API reference

`import so3c` exposes the algebra functions, `HermitianMetric`,
`SO3CActivation` and `SO3CInteraction`. The lift lives in `so3c.lift` and is
not re-exported. Classifiers are in `benchmarks.so3c_models`; the harness is
`benchmarks.models`, `benchmarks.tabular_runner`, `benchmarks.train` and
`benchmarks.ensemble_scores`.

Shapes use `…` for leading batch dimensions. Real tensors default to
`torch.float64`; the complex dtype follows the real one (`float64 → complex128`,
`float32 → complex64`). Installation and dependencies are in
[experiments.md](experiments.md#1-setup).

## `so3c.algebra`

| Name | Value |
|---|---|
| `DIM_C` | 3 — complex dimension |
| `DIM_R` | 6 — real dimension of $\mathbb C^3\cong\mathbb R^6$ |
| `N_GEN` | 6 — real dimension of $\mathfrak{so}(3,\mathbb C)$ |
| `ETA` | `tensor([1, 1, 1, -1, -1, -1], float64)`, equal to `so33.basis.ETA` |

| Function | Shapes | Description |
|---|---|---|
| `real_to_complex(v)` | `(…, 6)` → `(…, 3)` complex | $(x,y)\mapsto x+iy$ |
| `complex_to_real(z)` | `(…, 3)` → `(…, 6)` | $z\mapsto(\operatorname{Re}z,\operatorname{Im}z)$ |
| `complex_structure(dtype=float64)` | → `(6, 6)` | $J_c$, with $J_c^2=-I$ |
| `complex_bilinear(z, w)` | `(…, 3)`, `(…, 3)` → `(…)` | $\sum_i z_iw_i$, no conjugation |
| `bilinear_invariant(v)` | `(…, 6)` → `(…)` complex | $z\cdot z$ of the real state |
| `invariant_features(v)` | `(…, 6)` → `(…, 2)` | $\operatorname{asinh}$ of $(\operatorname{Re},\operatorname{Im})\,z\cdot z$ |
| `so3_generators(dtype=float64)` | → `(3, 3, 3)` | $(L_i)_{jk}=-\varepsilon_{ijk}$ |
| `so3c_generator_stack(dtype=float64)` | → `(6, 6, 6)` | $[R_1,R_2,R_3,B_1,B_2,B_3]$ on $\mathbb R^6$ |
| `cross_matrix(a)` | `(…, 3)` → `(…, 3, 3)` | $[a]_\times$, real or complex |
| `expm_so3c(a, t=1.0)` | `(…, 3)` → `(…, 3, 3)` complex | $\exp(t[a]_\times)$, closed form; real `a` is promoted to complex |
| `complex_matrix_to_real(Q)` | `(…, 3, 3)` → `(…, 6, 6)` | $\begin{pmatrix}\operatorname{Re}Q&-\operatorname{Im}Q\\ \operatorname{Im}Q&\operatorname{Re}Q\end{pmatrix}$ |
| `random_group_element(rot_scale=1.0, boost_scale=0.5, dtype=float64, generator=None)` | → `(6, 6)` | real form of $\exp([\rho+i\beta]_\times)$, $\rho\sim\mathcal N(0,\text{rot\_scale}^2)$, $\beta\sim\mathcal N(0,\text{boost\_scale}^2)$ per component |

## `so3c.HermitianMetric`

```text
HermitianMetric(hidden: int = 16, dtype: torch.dtype = torch.float64)
```

`Linear(2→hidden) → tanh → Linear(hidden→6)`, output layer zero-initialised.

| Method | Description |
|---|---|
| `forward(v)` | `(…, 6)` real state → `(…, 3)` complex connection $a=\rho+i\beta$, computed from `invariant_features(v)` only |
| `metric_tensor(v)` | `(…, 3, 3)` Hermitian matrix $I+i[\beta]_\times$ (diagnostic) |
| `weight_penalty()` | sum of squared output-layer weights and bias |

## `so3c.SO3CActivation`

```text
SO3CActivation(T=1.0, mode="dynamic", method="exact", rtol=1e-7, atol=1e-9,
               adjoint=False, reg_coef=1e-3, dtype=torch.float64,
               bound_input="bilinear", hidden=16, scale_connection=True,
               solver_options=None)
```

| Argument | Meaning |
|---|---|
| `T` | flow time |
| `mode` | `"dynamic"`: connection from `HermitianMetric`; `"static"`: 6 learnable scalars (initialised $\mathcal N(0,0.01^2)$) |
| `method` | `"exact"`: closed form; otherwise a torchdiffeq method (`"dopri5"`, `"rk4"`, …) |
| `rtol`, `atol`, `solver_options` | solver settings, ignored in exact mode |
| `adjoint` | adjoint backpropagation for solver methods |
| `reg_coef` | scale of `regularization_loss()` |
| `bound_input` | `"bilinear"`, `"euclidean"` or `"none"` ([theory.md §4](theory.md#4-the-flow-and-what-it-conserves)) |
| `hidden` | width of the metric MLP (dynamic mode) |
| `scale_connection` | soft-normalise $a\mapsto a/(1+\lVert a\rVert)$ |

Invalid `mode` or `bound_input` values raise `ValueError`.

| Method | Description |
|---|---|
| `forward(x)` | `(…, 6)` real or `(…, 3)` complex → same layout. Other trailing sizes raise `ValueError`. |
| `regularization_loss(x=None)` | static: `reg_coef·‖coeffs‖²`; dynamic: `reg_coef` × mean $\lVert a(s(x))\rVert^2$ over the batch `x`, or × the metric weight penalty when `x` is `None` |
| `invariant_drift(x)` | per-sample $\lvert z(T)\cdot z(T)-z(0)\cdot z(0)\rvert$ measured after the bound, i.e. for the flow alone |

Properties to rely on: $z\cdot z$ is conserved by the flow (machine precision
in exact mode); the flow is the identity while the connection is zero, but the
default `"bilinear"` bound still rescales the input. The layer is **not** an
$SO(3,\mathbb C)$-equivariant map when its connection is non-zero
([theory.md §5](theory.md#5-conservation-is-not-equivariance)); equivariant
models need a covariant connection built from several states.

## `so3c.SO3CInteraction`

```text
SO3CInteraction(hidden=16, T=1.0, method="dopri5", rtol=1e-6, atol=1e-8,
                adjoint=False, reg_coef=1e-3, dtype=torch.float64)
```

| Method | Description |
|---|---|
| `forward(x, mask=None)` | `(B, N, 6)` real or `(B, N, 3)` complex, optional `(B, N)` mask (1 = real) → same layout. Padded entries come back unchanged (up to round-off in the solver's output interpolation) and do not influence real ones. |
| `regularization_loss()` | `reg_coef` × squared output-layer weights of the coupling MLP |
| `invariant_drift(x, mask=None)` | per-particle $\lvert z_a(T)\cdot z_a(T)-z_a(0)\cdot z_a(0)\rvert$ |

Equivariant and conserving to solver tolerance, identity at initialisation
([theory.md §8](theory.md#8-the-multi-particle-ode-so3cinteraction)).

## `so3c.lift`

| Function | Shapes | Description |
|---|---|---|
| `bivector_lift(p4, q4)` | `(…, 4)`, `(…, 4)` → `(…, 3)` complex | $(E_p\vec q-E_q\vec p)+i(\vec p\times\vec q)$ |
| `jet_bivectors(p4, mask)` | `(B, K, 4)`, `(B, K)` → `(B, K, 3)` complex | $z_a=\operatorname{bivec}(p_a,P)$, $P$ the masked jet total; padding gives 0 |
| `minkowski_inner(p4, q4)` | `(…, 4)`, `(…, 4)` → `(…)` | $E_pE_q-\vec p\cdot\vec q$ |
| `lorentz_matrix(rho, beta, dtype=float64)` | `(3,)`, `(3,)` → `(4, 4)` | $\exp\omega$ with boost block $\beta$ and rotation block $[\rho]_\times$, in $(E,p_x,p_y,p_z)$ order |
| `random_lorentz_pair(rot_scale=1.0, boost_scale=0.5, dtype=float64, generator=None)` | → `((4, 4), (3,) complex)` | matched $\Lambda$ and $a=\rho+i\beta$ with $\operatorname{bivec}(\Lambda p,\Lambda q)=\exp([a]_\times)\operatorname{bivec}(p,q)$ |

`so3c.lift.__all__` also re-exports `so3_generators`.

## `benchmarks.so3c_models`

Constructor signatures (all `nn.Module`s with `forward(x)` and
`regularization_loss()`); model-level behaviour is described in
[models.md](models.md).

```text
SO3CInvariantSetClassifier(out_features=2, hidden=64, dtype=float64)
SO3CEquivariantSetClassifier(out_features=2, channels=4, hidden=64, act_hidden=16, T=1.0, dtype=float64)
SO3CCovariantSetClassifier(out_features=2, channels=4, hidden=64, act_hidden=16, T=1.0, dtype=float64)
SO3CMessageSetClassifier(out_features=2, channels=4, rounds=3, hidden=64, act_hidden=16,
                         scalar_dim=8, msg_dim=8, T=1.0, channel_mixing=True,
                         neighbors=None, beams=False, beam_energy=1.0, dropout=0.0,
                         mass_input=True, self_edges=True, relnorm_edge=False,
                         falpha=0, vector_channel=False, pair_latent=0, dtype=float64)
SO3CInteractionSetClassifier(out_features=2, hidden=64, interaction_hidden=16, T=1.0,
                             rtol=1e-5, atol=1e-7, dtype=float64)
SO3CBottleneck(in_features, out_features, mode="dynamic", T=1.0, hidden_metric=16,
               bound_input="none", dtype=float64)
MultiChannelSO3C(in_features, out_features, channels=4, mode="dynamic", T=1.0,
                 hidden_metric=16, bound_input="none", dtype=float64)
SO3CInvariantsClassifier(out_features=2, hidden=32, dtype=float64)
EtaOnlyClassifier(out_features=2, hidden=32, dtype=float64)
SO3CFlowClassifier(out_features=2, channels=4, hidden=32, act_hidden=16, T=1.0, dtype=float64)
```

With `beams=True`, `SO3CMessageSetClassifier` holds the beams in a buffer
`beam_p4` of shape `(2, 4)`, rows $E_b(1,0,0,+1)$ and $E_b(1,0,0,-1)$.
Transforming it in place (`model.beam_p4.copy_(...)`) is how the tests and
`run_boost_robustness` move the beams together with the jet. `beams`,
`vector_channel` or `pair_latent` combined with `neighbors` raise `ValueError`.
`N_POOLED_FULL = 11` and `N_POOLED_SIMPLE = 5` are the pooled-feature counts.

## Harness entry points

**`benchmarks.models.build_model(name, in_features, out_features, *, T=0.3, natural_hidden=256, dtype=float64, adjoint=True, so33_method="rk4", so33_step_size=None, representation="flat", bound_input=None, max_input_norm=8.0, pool="mean", so3c_kwargs=None)`**

- Set models need `representation="constituents"` and `in_features=4`.
  `so3c_kwargs` is passed to their constructors after filtering: the
  message-passing arguments `rounds`, `scalar_dim`, `msg_dim`,
  `channel_mixing`, `neighbors`, `beams`, `beam_energy`, `dropout`,
  `mass_input`, `self_edges`, `relnorm_edge`, `falpha`, `vector_channel` and
  `pair_latent` are dropped for every model except `so3c_message_set`;
  `channels`, `act_hidden` and `T` are also dropped for `so3c_invariant_set`,
  and `channels` and `act_hidden` for `so3c_interaction_set`.
- `so3c` and `so3c_static` need `representation="flat"` and receive
  `so3c_kwargs`; `so3c_multi` does not.
- `benchmarks.models.count_parameters(model)` counts trainable parameters.

**`benchmarks.tabular_runner.run_tabular_experiment(experiment, split, *, models=None, seed=0, epochs=30, batch_size=128, lr=3e-3, weight_decay=0.0, optimizer="adam", schedule="cosine", warmup_epochs=4, natural_hidden=256, T=0.3, representation="flat", pool="mean", results_dir="results", device="cpu", dtype=float64, so3c_kwargs=None, eval_chunk_size=4096, ckpt_dir=None, resume=False, max_seconds=None)`**
trains each model, evaluates it on the test split and writes
`<results_dir>/<experiment>__<model>__seed<seed>.json` plus a
`…__scores.npz` with per-example test scores. The record includes
`so3c_kwargs` and `dtype`; result files written before 2026-09-13 lack both.
The global seed is set before each model is built, so identical commands
reproduce each other. The file layout is described in
[experiments.md](experiments.md#5-output-files).

**`benchmarks.train.TrainConfig`** fields and defaults: `epochs=30`,
`batch_size=128`, `lr=3e-3`, `weight_decay=0.0`, `grad_clip=1.0`,
`cosine_schedule=True`, `optimizer="adam"`, `schedule="cosine"`,
`warmup_epochs=4`, `early_stop_patience=None`, `seed=0`, `device="cpu"`,
`eval_chunk_size=4096`, `ckpt_path=None`, `ckpt_every=1`, `resume=False`,
`max_seconds=None`. `optimizer` is `"adam"` or `"adamw"`. `schedule="cosine"`
anneals over the run; `schedule="lorentznet"` is the LorentzNet and PELICAN
recipe, `lorentznet_lr_factor(epoch, total_epochs, warmup=4, t0=4, t_mult=2, decay_epochs=3, gamma=0.5)`:
linear warm-up, cosine annealing with warm restarts (cycles of 4, 8 and 16
epochs in a 35-epoch run) and a final decay by 0.5 per epoch.
`cosine_schedule=False` turns scheduling off.
`train_classifier(model, X_train, y_train, X_val, y_val, cfg)` minimises
cross-entropy plus `model.regularization_loss()`, clips gradients, validates
every epoch and returns the final epoch's weights (there is no best-epoch
restore). A checkpoint stores model, optimizer, scheduler and RNG state, so a
resumed run continues exactly; `max_seconds` limits a single session.
`forward_in_chunks(model, X, chunk_size)` bounds evaluation memory for the
$K\times K$ pairwise readouts.

**`benchmarks.ensemble_scores`**, run as
`python -m benchmarks.ensemble_scores --results-dir DIR [--model M] [--experiment E] [--tolerance T] [--out FILE]`,
averages the saved class-1 probabilities over every seed of a model and prints
the ensemble's AUC and background rejection. Before that it recomputes each
member's metrics from its own scores and stops if they disagree with the stored
values, or if the members' test labels differ.

## Examples

Each block below runs as-is from the repository root.

**A flow, and what it conserves.**

```python
import torch
from so3c import SO3CActivation

torch.manual_seed(0)
act = SO3CActivation()                       # dynamic metric, closed form
with torch.no_grad():                        # give the zero-init connection a value
    act.metric.net[-1].weight.normal_(0, 0.5)
    act.metric.net[-1].bias.normal_(0, 0.5)

x = torch.randn(8, 6, dtype=torch.float64)   # (Re z, Im z)
y = act(x)                                   # (8, 6)
print(act.invariant_drift(x).max())          # below 1e-14: z.z is conserved
z_out = act(torch.complex(x[:, :3], x[:, 3:]))   # complex input -> complex output
print(torch.allclose(torch.cat([z_out.real, z_out.imag], dim=-1), y))  # True
```

**The lift commutes with the Lorentz group.**

```python
import torch
from so3c import expm_so3c
from so3c.lift import bivector_lift, random_lorentz_pair

p = torch.randn(5, 4, dtype=torch.float64)
q = torch.randn(5, 4, dtype=torch.float64)
L, a = random_lorentz_pair(boost_scale=1.0)   # 4x4 Lorentz matrix, matching a = rho + i beta
lhs = bivector_lift(p @ L.T, q @ L.T)
rhs = (expm_so3c(a) @ bivector_lift(p, q).unsqueeze(-1)).squeeze(-1)
print((lhs - rhs).abs().max())                # ~1e-15
```

**A jet tagger forward pass.**

```python
import torch
from benchmarks.models import build_model, count_parameters

model = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents")
print(count_parameters(model))                # 13862

jets = torch.zeros(2, 32, 5, dtype=torch.float64)
jets[..., 1:4] = torch.randn(2, 32, 3, dtype=torch.float64)
jets[..., 0] = jets[..., 1:4].norm(dim=-1)    # massless constituents: E = |p|
jets[:, :20, 4] = 1.0                         # 20 real constituents, 12 padding
jets[:, 20:, :4] = 0.0
logits = model(jets)                          # (2, 2)
print(logits.shape)
```

**Beams: covariant together with the beams, not without them.**

```python
import math
import torch
from benchmarks.models import build_model
from so3c.lift import random_lorentz_pair

torch.manual_seed(0)
model = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True, "channels": 8})
with torch.no_grad():                         # zero-initialised heads make the flow the identity
    for head in model.w_head:
        head.weight.normal_(0, 0.5)
        head.bias.normal_(0, 0.5)

p = torch.randn(6, 12, 3, dtype=torch.float64)
jets = torch.cat([p.norm(dim=-1, keepdim=True), p,
                  torch.ones(6, 12, 1, dtype=torch.float64)], dim=-1)

def transform(L, x):                          # act on the 4-momenta, keep the mask
    return torch.cat([x[..., :4] @ L.T, x[..., 4:]], dim=-1)

L, _ = random_lorentz_pair(boost_scale=0.5)
c, s = math.cos(1.1), math.sin(1.1)
Rz = torch.eye(4, dtype=torch.float64)        # rotation about the beam (z) axis
Rz[1, 1], Rz[1, 2], Rz[2, 1], Rz[2, 2] = c, -s, s, c

with torch.no_grad():
    base = model(jets)
    print((model(transform(L, jets)) - base).abs().max())    # clearly non-zero: the jet moved past fixed beams
    print((model(transform(Rz, jets)) - base).abs().max())   # ~1e-16: rotation about the beam axis
    beams = model.beam_p4.clone()
    model.beam_p4.copy_(beams @ L.T)                          # move the beams with the jet
    print((model(transform(L, jets)) - base).abs().max())    # ~1e-16: exactly covariant
    model.beam_p4.copy_(beams)
```

**The multi-particle ODE with padding.**

```python
import torch
from so3c import SO3CInteraction

torch.manual_seed(0)
layer = SO3CInteraction()
with torch.no_grad():                         # a non-zero coupling, as after training
    layer.phi[-1].weight.normal_(0, 0.5)
    layer.phi[-1].bias.normal_(0, 0.5)

x = torch.randn(2, 10, 6, dtype=torch.float64)
mask = torch.ones(2, 10)
mask[:, 7:] = 0                               # last three particles are padding
out = layer(x, mask=mask)                     # (2, 10, 6)
print((out[:, :7] - x[:, :7]).abs().max())    # O(1): real particles move
print((out[:, 7:] - x[:, 7:]).abs().max())    # ~1e-16: padding comes back unchanged
print(layer.invariant_drift(x, mask).max())   # ~1e-4 at the default rtol=1e-6; tighter tolerances shrink it
```
