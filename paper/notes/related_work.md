# Related-work notes — for Section 6 of the preprint

Goal of this file: gather, in one place, the citations the related-work
section must address, the one-sentence summary of each, and our positioning
relative to it. To be expanded into prose in Week 5.

## Lorentz-equivariant networks (the direct competitors)

### LGN — Bogatskiy et al. 2020 (ICML)
- First learnable architecture with exact $\mathrm{SO}(1,3)$ equivariance via
  tensor representations.
- Uses fixed analytic action of the group; no learned connection.
- Demonstrated on top-quark tagging.
- **Our positioning:** they fix the action; we learn it via an ODE
  connection. Their guarantees are exact; ours are approximate
  ($\sim 10^{-3}$ relative error, measured).

### LorentzNet — Gong et al. 2022 (JHEP)
- Simpler equivariant message-passing on Minkowski 4-momenta with
  invariant edge features.
- Significantly outperforms LGN and is competitive with the best
  non-equivariant top taggers on the standard benchmarks.
- **Our positioning:** LorentzNet is the SOTA-adjacent reference for
  real-data top tagging. We **do not beat it** and must say so explicitly.
  Our contribution is methodological: a different inductive bias on a
  related (higher-rank, $\mathrm{SO}(3,3)$) algebra, with a learnable
  connection.

### PELICAN — Bogatskiy et al. 2022
- Permutation-equivariant aggregator with Lorentz invariance/covariance.
- Strong results on jet tagging and reconstruction tasks.
- **Our positioning:** same as LorentzNet — we cite as the state of the
  jet-tagging art and decline to compete on raw AUC.

## Broader equivariance background

### EGNN — Satorras, Hoogeboom, Welling 2021 (ICML)
- $E(n)$-equivariant graph networks via invariant distance features.
- Foundational for "build invariants, then apply MLP" pattern.
- **Our positioning:** `eta_invariants` is structurally in this family
  (lift, then read out $\eta$-invariants). The contribution is that the
  lift itself is a geodesic on a non-trivial algebra.

### Cohen & Welling 2016 (ICML)
- G-CNNs: foundational paper on building group equivariance into
  architecture rather than learning it from augmentation.
- **Our positioning:** cite once in the intro paragraph framing the
  invariance-vs-equivariance distinction.

### EMLP — Finzi, Welling, Wilson 2021 (ICML)
- Practical recipe for equivariant MLPs on arbitrary matrix groups,
  including $\mathrm{O}(p,q)$.
- Very close conceptually to our work, especially the `eta_invariants`
  variant. **Important precedent to cite honestly.**
- **Our positioning:** EMLP builds equivariant linear maps from the
  algebra structure; we build a non-linear activation as a geodesic on
  the same kind of algebra. Complementary directions.

## ODE methods

### Neural ODE — Chen et al. 2018 (NeurIPS)
- Treats a network layer as an ODE flow integrated via adaptive solver.
- **Our positioning:** our activation is a Neural-ODE-style layer
  specialized to the structure-preserving flow on $\mathrm{SO}(3,3)$.

## Things we do not need to cite extensively

- General jet-tagging architectures (ParticleNet, Particle Transformer):
  one-sentence mention in the experiments preamble is enough -- they are
  not equivariant and our setting is not directly comparable.
- Generic group-theory references: keep to one canonical textbook.

## The positioning sentence (try in abstract / conclusion)

> "Where prior Lorentz-equivariant architectures fix the group action
> analytically, we instead parameterize a connection and integrate it
> as a geodesic ODE, accepting approximate equivariance in exchange for
> a learnable inductive bias on the same algebra."
