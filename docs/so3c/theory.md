# SO3C theory

Conventions used throughout: $z\cdot w=\sum_i z_i w_i$ is the complex
**bilinear** form (no conjugation). $[a]_\times$ is the cross-product matrix,
$[a]_\times u=a\times u$, for real or complex $a$. Four-vectors are
$(E,p_x,p_y,p_z)$ with Minkowski product $\langle p,q\rangle=E_pE_q-\vec p\cdot\vec q$.
Every function and class named here is in [`so3c/`](../../so3c/) or
[`benchmarks/so3c_models.py`](../../benchmarks/so3c_models.py).

## 1. The algebra

The complex antisymmetric $3\times3$ matrices form $\mathfrak{so}(3,\mathbb C)$.
As a real Lie algebra it is 6-dimensional: rotations $J_i$ and their imaginary
copies $K_i=iJ_i$, with

$$[J_i,J_j]=\varepsilon_{ijk}J_k,\qquad [J_i,K_j]=\varepsilon_{ijk}K_k,\qquad [K_i,K_j]=-\varepsilon_{ijk}J_k .$$

These are the Lorentz commutation relations, so
$\mathfrak{so}(3,\mathbb C)\cong\mathfrak{so}(3,1)$, and the group
$SO(3,\mathbb C)=\{Q\in\mathbb C^{3\times3}:\ Q^TQ=I,\ \det Q=1\}$ is isomorphic
to the proper orthochronous Lorentz group. The physical model is the
Riemann–Silberstein vector $F=E+iB$: rotations rotate $E$ and $B$ together,
boosts mix them through the imaginary part.

Basis in code (`so3_generators`): $(L_i)_{jk}=-\varepsilon_{ijk}$, so that
$\sum_i a_iL_i=[a]_\times$. An algebra element is stored as $a=\rho+i\beta\in\mathbb C^3$:
$\rho$ is the rotation part, $\beta$ the boost part.

## 2. Realification $\mathbb C^3\cong\mathbb R^6$

A complex vector $z=x+iy$ is stored as the real 6-vector $v=(x,y)$
(`real_to_complex`, `complex_to_real`). Multiplication by $i$ becomes the
complex structure $J_c=\begin{pmatrix}0&-I\\I&0\end{pmatrix}$
(`complex_structure`). The bilinear invariant splits into two real invariants:

$$\operatorname{Re}(z\cdot z)=|x|^2-|y|^2=v^T\eta v,\qquad \operatorname{Im}(z\cdot z)=2\,x\cdot y,\qquad \eta=\operatorname{diag}(1,1,1,-1,-1,-1).$$

`so3c.ETA` is the same tensor as `so33.basis.ETA`, so $\operatorname{Re}(z\cdot z)$
is exactly the $\eta$-invariant of the parent SO33 architecture. The realified
generators (`so3c_generator_stack`) are $R_i=\operatorname{blockdiag}(L_i,L_i)$
and $B_i=J_cR_i$. Each satisfies the $\mathfrak{so}(3,3)$ condition
$A^T\eta+\eta A=0$ and commutes with $J_c$: $SO(3,\mathbb C)$ is the subgroup of
$SO(3,3)$ that commutes with the complex structure. It preserves both real
invariants; $SO(3,3)$ preserves only the first. A complex matrix acts on
$\mathbb R^6$ as $\begin{pmatrix}\operatorname{Re}Q&-\operatorname{Im}Q\\ \operatorname{Im}Q&\operatorname{Re}Q\end{pmatrix}$
(`complex_matrix_to_real`).

For $z=E+iB$ the two invariants are the classical field invariants
$|E|^2-|B|^2$ and $2E\cdot B$.

## 3. The closed-form exponential

For $A=[a]_\times$ with complex $a$, Cayley–Hamilton gives
$A^3=-(a\cdot a)A$, and the exponential is the complex Rodrigues formula
(`expm_so3c(a, t)`):

$$\exp(tA)=I+\frac{\sin\theta}{\theta}\,tA+\frac{1-\cos\theta}{\theta^2}\,t^2A^2,\qquad \theta^2=t^2\,(a\cdot a)\in\mathbb C .$$

Both coefficients are even entire functions of $\theta$, so the branch of the
complex square root never matters. When $|\theta^2|<10^{-8}$ the code switches
to the series $1-\theta^2/6+\theta^4/120$ and $1/2-\theta^2/24+\theta^4/720$,
so the map is smooth through $a=0$. The result satisfies $Q^TQ=I$ and
$\det Q=1$; on 256 random complex $a$ in float64 both deviations were below
$2\times10^{-13}$ (the tests require $10^{-10}$ in float64 and $10^{-5}$ in
float32). Real `float64` inputs give `complex128`, `float32` gives `complex64`.

## 4. The flow and what it conserves

$$\dot z=-[a]_\times z .$$

Because $[a]_\times$ is antisymmetric, $\frac{d}{dt}(z\cdot z)=-2\,z^T[a]_\times z=0$:
the flow conserves $z\cdot z$ exactly, real and imaginary parts.

`SO3CActivation` offers two connections:

- `mode="static"` — six learnable scalars, $a=\rho+i\beta$ constant.
- `mode="dynamic"` (default) — $a=a(s)$ predicted by `HermitianMetric` from
  $s=(\operatorname{asinh}\operatorname{Re}\,z\cdot z,\ \operatorname{asinh}\operatorname{Im}\,z\cdot z)$.
  Since $s$ is conserved, $a$ is constant along each flow line and the whole
  flow is one group element,
  $$z(T)=\exp\big(-T\,[a(s_0)]_\times\big)\,z_0 .$$

`method="exact"` evaluates that formula. Any torchdiffeq method (`"dopri5"`,
`"rk4"`, …) integrates the same field instead, recomputing $a$ at every step;
the tests require the two to agree to $10^{-8}$ at `rtol=1e-10`. In exact mode
$z\cdot z$ drifts by at most $\sim10^{-15}$ (measured $1.4\times10^{-15}$ on 512
states with a non-zero connection).

**The metric.** `HermitianMetric` is `Linear(2→hidden) → tanh → Linear(hidden→6)`,
split into $\rho(s),\beta(s)\in\mathbb R^3$. The Hermitian form
$g(s)=I+i[\beta(s)]_\times$ is Hermitian for every $s$ because $[\beta]_\times$
is real antisymmetric (`metric_tensor`, a diagnostic); the flow consumes the full
connection $a=\rho+i\beta$. The output layer is zero-initialised, so $a\equiv0$
at the start of training.

**Bounding**, applied to the input before the flow (`bound_input`):

| value | map | commutes with $SO(3,\mathbb C)$ |
|---|---|---|
| `"bilinear"` (default) | $v\,/\,(1+\lvert z\cdot z\rvert^{1/2})$ | yes: an invariant scalar times $v$ |
| `"euclidean"` | $v\,/\,(1+\lVert v\rVert_2)$ | no: $\lVert v\rVert_2$ changes under boosts (ablation) |
| `"none"` | identity | yes |

With `scale_connection=True` (default) the connection is soft-normalised,
$a\mapsto a/(1+\lVert a\rVert)$ with $\lVert a\rVert^2=\sum(\operatorname{Re}^2+\operatorname{Im}^2)$,
which bounds the rapidity accumulated over the horizon. The norm is computed
with $10^{-12}$ under the square root: a plain `abs()` has an undefined
gradient at $a=0$, which is exactly where the zero-initialised metric starts.

## 5. Conservation is not equivariance

A layer $F$ is $SO(3,\mathbb C)$-equivariant when $F(Qz)=Q\,F(z)$. For the flow
of §4, with $a$ depending on $z$ only through invariants,

$$F(Qz)=\exp(-T[a]_\times)\,Qz,\qquad Q\,F(z)=Q\exp(-T[a]_\times)\,z,$$

with the **same** $a$ on both sides. They agree only when $Q$ commutes with
$[a]_\times$. Equivariance needs the connection to transform, $a\mapsto Qa$:
since $[Qa]_\times=Q[a]_\times Q^T$ for $\det Q=1$,

$$\exp(-T[Qa]_\times)\,Qz=Q\exp(-T[a]_\times)\,z .$$

What an invariant-fed flow does preserve is each output's own invariant,
$F(Qz)\cdot F(Qz)=z\cdot z$. What it does not preserve is any product of two
different flowed states. For two channels $z_c=w_cz$,

$$z_c(T)\cdot z_d(T)=w_cw_d\;z^T\exp(T[a_c]_\times)\exp(-T[a_d]_\times)\,z,$$

which changes under $Q$ unless $Q$ commutes with that product. Pairwise
$z_a\cdot z_b$ terms in a set readout fail the same way.

Measured in float64 with the connection heads set to random non-zero values:

| Layer or model | Transformation | Relative change |
|---|---|---|
| `SO3CActivation`, dynamic or static | $\lvert F(gx)-gF(x)\rvert$, $SO(3,\mathbb C)$ boost scale 0.3 and 1.0 | 0.35 – 0.81 |
| `SO3CActivation` | $z\cdot z$ of the output (absolute) | $\le1.2\times10^{-13}$ |
| `SO3CFlowClassifier` | logits, boost scale 0.3 / 1.0 | $2.3\times10^{-2}$ / $1.4\times10^{-1}$ |
| `so3c_equivariant_set` | logits, Lorentz boost scale 0.5 / 1.5 | $6.7\times10^{-2}$ / $3.3\times10^{-1}$ |

This is the defect in `so3c_equivariant_set` (and in the synthetic
`SO3CFlowClassifier`). A freshly built model conceals it: with $a\equiv0$ the
flow is the identity and every invariance check passes. The tests that matter
therefore excite the connection first (`_excite_connection` and the
`*_when_excited` tests in [`tests/test_so3c_models.py`](../../tests/test_so3c_models.py)).

## 6. Why a single state cannot be repaired

The only vectors built from one state $z$ that transform covariantly are
$f(\text{invariants})\,z$. Used as a connection, $[fz]_\times z=f\,z\times z=0$:
rotating about your own axis is the identity. An equivariant flow therefore
has to combine several states.

## 7. The covariant connection (`so3c_covariant_set`)

The cross product is covariant for $SO(3,\mathbb C)$: $Q(u\times v)=(Qu)\times(Qv)$
when $\det Q=1$. Weighting by invariants keeps that, so for particle $a$ and
channel $c$

$$a_a^{(c)}=z_a\times r_a^{(c)},\qquad r_a^{(c)}=\frac1n\sum_b\varphi_c(s_{aa},s_{bb},s_{ab})\,z_b,\qquad s_{ab}=z_a\cdot z_b,$$

transforms as $a\mapsto Qa$, and $z_a^{(c)}(T)=\exp(-T[a_a^{(c)}]_\times)\,z_a$
is exactly equivariant. $\varphi$ is `Linear(6→act_hidden) → tanh → Linear(act_hidden→C)`
on the arcsinh of $(\operatorname{Re},\operatorname{Im})$ of $s_{aa},s_{bb},s_{ab}$,
masked to real particle pairs and zero-initialised; $n$ is the number of real
particles. The connection is evaluated once at $t=0$ and held fixed, so each
state is updated by one closed-form group element with no solver.

The reference vector has to be a *weighted* sum: for the jet lift the plain
sum $\sum_b z_b$ vanishes identically (§9).

## 8. The multi-particle ODE (`SO3CInteraction`)

$$\dot z_a=-A_az_a,\qquad A_a=\frac1n\sum_b\varphi_{ab}\,\big(z_az_b^T-z_bz_a^T\big),\qquad \varphi_{ab}=\varphi_\theta(s_{aa},s_{bb},s_{ab})\in\mathbb C .$$

Contracting the bivector gives
$\dot z_a=-\frac1n\sum_b\varphi_{ab}\,(s_{ab}\,z_a-s_{aa}\,z_b)$, so no matrices
are built and the cost is $O(N^2)$ products. Bivectors transform in the
adjoint, $(Qz_a)(Qz_b)^T-(Qz_b)(Qz_a)^T=Q(z_az_b^T-z_bz_a^T)Q^T$, and
$\varphi$ sees only invariants, so the vector field is equivariant. $A_a$ is
antisymmetric, so each $z_a\cdot z_a$ is conserved; the pairwise $z_a\cdot z_b$
are not, which is the feature mixing. Because $A_a$ changes along the
trajectory there is no closed form: the layer integrates with torchdiffeq
(`dopri5`, `rtol=1e-6`, `atol=1e-8` by default). Padded particles are masked
out of $\varphi$ and their own update is zeroed. The output layer of
$\varphi$ is zero-initialised, so the layer starts as the identity map.

## 9. Lifting 4-momenta: the bivector map

A single 4-momentum lives in the $(\tfrac12,\tfrac12)$ representation, not in
$\mathbb C^3$; placing $(E,\vec p)$ into $\mathbb C^3$ componentwise is not
covariant. The $(1,0)$ representation — $\mathbb C^3$ with $SO(3,\mathbb C)$ —
is the self-dual part of an antisymmetric tensor, whose natural source is a
pair of 4-vectors (`bivector_lift`):

$$z=\operatorname{bivec}(p,q)=\underbrace{(E_p\,\vec q-E_q\,\vec p)}_{\text{"electric"}}\;+\;i\,\underbrace{(\vec p\times\vec q)}_{\text{"magnetic"}} .$$

If both 4-vectors are transformed by $\Lambda=$ `lorentz_matrix(rho, beta)`,
then $z\mapsto Qz$ with $Q=$ `expm_so3c(rho + 1j*beta)` and the **same**
$(\rho,\beta)$. `random_lorentz_pair` returns such a matched pair; on 64 random
pairs the largest deviation was $8.9\times10^{-15}$ (relative
$6.5\times10^{-16}$).

**Jet lift** (`jet_bivectors`): $z_a=\operatorname{bivec}(p_a,P)$ with
$P=\sum_ap_a$ over real constituents. It is $O(K)$, exactly covariant and
permutation-equivariant; padding produces $z=0$.

**Structural zeros.** Three quantities vanish identically on this lift:

- $\operatorname{Im}(z\cdot z)=2E\cdot B=0$ for any simple bivector $p\wedge q$
  (measured $1.8\times10^{-15}$).
- $\operatorname{Im}(z_a\cdot z_b)\propto\varepsilon(p_a,P,p_b,P)=0$ for two
  bivectors sharing the leg $P$.
- $\sum_a\operatorname{bivec}(p_a,P)=\operatorname{bivec}(P,P)=0$, because the
  lift is linear in its first argument; the jet-total invariant is zero.

So parity-odd information does not survive the lift, and before any flow 6 of
the 11 pooled readout features (§13) are identically zero. The models that
read the raw lift (`so3c_invariant_set`) use only the other 5; a flow that
mixes particles makes the states non-simple and populates all 11.
$\operatorname{Re}(z_a\cdot z_b)$ is a Lorentz-invariant polynomial in the
Minkowski products of $p_a$, $p_b$ and $P$.

## 10. Covariant message passing (`so3c_message_set`)

The lift is first bounded by an invariant scale,
$z_a\leftarrow z_a/(1+\lvert z_a\cdot z_a\rvert^{1/2})$; channels start as complex
multiples $z_a^{(c)}=w_cz_a$; the scalar state $h_a\in\mathbb R^D$ is
initialised by `Linear(3→D)` on
$\operatorname{asinh}(\operatorname{Re}z_a\cdot z_a,\ \operatorname{Im}z_a\cdot z_a,\ \langle p_a,p_a\rangle)$.
Each round $r$ then does, for particles $a,b$ (all pairs, or the $k$ nearest):

$$\begin{aligned}
e_{ab}&=\tanh\!\big(W_r[\,\operatorname{asinh}(\operatorname{Re},\operatorname{Im})\ \text{of}\ s^{(c)}_{ab},s^{(c)}_{aa},s^{(c)}_{bb}\ \text{for all}\ c;\ h_a;\ h_b\,]\big)\\
h_a&\leftarrow h_a+\mathrm{node}_r\big([\,h_a,\ \tfrac1n\textstyle\sum_b \mathrm{msg}_r(e_{ab})\,]\big)\\
r_a^{(c)}&=\tfrac1n\textstyle\sum_b w_r^{(c)}(e_{ab})\,z_b^{(c)},\qquad a_a^{(c)}=\dfrac{z_a^{(c)}\times r_a^{(c)}}{1+\lvert (z_a^{(c)}\times r_a^{(c)})\cdot(z_a^{(c)}\times r_a^{(c)})\rvert^{1/2}}\\
z_a^{(c)}&\leftarrow \exp(-T[a_a^{(c)}]_\times)\,z_a^{(c)},\qquad z^{(c)}\leftarrow\textstyle\sum_d M_r^{cd}\,z^{(d)} .
\end{aligned}$$

The edge network is shared across channels and takes all channels'
invariants at once ($6C+2D$ inputs); the weight head $w_r$ is zero-initialised
(identity flow at step 0) and the complex channel-mixing matrices $M_r$ start
at the identity. Equivariance holds round by round, hence end to end:
edge features are invariants or scalars $h$; an invariant-weighted sum of
covariant vectors is covariant; the cross product is covariant; the exponential
conjugates as in §5; and $M_r$ acts on the channel index while $Q$ acts on the
vector index, so they commute. With excited weight heads the logits changed by
at most $4.2\times10^{-14}$ (relative) under Lorentz boosts in float64.

**Why the bound.** The exponential of a complex generator includes boosts, so
the Hermitian norm $\lvert z\rvert$ can grow from round to round. It cannot be
normalised away: the only scalars available are the bilinear invariants, and a
null vector has $z\cdot z=0$ at any magnitude. The invariant soft bound is
applied to the input lift and to every connection instead.

**Sparse neighbours** (`neighbors=k`). Each particle keeps its $k$ partners
with the largest $\lvert\operatorname{Re}(z_a\cdot z_b)\rvert$ on the input
bivectors, self-edges excluded; the graph is built once and reused in every
round. The key is an invariant, so the neighbour set is the same in every
frame. A frame-dependent key (angular distance, $p_T$ order) would reshuffle
edges under a boost and break equivariance with nothing to flag it.

## 11. Beams: what symmetry survives

`beams=True` (`--beams` on the command line) adds two beam 4-vectors,
$b_\pm=E_b\,(1,0,0,\pm1)$, with $E_b$ = `beam_energy` $=1$ in the normalised
units of the input. They are stored in the buffer `beam_p4` and join the set as
two extra nodes, lifted against the same jet momentum as the constituents:
$z_{b_\pm}=\operatorname{bivec}(b_\pm,P)$, where $P$ stays the sum over real
constituents. The beam nodes send and receive messages and are flowed like any
other node. Every node's scalar state is initialised with three more inputs,
$\langle p,b_+\rangle=E_b(E-p_z)$, $\langle p,b_-\rangle=E_b(E+p_z)$ and a beam
label. They give the network each constituent's lab-frame energy and
longitudinal momentum, which no Lorentz invariant of the jet alone contains.
The readout still pools real constituents only, and appends the two beam
nodes' final scalar states and $\operatorname{asinh}\langle P,b_\pm\rangle$
(§13). Beams need the dense graph: combining them with `neighbors` raises
`ValueError`.

Every feature is still an invariant of the whole set $\{p_a,b_+,b_-\}$ and every
update is covariant, so the network is exactly Lorentz-covariant as a function
of the jet **and** the beams: transforming both by the same $\Lambda$ leaves
the logits unchanged. With the beams held at their lab values, the output is
unchanged only under transformations that fix both beams. Such a transformation
fixes $t=(b_++b_-)/2E_b=(1,0,0,0)$, so it is a rotation, and it fixes the beam
axis $(b_+-b_-)/2E_b$, so it is a rotation about that axis. A boost along the
beam axis is not one of them: it rescales $b_\pm$ by $e^{\pm\eta}$. LorentzNet
and PELICAN also feed beam particles, and their outputs have the same residual
symmetry.

Measured in float64 on the headline configuration (`beams=True, channels=8`)
with the weight heads set to random values, on six random jets:

| Transformation of the jet | Beams | Largest absolute logit change |
|---|---|---:|
| random Lorentz transformation, boost scale 0.5 | transformed with the jet | $2.2\times10^{-16}$ |
| rotation about the beam axis | fixed | $1.4\times10^{-16}$ |
| rotation about a transverse axis | fixed | $2.2\times10^{-2}$ |
| boost along the beam axis, rapidity 0.5 | fixed | $4.8\times10^{-3}$ |
| random Lorentz transformation, boost scale 0.5 | fixed | $3.1\times10^{-2}$ |

On real test jets a trained model keeps its AUC (0.9684) with the beams moved
along up to boost scale 3, and loses 0.035 by scale 3 with them fixed
([experiments.md, table E](experiments.md#e-boost-robustness)).

## 12. Switches implemented but not yet trained

Six arguments of `SO3CMessageSetClassifier` change what the model computes.
Each defaults to the behaviour the results in [experiments.md](experiments.md)
were measured with, and none has been trained on the full data yet. Each keeps
exact covariance: it adds or recompresses invariants, or updates covariant
vectors by invariant-weighted linear combinations.

- **`mass_input=False`** drops $m^2=\langle p_a,p_a\rangle$ from the scalar
  initialisation and the three $m^2$ moments from the Minkowski readout
  statistics (§13). The constituents in this dataset are massless, so their
  $m^2$ is floating-point rounding noise.
- **`self_edges=False`** removes $a=b$ from the dense graph and averages
  messages over the other nodes. The flow is unaffected ($z_a\times z_a=0$);
  the scalar messages and the normalisation are not.
- **`relnorm_edge=True`** adds $d_{ab}=s_{aa}+s_{bb}-2s_{ab}$, real and
  imaginary parts per channel, to the edge features: the analogue of
  LorentzNet's $\lVert x_i-x_j\rVert^2$, which cannot be recovered once
  $s_{aa}$, $s_{bb}$ and $s_{ab}$ have been compressed separately.
- **`falpha=n`** compresses the pair invariants with $n$ learnable signed
  functions $f_\alpha(x)=\operatorname{sign}(x)\,\big((1+\lvert x\rvert)^{\alpha^2}-1\big)/\alpha^2$
  instead of asinh, with $\alpha$ initialised log-uniformly over $[0.05,0.5]$,
  after the input embedding of PELICAN. The diagonal invariants keep asinh.
- **`vector_channel=True`** carries a real 4-vector $v_a$ per node, initialised
  to $p_a$ (beams included). The bivector lift is unchanged under
  $p_a\mapsto p_a+\lambda P$, so it discards each constituent's component along
  the jet axis; $v_a$ keeps it. Each round adds
  $\operatorname{asinh}\langle v_a,v_b\rangle$ and
  $\operatorname{asinh}\langle v_a-v_b,v_a-v_b\rangle$ to the edge features and
  updates $v_a\leftarrow v_a+\frac1n\sum_b u_{ab}v_b$ with an invariant weight
  $u_{ab}$ from a zero-initialised head. The readout adds $\langle V,V\rangle$,
  the mean and the maximum of $\langle v_a,V\rangle$ and the mean of
  $\langle v_a,v_a\rangle$, with $V=\sum_a v_a$ over real constituents, and
  $\langle V,b_\pm\rangle$ when there are beams.
- **`pair_latent=`$C_p$** carries a real pair state $E_{ab}\in\mathbb R^{C_p}$
  through the rounds, as PELICAN does, instead of reducing pairs to nodes in
  every round. It is initialised from the first round's edge features, feeds
  every edge network, and is updated residually from 7 of PELICAN's 15
  permutation-equivariant rank-2 maps (identity, transpose, row mean, column
  mean, the diagonal broadcast along rows and along columns, global mean)
  together with that round's edge hidden state. The readout adds the mean of
  $E$ over real pairs and over its diagonal. $E$ is built from invariants only.

`vector_channel` and `pair_latent` also need the dense graph.

## 13. Readout features

All set models end in a ReLU MLP `in → hidden → hidden → 2` on arcsinh-compressed
invariants (`so3c_message_set` can add dropout between the layers). With $n$
real particles and $n_\text{off}$ ordered off-diagonal real pairs:

| Group | Features | Count |
|---|---|---|
| Pooled bivector invariants (`simple=False`) | mean $\operatorname{Re}s_{aa}$, mean $\lvert\operatorname{Re}s_{aa}\rvert$, mean $\operatorname{Im}s_{aa}$, mean / mean square / max-abs of off-diagonal $\operatorname{Re}s_{ab}$ and of $\operatorname{Im}s_{ab}$, $\operatorname{Re}q_\text{tot}$, $\operatorname{Im}q_\text{tot}$ | 11 |
| Pooled, raw lift (`simple=True`) | the five $\operatorname{Re}$ features above that are not identically zero | 5 |
| Minkowski statistics | mean $m^2$, mean $m^4$, mean $\lvert m^2\rvert$, mean / mean square / max-abs of off-diagonal $\langle p_a,p_b\rangle$, $\langle P,P\rangle$ | 7, or 4 with `mass_input=False` |
| Cross-channel | $\operatorname{Re}$ and $\operatorname{Im}$ of $z^{(c)}_\text{tot}\cdot z^{(d)}_\text{tot}$ for $c\le d$, $z^{(c)}_\text{tot}=\sum_a z^{(c)}_a$ | $C(C+1)$ |
| Scalar channel (`so3c_message_set`) | masked mean and max of $h$ over real constituents | $2D$ |
| Beams | final scalar states of the two beam nodes; $\operatorname{asinh}\langle P,b_\pm\rangle$ | $2D+2$ |
| Vector channel | $\langle V,V\rangle$; mean and max of $\langle v_a,V\rangle$; mean of $\langle v_a,v_a\rangle$; with beams also $\langle V,b_\pm\rangle$ | 4, or 6 with beams |
| Pair latent | mean of $E$ over real pairs and over its diagonal | $2C_p$ |

## 14. Numerical precision

In the frame the data were recorded in, float32 is adequate: the float32
tests hold group membership, conservation, the lift correspondence and the
logit invariance of `so3c_invariant_set` to $10^{-5}$, and the GPU runs in
[experiments.md](experiments.md) used float32. Evaluating on boosted jets is
different. The lift is quadratic in the momenta, so a boost scale of 2
inflates input magnitudes by about $e^2\approx7.4$ and lifted magnitudes by
about 55. The note in
[`benchmarks/run_boost_robustness.py`](../../benchmarks/run_boost_robustness.py)
records a drop of 0.0232 AUC in float32 against 0.0001 in float64 for the same
weights and jets, which is why that diagnostic defaults to float64.
