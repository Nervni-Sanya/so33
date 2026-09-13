# Gap to SOTA -- workflow findings, 2026-09-13

Raw agent output from the `so3c-gap-to-sota` workflow (run `wf_4b9f21c4-439`), written out in full so it does not live only in a temp directory. The machine-readable copy is `gap_to_sota_2026-09-13.json` beside this file.

## Read this first

- **Unverified agent output.** Nothing below has been checked by a second agent or re-derived, except where a claim matches something already verified in this repo (noted inline).
- **Incomplete.** Three of six agents failed on the account usage limit: `our-error-profile` (where our model loses, binned by jet mass / pT / multiplicity), `synthesise` (the ranked plan) and `critique` (the adversarial pass). So there is **no ranked plan and no critique** here; the findings are unranked.
- **Completed angles:** `pelican-internals`, `lorentznet-and-data`, `beyond-pelican` -- 26 findings in total.
- **Independently verified:** the PELICAN size-sweep numbers quoted below (Table 2, arXiv:2307.16506) match the pypdf extraction of 2026-09-11 in `paper/figures/pelican_scaling.csv`: 326 par 0.9801 / 669, 605 par 0.9823 / 901, 1k 0.9835 / 1145, 11k 0.9858 / 1879, 208k 0.9870 / 2250.

Context the agents were given: current best `so3c_message_set` with beams + channels 8, K=64, canonical, 30 epochs, 22834 parameters: AUC 0.98333 +- 0.00010, 1/eps_B at eps_S=0.3 of 1131 +- 22; two-seed ensemble 0.98373 / 1174. PELICAN 0.9870 / 2250 at 208k.

## Angle: `pelican-internals`

### 1. Rank-2 latent state with the 15-element Eq2->2 aggregator basis (the dominant carrier)

**mechanism.** PELICAN never collapses the pair index. Its hidden state is T_ij^c of shape [B, N, N, C], and each of the L blocks applies a channel-MLP followed by all 15 linear permutation-equivariant maps from rank-2 arrays to rank-2 arrays (Eq. 2.6, basis pictured in Figure 1). The 15 split as: 5 'order zero' maps that do no aggregation at all (identity, transpose, and 3 ways of embedding the diagonal T_ii back into the square array) -- the paper calls these 'permutation-equivariant skip-connections'; 8 'order one' maps that aggregate over N components (row sums, column sums, or the diagonal sum) and re-embed the resulting vector into the output array in every equivariant way; and 2 'order two' maps that aggregate over all N^2 components. The 15 x C_in outputs are then mixed down to C_out by a dense layer. The key point is that information flows pair -> pair through L rounds, so the network can represent functions of triples and quadruples of constituents that a rank-1 state cannot reach in the same depth.

**evidence.** Table 2 (p.11) is a depth/width study on the identical Kasieczka benchmark and is the single most damaging number for our current design. L=1, width 6/3, 326 parameters: AUC 0.9801, 1/eps_B = 669 +- 41. L=2, 605 parameters: 0.9823, 901 +- 59. L=3, width 6/4, 1k parameters: 0.9835, 1145 +- 74. L=5, width 25/15, 11k parameters: 0.9858, 1879 +- 103. L=5, 208k: 0.9870, 2250. Read against our numbers: our 13,862-parameter message-passing model (0.98073, 850) is matched by a 605-parameter PELICAN, and our 22,834-parameter beams+channels model (0.98333, 1131) is matched by a 1,000-parameter PELICAN. That is a 23x parameter deficit at equal accuracy. Table 2 also shows that going 1k -> 208k parameters only buys +0.0035 AUC, i.e. width is worth far less than the rank-2 structure. This is consistent with our own measurement that capacity without beams was flat -- capacity is not our bottleneck, state rank is.

**source.** arXiv:2307.16506v2, Sec 2.4 pp.6-7 (Eq. 2.6, Figure 1), Sec 3.2 pp.9-10 (Figure 2, Eq. 3.1), Table 2 p.11.

**applies to us.** No analogue. Our state is rank 1: per-particle z_a plus one invariant scalar channel. We recompute z_a.z_b every round and immediately reduce it to node level via sum_b w_b z_b. In PELICAN's nomenclature we apply an Eq2->1 collapse at every single round and never carry a rank-2 latent. Of their 8 order-one aggregators we implement essentially one (the row mean); we have none of the 5 order-zero skip maps, none of the diagonal re-embeddings, and neither order-two map. Our covariant flow z_a <- exp(-T[z_a x sum_b w_b z_b]_x) z_a has no PELICAN counterpart and may already supply some of this expressivity, which is the main reason the transplant could fail.

**smallest experiment.** Carry an edge tensor E_ab^c of shape [B,64,64,C_edge] with C_edge = 6-8, initialised from the embedded invariants Re(z_a.z_b), Im(z_a.z_b). Each of the 3 existing rounds: 2-layer channel MLP on E, then a reduced aggregator set -- start with 7 of the 15 (identity, transpose, diagonal-to-diagonal, diagonal-broadcast, row mean, column mean, global mean), each scaled by N^alpha/Nbar^alpha with learnable alpha. Mix 7 x C_edge down to C_edge with an UNFACTORIZED dense layer (see the factorization finding). Keep the existing covariant z-flow untouched and feed w_b from a row-pool of E. Memory at batch 100: 100 x 4096 x 8 x 4 B = 13 MB per stored activation, trivial on a P100.

**expected gain.** This is the only change in the list with the headroom to close the 1131 -> 2250 gap. If the rank-2 latent is genuinely orthogonal to our covariant flow, PELICAN's own depth curve suggests the 1400-1800 rejection band at 3 rounds. If it merely duplicates what the flow already computes, expect < +0.0005 AUC. Bimodal outcome, which is exactly why it should be screened cheaply before a full run.

**test cost gpu hours.** Screen at 20% of the training set (242k jets), K=64, 30 epochs, 3 seeds. The new block roughly doubles the pairwise cost, so ~1.8x epoch time: 1272 x 0.2 x 30 x 1.8 / 3600 = 3.8 h per seed, 11.4 h for 3 seeds. Paired baseline at the same budget costs 6.4 h and is reusable for every other experiment here. Go/no-go for ~18 GPU-h, i.e. under one week. If positive, one full-protocol confirmation run at K=64, 30 epochs is 19.1 h -- the following week's entire budget.

**confidence.** high that this is what carries PELICAN's performance; medium that it transplants onto our covariant-flow model

### 2. The f_alpha input embedding: a bank of learnable Box-Cox exponents on the dot products

**mechanism.** Before anything else, every dot product d_ij is passed through f_alpha(x) = ((1+x)^(alpha^2) - 1)/alpha^2 for several values of a TRAINABLE alpha, initialised to span the interval [0.05, 0.5]. Because the exponent is alpha^2, the actual powers are in [0.0025, 0.25] -- these are strongly compressive, and the alpha -> 0 limit is exactly log(1+x). So the network sees the same dot product simultaneously at several resolutions: the small-alpha channels are near-logarithmic and resolve the soft/collinear regime where d_ij -> 0, while the larger-alpha channels keep linear sensitivity in the hard core. The paper independently confirms the dynamic range this is fighting: it states that the energy weights on realistic data 'span up to 8 orders of magnitude' (Sec 10.3, p.32), which is why no single fixed normalisation of d_ij can work.

**evidence.** Sec 3.1 p.9 gives the formula and the [0.05, 0.5] initialisation range verbatim. The indirect but strong evidence is Table 2 p.11: the 326-parameter, depth-1 model still reaches 1/eps_B = 669 +- 41 and AUC 0.9801 -- matching our 9,078-parameter covariant-flow model (0.9772, 638) with 28x fewer parameters. A 326-parameter network has essentially no capacity to learn a range compression internally, so whatever it is doing well is being done by f_alpha and the aggregator basis, not by the MLP.

**source.** arXiv:2307.16506v2, Sec 3.1 'Embedding of dot products' p.9; dynamic-range remark Sec 10.3 p.32; Table 2 p.11.

**applies to us.** Partial analogue at best. We feed bilinear invariants z_a.z_b into edge features with a single fixed scaling. We have no multi-resolution bank and no learnable compression exponent. This is the cheapest possible fix in the entire list: about 8-16 scalar parameters and no measurable FLOPs.

**smallest experiment.** Replace our current invariant featurisation with the concatenation [f_a1(x), ..., f_a8(x)] applied separately to Re(z_a.z_b) and Im(z_a.z_b), alphas trainable, geomspaced-initialised over [0.05, 0.5]. Our invariants can be negative (unlike PELICAN's d_ij on physical momenta), so use the signed extension f_alpha(x) = sign(x)((1+|x|)^(alpha^2) - 1)/alpha^2, which is C^1 at 0 and reduces to theirs for x > 0. Bundle this with the three other near-free changes below into a single arm.

**expected gain.** +0.0005 to +0.0015 AUC, roughly +50 to +150 on 1/eps_B. Highest expected return per GPU-hour in the whole list, because it costs nothing to run and attacks a failure mode (unnormalised heavy-tailed inputs) that extra channels cannot fix -- which is a plausible explanation for our measured 'capacity was flat' result.

**test cost gpu hours.** Zero added runtime. Screened as part of one bundled arm: 20% of training data, K=64, 30 epochs, 3 seeds = 1272 x 0.2 x 30 x 3 / 3600 = 6.4 h. Full-protocol confirmation if positive: 10.6 h.

**confidence.** high

### 3. Masked batch normalisation, so variable multiplicity does not leak into the normalisation statistics

**mechanism.** The message block is Dense + LeakyReLU + BatchNorm2D, where BatchNorm2D normalises over the first three dimensions (B, N_max, N_max) per channel, followed by a per-channel affine. Critically it is a masked implementation: a binary [B, N_max, N_max] mask excludes the zero padding from both the BN statistics and the aggregation means. The reason this matters is specific and not generic hygiene: N varies from ~10 to 200 across jets, so the padding fraction is a large and strongly class-correlated quantity (top jets have higher constituent multiplicity than QCD jets). An unmasked BN would therefore compute means and variances that are themselves a function of the label, which both leaks and destabilises. Note also the regularisation levels they actually use alongside it -- dropout 0.025 and AdamW weight decay 0.005 -- an order of magnitude milder than the dropout 0.2 / wd 0.01 recipe we measured at 0.97769.

**evidence.** Sec 3.2 p.9: 'we use a masked implementation of batch normalization so that the variable particle number is respected', and the mask is applied to 'operations like BatchNorm and aggregation'. Training settings in Sec 4.2 p.11: dropout 0.025, AdamW weight decay 0.005, 35 epochs, 4 warm-up epochs to lr 1e-3, 28 epochs CosineAnnealingLR with T_0 = 4 and T_mult = 2, then 3 epochs of exponential decay with gamma = 0.5, batch size 100.

**source.** arXiv:2307.16506v2, Sec 3.2 p.9; Sec 4.2 pp.10-11.

**applies to us.** Unknown until audited, and the audit is free. We mask padding in the pairwise sums, but the question is whether our per-particle scalar channel and our edge features are normalised with mask-aware statistics or with a plain BN/LayerNorm over the padded [B,64,...] tensor. If the latter, every normalisation layer in our network is currently reading multiplicity. This is also the most likely explanation for why the LorentzNet/PELICAN training recipe cost us 0.006 AUC: with leaky normalisation, adding strong weight decay and dropout on top attacks the wrong problem.

**smallest experiment.** Step 1 (free, do first): grep our normalisation layers and check whether the mask is passed. Step 2, only if it is not: masked BN over the (B,K) node tensor and (B,K,K) edge tensor, one screening arm. Step 3, only if masked BN lands positive: re-test PELICAN's exact mild regularisation (dropout 0.025, wd 0.005) on top of it -- our earlier negative result was at dropout 0.2 / wd 0.01, which is 8x and 2x their setting and does not refute theirs.

**expected gain.** Zero if we already mask correctly. Up to +0.001 AUC if we do not. The retest of mild regularisation on top is worth a further +0.0003 and is what let PELICAN train 35 epochs without underfitting, where our 35-epoch run gained only +0.00015.

**test cost gpu hours.** Audit: 0. Masked-BN arm at 20% data, K=64, 30 epochs, 3 seeds: 6.4 h. Mild-regularisation retest, same budget: 6.4 h. Both fit in one 30 h week together with the reusable baseline.

**confidence.** medium

### 4. Learnable N^alpha/Nbar^alpha rescaling on every aggregator -- how PELICAN handles variable N

**mechanism.** Every aggregator computes a masked MEAN over its subset, then multiplies by N^alpha / Nbar^alpha where alpha is a separate trainable exponent per aggregator (initialised uniform on [0,1]), N is that event's constituent count, and Nbar is a fixed hyperparameter equal to the typical constituent count in the dataset. alpha = 0 recovers mean-pooling, alpha = 1 recovers sum-pooling, and the network interpolates by gradient descent independently for each of the 15 aggregators and each channel. The paper explicitly notes that 'combining multiple aggregators is known to boost accuracy'. This single mechanism is the entirety of how the architecture handles variable particle number -- there is no other multiplicity handling anywhere.

**evidence.** Sec 2.4 p.7 defines S_a as 'the mean of its inputs followed by an additional scaling by a factor of N^alpha_a / Nbar^alpha_a with learnable exponents alpha_a'; Sec 3.2 pp.9-10 repeats it for the Eq2->2 block with the [0,1] initialisation and the alpha = 1 -> sum remark. The sensitivity is real: Sec 10.2 p.33 notes that because N is not IR-safe, the IRC-safe variant cannot use means at all and must substitute a Lorentz-invariant Soft Drop multiplicity n_SD in the aggregators.

**source.** arXiv:2307.16506v2, Sec 2.4 p.7; Sec 3.2 pp.9-10; Sec 10.2 p.33.

**applies to us.** No analogue -- we pool with one fixed convention. This bites us harder than it bites PELICAN because of our covariant update: z_a <- exp(-T [z_a x sum_b w_b z_b]_x) z_a. The magnitude of sum_b w_b z_b directly sets the rotation angle, so with sum-pooling an 80-constituent jet rotates roughly twice as far as a 40-constituent jet at identical physics, and with mean-pooling it rotates identically. Neither is obviously correct and we are currently asserting one of them by hand. Our K=64 truncation additionally makes N a censored variable, so the effective N our pooling sees is min(N, 64).

**smallest experiment.** Add one learnable scalar alpha per pooling operation -- there are roughly 5 in our model (the message aggregation sum_b w_b z_b, the scalar-channel aggregation, and the readout pools) -- replacing each sum with mean x (N/Nbar)^alpha, Nbar set to the median post-truncation constituent count (~40). Five extra parameters, zero extra FLOPs. Bundle into the same near-free arm as f_alpha.

**expected gain.** +0.0002 to +0.0008 AUC. Small but free, and it has a second-order benefit: the learned alpha values tell us empirically whether our flow wants sum or mean semantics, which is diagnostic information we currently do not have.

**test cost gpu hours.** Zero added runtime; rides in the bundled 6.4 h arm at 20% data, K=64, 30 epochs, 3 seeds.

**confidence.** high

### 5. The Eq2->0 readout has TWO aggregators -- trace and total sum -- not one pooled quantity

**mechanism.** The classifier head is an Eq2->0 layer that is 'otherwise identical to the equivariant layer' (same message MLP, same masked BN) but uses just 2 aggregation functions: the trace of the rank-2 array and its total sum. Dropout, then one linear layer to the class logits (Eq. 3.2, p.10). The trace is not redundant with the sum: the diagonal entries d_ii = p_i . p_i are per-particle invariants (masses on the input layer, learned per-particle summaries after L blocks), so the readout hands the classifier a particle-pooled stream and a pair-pooled stream side by side. The same slot swaps to Eq2->1 (5 aggregators: diagonal, row sums, column sums, trace, full sum) for 4-vector regression, per Eq. 3.4 p.11, which is what makes the architecture dual-use.

**evidence.** Sec 3.3 p.10: 'This layer involves just 2 aggregation functions instead of 15 - the trace and the total sum of the input square matrix'. Sec 3.2 p.10 gives the aggregator counts for the sibling blocks: Eq1->2, Eq2->1, and Eq2->0 involve 'just 5, 5, and 2 aggregators, respectively'.

**source.** arXiv:2307.16506v2, Sec 3.2 p.10; Sec 3.3 pp.10-11 (Eq. 3.2 and Eq. 3.4).

**applies to us.** Half an analogue. Our readout pools bilinear invariants over constituents, which is their 'total sum' aggregator. We do not carry the diagonal as a separate pooled stream, and more importantly we never propagate it -- 3 of their 5 order-zero aggregators exist purely to re-embed the diagonal into the off-diagonal block, and 3 of the 8 order-one aggregators spread the diagonal sum. For us the diagonal is z_a . z_a, a per-particle invariant of the lifted bivector, which is genuinely different information from the off-diagonal z_a . z_b.

**smallest experiment.** Split our readout into three concatenated streams, each with its own learnable N^alpha/Nbar^alpha exponent: sum over a != b of the edge features, sum over a of the diagonal z_a.z_a features, and the existing per-particle scalar pool. Roughly 30 extra parameters, no measurable time cost. Bundle with f_alpha and the learnable-alpha pooling.

**expected gain.** +0.0002 to +0.0005 AUC. The mechanism is concrete -- the jet mass and the per-constituent mass spectrum enter the classifier on their own channel instead of being averaged into the pair sum -- but the effect is small because a 3-round message-passing model can partially reconstruct it.

**test cost gpu hours.** Zero added runtime; rides in the bundled 6.4 h arm.

**confidence.** medium

### 6. Do NOT factorise the aggregator-to-channel mixing at our widths -- the paper says it hurts small models

**mechanism.** The dense layer that mixes C_in x 15 aggregator outputs down to C_out is large, so PELICAN factorises the weight tensor W_abc (a = input channel, b = basis index 1..15, c = output channel) as W_ab^0 W_ac^1 + W_cb^2 W_ac^3 (Eq. 3.1, p.10): the first term mixes the 15 aggregators per output channel then mixes channels, the second mixes the 15 per input channel then mixes channels. The paper then states the limit of this trick explicitly.

**evidence.** Sec 3.2 p.10, verbatim consequence: the factorised network performs as well as the unfactorised one 'except at very low network widths, in which case the unfactorized network performs better and may even have fewer parameters'. Our target width for a rank-2 block is C_edge = 6-8, squarely in the 'very low' regime where they say factorisation is the wrong choice.

**source.** arXiv:2307.16506v2, Sec 3.2 p.10, Eq. 3.1.

**applies to us.** Prospective, not retrospective -- it is a design instruction for the rank-2 block proposed above, not a gap in our current model. With 7 aggregators and C_edge = 8 the unfactorised mix is 7 x 8 x 8 = 448 parameters, which is affordable outright; factorising would cost more implementation complexity for a documented performance loss at this width.

**smallest experiment.** None needed on its own -- just build the block unfactorised. If the rank-2 arm succeeds and we later scale C_edge past ~32, revisit Eq. 3.1 then.

**expected gain.** Avoids a known loss rather than producing a gain. Worth stating because it is the kind of implementation detail that gets copied blindly from a reference implementation tuned at width 132.

**test cost gpu hours.** 0 -- design decision, no separate run.

**confidence.** high

### 7. Constituent cap: their N=80 saturation says stop spending budget on K

**mechanism.** PELICAN caps the constituent list at 80 out of the 200 stored in the dataset and reports that nothing is gained beyond that. Since we already use 64 of up to 200 and our cost grows as K^2, this bounds how much our truncation can possibly be costing us and removes a whole expensive direction from consideration.

**evidence.** Sec 4.2 p.11, verbatim parenthetical: 'The number of jet constituents was capped at 80 (no noticeable performance gain was seen beyond that number)'. This is at the 208k-parameter, L=5, 2250-rejection operating point -- i.e. even the state-of-the-art configuration saturates at 80. Their per-batch timings at that cap are also worth calibrating against: 0.43 s/batch for the 208k model and 0.08 s/batch for the 11k model on an H100 at batch size 100, so 12,110 batches/epoch = 5,207 s and 969 s per epoch respectively. Our 1,272 s/epoch on a P100 is in the same order as their 11k model on an H100, meaning we are already spending PELICAN-class compute and getting 1131 instead of 1879.

**source.** arXiv:2307.16506v2, Sec 4.2 pp.10-11.

**applies to us.** Direct. It tells us our K=64 truncation is nearly free and that a K=80 run is near-certainly a waste. It also implies a faithful full-size PELICAN reimplementation is out of reach for us: scaling their 208k model's 5,207 s/epoch by the ~7x fp32 gap between H100 and P100 gives ~10 h per epoch. Even their 11k model would be ~1.9 h/epoch, 68 h for 35 epochs. Only the small end of Table 2 (L=3, width 6/4, 1k parameters, 1145 +- 74) is affordable to reproduce.

**smallest experiment.** Explicitly decline the K=80 experiment: 1272 x (80/64)^2 = 1,988 s/epoch, 30 epochs = 16.6 GPU-h, for a gain the paper predicts to be ~zero. Redirect that 16.6 h to the rank-2 arm. Optional high-value alternative for the same money: implement the Eq2->2 block standalone and reproduce Table 2's L=3, width 6/4, 1k-parameter row (target 0.9835 / 1145) as a reference point in our own pipeline -- estimated ~16-17 h on a P100 for 35 epochs, and it would tell us unambiguously how much of the 2250 is aggregation structure versus everything else in their stack.

**expected gain.** Saves 16.6 GPU-h of certain waste. The optional reproduction produces no new AUC but converts the architecture question from inference to measurement.

**test cost gpu hours.** K=80 test: 16.6 h, recommended AGAINST. Table 2 L=3 reproduction: ~16.5 h, one week's budget, optional.

**confidence.** high

### 8. Jet-frame de-dimensionalised dot products d_hat_ij = 1 - cos(Theta_ij) as an extra input channel (take the conditioning, skip the IRC constraint)

**mechanism.** For the IRC-safe variant PELICAN defines jet-frame energies E_i = (p_i . J)/m_J where J = sum_i p_i is the jet total (Eq. 10.4), weights z_i = E_i / sum_j E_j, and rescaled dot products d_hat_ij = d_ij/(E_i E_j) (Eq. 10.5). Footnote 4 on p.31 gives the payoff: for massless inputs d_hat_ij = 1 - cos(Theta_ij), the pairwise spatial angle in the jet frame. That is a bounded quantity in [0,2], perfectly conditioned, where the raw d_ij spans many orders of magnitude. The IRC-safe architecture additionally inserts the z_i weights into all 8 order-one and both order-two aggregators, which is where the expressivity is lost.

**evidence.** Table 1 p.12 prices the full IRC-safe construction at AUC 0.9870 -> 0.9844 and 1/eps_B 2250 +- 75 -> 1711 +- 208 on top tagging, at the same 208k parameters. So full IRC-safety is a clear net loss for tagging and should not be adopted. But the input rescaling of Eq. 10.5 is separable from the weighted aggregators, and the loss is attributable to the latter. Supporting note on q/g tagging, Sec 10.3 p.34: there the IRC-safe classifier performs 'almost as well' (0.9059 -> 0.8955), so the penalty is task-specific and top-tagging-specific.

**source.** arXiv:2307.16506v2, Sec 10.2 pp.31-32 (Eq. 10.4, Eq. 10.5, footnote 4); Table 1 p.12; Sec 10.3 p.34.

**applies to us.** We already have the required reference vector -- P, the jet total, is exactly their J, and it is already what our bivector lift z_a = bivec(p_a, P) is built against. So E_a = (p_a . P)/m_P costs one dot product we very likely already compute. We do not currently feed any angle-like bounded invariant; our edge features are raw bilinears.

**smallest experiment.** Append d_hat_ab = (z_a . z_b)/(E_a E_b) as two extra edge channels (real and imaginary parts) ALONGSIDE the existing raw invariants, not replacing them, and do NOT touch the aggregators. Two extra input channels, negligible cost. Bundle into the same near-free arm as f_alpha, the learnable-alpha pooling, and the split readout.

**expected gain.** +0.0002 to +0.0006 AUC. Partially overlapping with the f_alpha finding -- both attack input conditioning -- so if they are bundled and the arm wins, a follow-up ablation is needed to attribute. Explicitly do not adopt the weighted aggregators: Table 1 prices that at -539 rejection.

**test cost gpu hours.** Zero added runtime; rides in the bundled 6.4 h arm.

**confidence.** medium

### 9. Validated fraction-of-data screening protocol on this exact benchmark -- and the floor below which screening goes blind

**mechanism.** PELICAN trained all three widths on 0.5%, 1%, 5% and 100% of the training set with a stretched schedule (70 epochs, 60 of CosineAnnealingLR instead of 28, 6 of exponential decay instead of 3) and reports the full grid. This gives us empirical, same-benchmark calibration for how small a training fraction still separates architectures -- which is the binding question for a 30 GPU-h/week budget.

**evidence.** Table 3 p.14. At 5%: 132/78 gets 1270 +- 65, 60/35 gets 1148 +- 49, 25/15 gets 1111 +- 108 -- the ranking survives and the relative uncertainty on 1/eps_B is 5-10%. At 1%: 789 +- 49, 799 +- 52, 798 +- 116 -- all three widths are inside each other's error bars and the ordering is destroyed. At 0.5%: 633, 637, 615 -- pure noise. The paper's own reading (p.13): at low data 'the differences in performance between models of different width become much less significant', and larger networks 'benefit from seeing a larger training dataset'. Corollary for us: screen at 5-20%, never at 1%, and expect a 5-20% screen to resolve effects larger than about 15% in rejection but not smaller ones.

**source.** arXiv:2307.16506v2, Sec 4.3 p.13; Table 3 p.14.

**applies to us.** Directly, as budget policy rather than architecture. Our full protocol is 1,272 s/epoch x 30 epochs = 10.6 h, so a 30 h week buys under 3 full runs and no seed averaging -- yet our own quoted error bar is +- 0.00010 AUC and +- 22 on rejection, which needs multiple seeds to establish. Fraction screening is the only way to run 3 seeds per arm.

**smallest experiment.** Adopt: phi = 0.20 (242k jets), K=64, 30 epochs, 3 seeds as the standard screening arm = 1272 x 0.2 x 30 x 3 / 3600 = 6.4 h. Run the unchanged baseline once at this budget (6.4 h) and reuse it as the control for every subsequent arm -- that converts a 30 h week from 2 paired comparisons into 3-4 unpaired ones against a fixed control. Recommended first week: baseline (6.4 h) + bundled near-free pack of f_alpha / learnable-alpha pooling / split readout / d_hat channel (6.4 h) + masked-BN audit and arm (6.4 h) = 19.2 h, leaving ~10 h slack. Second week: the rank-2 arm (11.4 h) plus, if it wins, the start of a full-protocol confirmation.

**expected gain.** No AUC gain of its own; roughly triples the number of hypotheses testable per week and makes the results seed-averaged rather than single-shot. Given that our current best result is a +- 0.00010 AUC measurement, this is the difference between measuring effects and guessing at them.

**test cost gpu hours.** 6.4 h per screening arm, 3 seeds included; the reusable baseline is a one-time 6.4 h.

**confidence.** high

## Angle: `lorentznet-and-data`

### 10. The K lever is 97% exhausted at K=64 — cancel the K=128 run (saves ~36 GPU-h)

**mechanism.** Top tagging is dominated by the jet invariant mass peak at m_top=173 GeV. Truncating to the leading-K constituents by pT biases m_jet downward. The relevant question is not 'how many particles exist' but 'how much of m_jet does K recover'. I measured this directly on the full 200-constituent parquet (all 404k test jets, row group 0).

**evidence.** Measured on C:/Users/Пользователь/Documents/so33/data/toptagging/test.parquet (404,000 jets, 806 columns, all 200 constituents): true multiplicity mean 49.0, median 48, p90 71, max 165. Energy captured and jet-mass fidelity vs the full jet:
  K=16 : 85.56% of E, median |dm|/m = 0.2693, 97.4% of jets off by >2%
  K=32 : 95.65% of E, median |dm|/m = 0.0664, 73.6% of jets off by >2%
  K=64 : 99.75% of E, median |dm|/m = 0.0000,  7.3% of jets off by >2%
  K=128: 100.00% of E, median |dm|/m = 0.0000,  0.0% of jets off by >2%
Only 17.8% of jets have more than 64 constituents, and 0.0% have more than 128. Script: C:/Users/73B5~1/AppData/Local/Temp/claude/C--Users--------------Documents-Godot-projects-Jam/3dfd079f-696e-4c56-9481-a7af047ab573/scratchpad/gap/probe3.py

**source.** My own measurement on the shipped parquet. Contradicts the plan recorded in C:/Users/Пользователь/Documents/so33/benchmarks/build_notebooks.py lines 144-146 ('K is the biggest lever we have... every published model at 0.987 uses the full jet') and lines 547-553.

**applies to us.** Directly. The stated plan is to spend quota raising K past 64. At K=64 you already have 99.75% of the jet energy and an exact jet mass for 92.7% of jets. The remaining 0.25% of energy is the softest tail, which is also the most pileup/detector-noise-contaminated. The 0.987-vs-0.983 gap to LorentzNet is NOT a constituent-count gap.

**expected gain.** Negative as a cost avoided: prevents spending 36 GPU-h (more than a full week's budget) on <0.25% of jet energy. Reallocate that quota to the depth and edge-feature items below.

**test cost gpu hours.** 0 — already measured on CPU. Avoiding the K=128 run saves ~36 GPU-h: cost scales 4x on the K^2 terms, 1272 s/epoch -> ~4325 s/epoch (assuming ~80% of the epoch is the (B,K,K,6C+2D) edge MLP and the two KxK matmuls), x30 epochs = 36 h.

**confidence.** high — direct measurement on the exact file, full test split

### 11. Length-bucketed dynamic K: get every constituent at 0.66x the current pairwise cost

**mechanism.** LorentzNet never uses a fixed K. Its collate pads to the batch maximum, then drop_zeros() removes all-padding columns, so the node dimension per batch equals the batch's largest jet; edge_mask = atom_mask (x) atom_mask kills padded pairs. Because cost is K^2 and the multiplicity distribution is tight (mean 49, p90 71), sorting jets by multiplicity into buckets before forming batches makes the per-batch max track the local multiplicity instead of the global one. You then see all 165 constituents of the longest jet while paying less than a fixed K=64.

**evidence.** Paper section 4.1 (page 8): 'Each jet contains an average of 50 particles, and events with less than 200 are zero-padded.' Implementation confirmed in LorentzNet-release top/collate.py: dynamic padding to batch max, drop_zeros(), edge_mask = atom_mask.unsqueeze(1)*atom_mask.unsqueeze(2) with the diagonal removed via ~torch.eye().
My measurement of the cost (probe4.py, 404k test jets, real multiplicities):
  fixed K=64                          : K^2 = 4096
  dynamic, RANDOM batching, B=256     : mean K 106.5, mean K^2 11448  (2.79x worse)
  dynamic, LENGTH-BUCKETED, B=256/512 : mean K  49.1, mean K^2  2710  (0.66x of K=64, 0.17x of K=128)
Random batching makes dynamic K *worse*; the bucketing is the whole trick.

**source.** arXiv:2201.08187 section 4.1 p.8; LorentzNet-release top/collate.py (collate_fn); my measurement in scratchpad/gap/probe4.py

**applies to us.** Directly, and it is safer for us than for LorentzNet. The known hazard of length bucketing is class skew: I measured constituent multiplicity alone to be a classifier with AUC 0.7037 (signal mean 54.8 particles, background 43.3), so buckets are strongly label-imbalanced. LorentzNet would suffer because it uses BatchNorm1d(72) inside phi_x/phi_e/phi_h (section 3.3, page 7) and skewed batches corrupt running statistics. Our so3c_models.py SO3CMessageSetClassifier has no normalization layer at all (Linear+Tanh / Linear+ReLU only), so the skew costs us only gradient noise, not corrupted statistics. Implementation touches _stack_constituent_jets in benchmarks/datasets.py (which currently hard-pads to a fixed K) plus the batch sampler.

**expected gain.** Epoch time 1272 s -> ~930 s (1272 x (0.8x0.66 + 0.2)), i.e. 10.6 h -> 7.7 h per 30-epoch run, a 27% cut to every subsequent experiment. AUC gain from the recovered 0.25% of energy is small on its own (the K=64 row above), but the run is strictly cheaper than the status quo, so the gain is free. Use coarse buckets (e.g. ceilings at 48/64/80/168) rather than a full sort if the label skew worries you; that still lands near 0.7x.

**test cost gpu hours.** ~0.2 to validate (4 epochs on 20% of train at the new cost: 4 x 930 x 0.2 / 3600 = 0.21 h), then it pays for itself on the first full run. Net saving 2.9 GPU-h per 30-epoch run thereafter.

**confidence.** high on the cost arithmetic (measured distribution); medium on the AUC gain

### 12. Answer to Q1b: the public dataset ships nothing usable that you are not already using — and the constituent mass LorentzNet's paper names is pure float32 rounding noise

**mechanism.** The Kasieczka reference file has 806 columns: E/PX/PY/PZ_0..199 (800 = the 4-momenta), truthE/truthPX/truthPY/truthPZ (the MC-truth parton 4-momentum — a regression target, using it as input is label leakage), ttv (the split flag) and is_signal_new (the label). There is no PID, no charge, no vertex, no track/tower type. So the only per-particle scalar derivable is the mass m^2 = E^2 - |p|^2 — and Delphes E-flow constituents in this sample are exactly massless, so that quantity is float32 rounding error.

**evidence.** Schema read from test.parquet: 806 columns, listed above. m^2 of real constituents across 404k jets: mean 1.86e-07, symmetric about zero, 1%/99% quantiles -9.77e-04 / +9.77e-04, min -0.5 (unphysical, i.e. spacelike), max +0.5; 91.5% of constituents have |m^2| < 1e-4. The decisive test is that std(m^2) scales exactly as E^2 — the float32 relative-precision signature:
  E in [0,10)      n=13.5M  std(m2)=2.33e-06
  E in [10,100)    n= 5.5M  std(m2)=1.90e-04
  E in [100,1000)  n=785k   std(m2)=8.07e-03
  E in [1000,10000) n=3108  std(m2)=1.28e-01
Each decade in E multiplies std(m^2) by ~100. Scripts probe3.py / probe5.py in the scratchpad.

**source.** arXiv:2201.08187 section 3.1 p.6 ('the scalars include the mass of the particle (i.e. (E)^2-(px)^2-(py)^2-(pz)^2) or particle identification (PID) information directly if it is available') and section 4.2/Table 2 p.8 (PID one-hot is used only for quark-gluon, the dataset that actually ships particle types); my measurement of the top-tagging parquet.

**applies to us.** Two consequences. (1) Stop looking for a missing input feature — there is none; both we and LorentzNet see only 4-momenta on this benchmark, so the 0.9833-vs-0.9868 gap is entirely architectural. (2) We are actively feeding the noise: benchmarks/so3c_models.py SO3CMessageSetClassifier.forward builds node_inv = [q.real, q.imag, m2] (+3 beam channels) and passes it through h_init. That m2 channel is 1 of 3 (or 1 of 6 with beams) inputs to the scalar-state seeding and carries zero information — and with our null beams (E,0,0,±E) the beams have m^2=0 too, so it does not even act as a beam tag.

**expected gain.** Removing the dead m2 channel: small but nonzero, it is a pure noise injection into h_init at every node. Bundle it with the edge-feature change below rather than testing alone. Main value is the negative result: it closes off 'find a better per-particle scalar' as a direction.

**test cost gpu hours.** 0 to establish (CPU measurement, done). Bundled into the edge-feature probe below at ~0.2 h.

**confidence.** high — the E^2 scaling of std(m^2) is conclusive

### 13. LorentzNet's actual per-particle scalar on top tagging is a single ±1 beam/constituent tag, not the mass the paper describes

**mechanism.** The released top-tagging code prepends two beam nodes and sets the scalars tensor to +1 on the two beam rows and -1 on every real constituent row. That single channel is the entire scalar input s_i to the Linear(scalar_dim, 72) embedding; everything else the network knows comes through the Minkowski invariants inside the LGEB.

**evidence.** LorentzNet-release top/collate.py, collate_fn: with add_beams=True, labels = torch.ones(batch, 2) concatenated with -torch.ones(batch, n_atoms), stored as data['scalars']. The beam 4-momenta are [[sqrt(1+beam_mass**2),0,0,1], [sqrt(1+beam_mass**2),0,0,-1]]; top/dataset.py calls collate_fn(data, scale=1, add_beams=True, beam_mass=1). The paper's section 3.1 text (p.6) instead says the scalars are the particle mass or PID — for top tagging that mass is the rounding noise measured in the previous finding, so the code and the paper are only reconcilable if the beam tag is what actually carries.

**source.** LorentzNet-release top/collate.py and top/dataset.py (github.com/sdogsq/LorentzNet-release, master); paper section 3.1 p.6, section 4.1 p.8

**applies to us.** Confirms we already have the right scalar set and should not go hunting for more. Our node_inv with beams is [Re(z.z), Im(z.z), m2, <p,b+>, <p,b->, is_beam] — the is_beam flag is exactly LorentzNet's ±1 channel, and <p,b±> = E ∓ pz gives the lab-frame energy/longitudinal momentum that no jet-internal invariant can supply. We are a superset of their input, minus the dead m2. Treat this as a completed check, not a to-do.

**expected gain.** None directly — it rules out an input-feature explanation for the gap, which is worth knowing before spending quota on it.

**test cost gpu hours.** 0

**confidence.** high on the code; the paper/code discrepancy is stated as such

### 14. We report the last-epoch model; LorentzNet reports the best-validation-accuracy checkpoint

**mechanism.** Over 30 epochs the validation metric fluctuates by more than the seed spread. Selecting the best-validation epoch instead of the final one is a free draw from that fluctuation. LorentzNet evaluates on validation every epoch and keeps the argmax; we compute the same number and throw the weights away.

**evidence.** Ours: benchmarks/train.py lines 250-256 compute improved = val_acc > best_val_acc and update best_val_acc, but the checkpoint written at lines 261-276 always saves the CURRENT model.state_dict() — no copy of the best-val weights is ever retained, so the reported test metric is the last epoch. Corroborated by the loader comment in benchmarks/datasets.py lines 496-499: 'Val is only used for monitoring in our training loop (no early stopping; final model = last epoch)'.

**source.** arXiv:2201.08187 section 3.3, p.8: 'We test the model on the validation dataset at the end of each training epoch, and the model with the highest validation accuracy is saved as our best model for the final test.' vs C:/Users/Пользователь/Documents/so33/benchmarks/train.py lines 250-276

**applies to us.** Directly and cheaply. Our reported spread is AUC 0.98333 +- 0.00010 over seeds; epoch-to-epoch validation fluctuation late in training is typically larger than that, so last-epoch reporting is leaving a systematic, if small, amount on the table — and it makes our numbers not strictly comparable to LorentzNet's protocol. Fix: keep a second state_dict for the best-val epoch (a ~90 KB copy at 22.8k params) and score the test set with it. Rank the epochs by validation AUC rather than accuracy, since AUC is the reported metric.

**expected gain.** Order +0.0002 to +0.0005 AUC, and a like-for-like protocol match. Small, but it is the only item on this list that costs literally nothing.

**test cost gpu hours.** 0 — a code change that rides along on the next full run at no added cost. No standalone run needed.

**confidence.** high that the difference exists; medium on the size of the gain

### 15. Missing edge feature: LorentzNet feeds the relative Minkowski norm ||x_i - x_j||^2 alongside <x_i,x_j>, and after our asinh compression it is NOT recoverable from what we feed

**mechanism.** LorentzNet's edge message is m_ij = phi_e(h_i, h_j, psi(||x_i-x_j||^2), psi(<x_i,x_j>)) with psi(v)=sgn(v)log(|v|+1). We feed asinh(s_ab), asinh(s_aa), asinh(s_bb) as six separate channels (real and imaginary). The relative norm is s_aa + s_bb - 2 s_ab, a linear combination of the three raw quantities — but we apply asinh BEFORE the linear layer, and asinh(a)+asinh(b)-2asinh(c) != asinh(a+b-2c). So the first Linear cannot form it. The paper is explicit that this term is included even though it is redundant with the dot product: 'the interaction between particles relies on this term and we include it as a prior feature for ease of learning'.

**evidence.** Paper Equation 3.2 and the paragraph beneath it, section 3.1 p.6. Our side: benchmarks/so3c_models.py, the dense branch of SO3CMessageSetClassifier.forward builds e = cat([asinh(s_ab.real), asinh(s_ab.imag), asinh(row.real), asinh(row.imag), asinh(col.real), asinh(col.imag)] + scal). Note we DO already have the psi normalizer (asinh is the smooth form of sgn(v)log(|v|+1)) and we DO already feed the full bilinear rather than only a radial norm — so those two boxes are ticked; only the relative-norm channel is absent.

**source.** arXiv:2201.08187 Eq. 3.2, section 3.1 p.6; C:/Users/Пользователь/Documents/so33/benchmarks/so3c_models.py (SO3CMessageSetClassifier.forward, dense branch)

**applies to us.** Directly. Add asinh(Re(d_ab)) and asinh(Im(d_ab)) where d_ab = s_aa + s_bb - 2 s_ab, taking edge_in from 6C+2D=64 to 8C+2D=80. Parameter cost: the first edge Linear grows from (64,16) to (80,16), i.e. +256 weights per round x 3 rounds = +768 on 22,834 parameters (+3.4%). Test it together with dropping the dead m2 node channel — both are single-variable input changes with no architectural risk, and neither changes the cost curve.

**expected gain.** Modest, order +0.0003 to +0.001 AUC. It is the cheapest architectural item left: the paper singles this term out as a deliberate, redundant-but-helpful prior, which is exactly the kind of thing that shows up as an optimization win rather than a capacity win.

**test cost gpu hours.** ~0.2 for a 4-epoch probe on 20% of train (4 x 930 x 0.2 / 3600, at the bucketed epoch cost), ~0.3 at the current 1272 s/epoch. Full 30-epoch confirmation 7.7 h bucketed / 10.6 h as-is.

**confidence.** medium-high on the mechanism (the asinh non-linearity argument is solid); medium on the size

### 16. Answer to Q2: LorentzNet has exactly one ablation, and it attributes everything to equivariance plus the Minkowski dot product — there is NO depth, width, or beam ablation to borrow

**mechanism.** The paper's only controlled ablation (section 4.5) replaces the invariant edge inputs with the raw 4-vectors, m_ij = phi_e(x_i, x_j, h_i, h_j), breaking equivariance while holding hyperparameters and architecture fixed. The second attribution is argued against EGNN rather than ablated: EGNN uses only the radial distance ||x_i-x_j||^2, LorentzNet adds <x_i,x_j> 'to recover the information of angles according to Equation (3.1)'.

**evidence.** Table 4 p.12: LorentzNet w/o equivariance 0.934 / AUC 0.9832 / 1/eps_B(0.5) 290+-30 / 1/eps_B(0.3) 1105+-59, vs LorentzNet 0.942 / 0.9868 / 498+-18 / 2195+-173. Equivariance alone is worth ~2x background rejection. Table 1 p.9: EGNN (E(4)-equivariant, matched parameter order, 222k) scores 0.922 / 0.9760 / 397+-7 / 540+-49 — a 4x rejection deficit against LorentzNet's 224k. Everything else is fixed by fiat in section 3.3 p.7-8: L=6 LGEB blocks, width 72, dropout 0.2, c = 5e-3 for top tagging vs 1e-3 for quark-gluon 'chosen to achieve the best performances', AdamW wd 0.01, 35 epochs, batch 32 per GPU on 4x V100. Table 3 p.11 is a data-fraction study, not an architecture ablation. Table 5 p.13 is inference cost only.

**source.** arXiv:2201.08187 section 4.5 and Table 4 (p.12); Table 1 (p.9); section 3.3 (p.7-8); Table 3 (p.11); Table 5 (p.13)

**applies to us.** Both of their attributed mechanisms are already ours: we are exactly equivariant by construction, and we feed the full bilinear z_a.z_b (the complexified analogue of the Minkowski dot product), not a radial norm. Their LorentzNet-w/o row (AUC 0.9832) is almost exactly our current 0.98333 — which is the useful read: we have matched what a non-equivariant 224k-parameter network gets, and their +0.0036 over it comes from an axis we already have. So the remaining gap is not explained by anything they ablated. The two untested structural axes left are depth (their L=6 vs our rounds=3) and the flow step scale (their tuned c=5e-3 vs our T=1.0 default — I confirmed no --T flag is passed in notebooks/kaggle_beams_k64_seed0.ipynb, so T=1.0 from the class default). Given your measured result that capacity in channels was flat without beams and only +0.0004 with, depth-with-beams is the distinct untried axis; and a 200x mismatch in the step scale against a value the authors say they tuned per task is worth three points of a sweep.

**expected gain.** Depth rounds 3->6: unknown, but it is the largest remaining structural difference (they run 6 residual updates of both h and x; we run 3). T sweep {0.1, 0.3, 1.0}: this is a stability/optimization knob, plausibly +0.0005 if T=1.0 is over-rotating early rounds — our zero-init w_head and the 1+|z.z|^(1/2) soft bound already mitigate what c guards against, so this is the lower-confidence of the two.

**test cost gpu hours.** T sweep: 3 values x 4 epochs x 20% train = ~0.6 h. Depth 3->6 probe: 4 epochs x 20% at 2x cost = ~0.4 h; full 30-epoch run at rounds=6 is ~15.5 h bucketed (2x the round-linear cost) / 21 h as-is — one run fits the weekly 30 h budget only if the K=128 plan is cancelled per finding 1.

**confidence.** high on what the paper does and does not ablate; medium on depth being the payoff axis

### 17. Answer to Q3, plus a concrete mismatch: LorentzNet masks the self-edge out of the graph; we leave the diagonal in

**mechanism.** LorentzNet builds edge_mask = atom_mask (x) atom_mask and then removes the diagonal with ~torch.eye(), so no particle sends itself a message. Padding is handled entirely by these masks plus dynamic truncation to the batch maximum — there is no fixed node count anywhere in the pipeline. Our dense branch builds pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2) with no (1 - eye) factor.

**evidence.** LorentzNet-release top/collate.py: atom_mask = data['Pmu'][...,0] != 0.; edge_mask = atom_mask.unsqueeze(1)*atom_mask.unsqueeze(2), diagonal masked out with ~torch.eye(). Paper section 4.1 p.8 for the zero-padding statement. Ours: so3c_models.py, dense branch — pair_mask = mask_all.unsqueeze(-1) * mask_all.unsqueeze(-2), diagonal retained; note that the same file's _pooled_bivector_invariants (line 81) and _minkowski_stats (line 126) and _neighbour_graph DO apply (1.0 - eye), so the message-passing path is the inconsistent one.

**source.** LorentzNet-release top/collate.py; arXiv:2201.08187 section 4.1 p.8; C:/Users/Пользователь/Documents/so33/benchmarks/so3c_models.py

**applies to us.** Partially. The self-edge contributes nothing to the covariant update — cross(z_a, w_aa z_a) = 0 identically — so the flow is unaffected. But it does enter (a) the scalar message sum, where msg(e_aa) is averaged in as if it were a real neighbour, and (b) the denominator, which counts N rather than N-1. With mean N=49 that is a ~2% dilution of every scalar message plus one self-referential term. Small, and it is an internal-consistency fix rather than a borrowed idea. Bundle it with the edge-feature change; do not spend a separate probe on it.

**expected gain.** Marginal on its own, likely under +0.0002. Listed because it is a real, verifiable divergence from the reference implementation and costs nothing to correct.

**test cost gpu hours.** 0 standalone — fold into the ~0.2 h edge-feature probe as part of the same bundle.

**confidence.** high that the divergence exists; low that it matters much

### 18. Housekeeping: the local npz cache is truncated to 32 constituents, so no local measurement can see past K=32

**mechanism.** benchmarks/download_top_tagging.py takes --n-constituents (default MAX_PARTICLES=200) and writes a per-file scalar n_constituents into the npz. The local cache was built with 32. datasets.select_leading_constituents then zero-pads to whatever n_constituents the loader is asked for, silently and without warning — so load_top_tagging_constituents(n_constituents=64) against this cache returns 32 real rows plus 32 rows of guaranteed zeros, and every K^2 pairwise term over that padding is computed and then masked to nothing.

**evidence.** All three local files are (N, 32, 4) with the n_constituents field = 32: train (1211000, 32, 4), val (403000, 32, 4), test (404000, 32, 4) at C:/Users/Пользователь/Documents/so33/data/. Real rows among the stored 32: mean 30.6, 83.4% of jets fill all 32. Meanwhile the parquet under data/toptagging/ ships all 200. The Kaggle side is fine — notebooks/kaggle_dataprep_k128.ipynb exists and notebooks/kernel-metadata-beams-k64-seed0.json lists kernel_sources ['nsanya/so3c-data-prep-k-128'] — so the published K=64 numbers are genuine.

**source.** My measurement (scratchpad/gap/probe2.py) on C:/Users/Пользователь/Documents/so33/data/top_tagging_{train,val,test}.npz; C:/Users/Пользователь/Documents/so33/benchmarks/download_top_tagging.py lines 53, 226, 274-275; notebooks/kernel-metadata-beams-k64-seed0.json

**applies to us.** As a trap, not a result. Any local CPU sanity check, throughput measurement, or ablation probe run against this cache is capped at 32 real constituents regardless of the --n-constituents you pass, and will quietly disagree with the Kaggle numbers. Two cheap fixes: rebuild the local cache with --n-constituents 128 (CPU only, ~30 min, no GPU quota), and add an assertion in load_top_tagging_constituents that raises when the requested n_constituents exceeds the file's stored n_constituents field instead of padding. One side note to verify: kernel-metadata-dataprep-k128.json declares id 'nsanya/so3c-dataprep-k128' while the consumer lists 'nsanya/so3c-data-prep-k-128' — different slugs; worth confirming which kernel the K=64 runs actually mounted.

**expected gain.** No AUC gain. Prevents a class of silently wrong local measurement, and the loader assertion makes the same mistake impossible to repeat.

**test cost gpu hours.** 0 GPU-h (CPU parquet conversion, ~30 min wall clock)

**confidence.** high on the local cache; medium on the Kaggle slug question, which I could not verify without Kaggle access

## Angle: `beyond-pelican`

### 19. Depth beats width at small parameter budgets — keep ~10 blocks, shrink channels instead

**mechanism.** L-GATr-slim's downscaling study varies two axes separately: (a) jointly cutting blocks and width (12→4→2→1 blocks), and (b) holding blocks fixed at 10 and cutting only per-layer channel width. Path (a) collapses; path (b) survives to ~1000 parameters. The equivariant covariant update is a per-round contraction of the geometry; the number of rounds sets how many nested substructure relations (top→W→2 prongs) the network can compose, while width only sets how many parallel copies it holds. Your model has 3 rounds and 8 channels — it is on the wrong side of this trade.

**evidence.** arXiv:2512.17011 'Economical Jet Taggers — Equivariant, Slim, and Quantized' (Dec 2025), Sec. 2.5 / Fig. 3 / Table 7: 'when the number of network blocks and width are decreased jointly, the LLoCa-Transformer emerges as the leading architecture at small network sizes'; but when blocks are held at 10 and only per-layer width is reduced, L-GATr-slim reaches background rejection 'above 1000 using only two vector and four scalar channels' at ~1000 parameters. That is ~1000 rejection at ~1k params vs your 1131 at 22.8k params — a ~20x parameter-efficiency gap, and the paper's own 200k-param point sits around 1400-1500.

**source.** arXiv:2512.17011v2, Section 2.5 and Figure 3 (Table 7 gives the per-scale hyperparameters: channel dims, heads, block counts at 2M/200k/20k/2k for Transformer, LLoCa, ParT, L-GATr-slim). https://arxiv.org/html/2512.17011v2

**applies to us.** Directly. Your architecture is already block-structured: 3 rounds of (edge invariants → scalar MLP → covariant exp-flow update). Going to 8-10 rounds with channels cut from 8 to 3-4 is a hyperparameter change, not a rewrite. Caveat: their blocks are attention blocks with residual streams; at 10 rounds you will need a residual/identity path on both the scalar channel and the covariant z-update (your exp(-T[...]) flow is already multiplicative-identity at T=0, which helps) plus per-round LayerNorm on the invariant channel, or the depth will not train.

**expected gain.** This is the single largest published parameter-efficiency lever and the one your measured results most directly contradict. Their 20k-parameter point (your regime) sits near 1000-1100 rejection, i.e. roughly where you are — but their 200k point reaches ~1400-1500 on the same depth-preserving path, so the mechanism keeps paying. Realistic target: 1131 → 1400-1700 at comparable or lower parameter count.

**test cost gpu hours.** Pilot at K=32, 9 rounds, channels 4, 20 epochs: the K^2 pairwise term drops ~4x at K=32, and 3x more rounds at half channels is roughly 1.2-1.5x, so ~1272/4*1.35 ≈ 430 s/epoch → 2.4 GPU-h. Confirm the winner at K=64, 30 epochs: ~1500-1700 s/epoch → 12.5-14 GPU-h. Total ~15-17 GPU-h, fits in one week with room for a second run.

**confidence.** high

### 20. L-GATr-slim: drop higher grades, keep scalars + vectors only, gate with GLU on vector inner products

**mechanism.** Full L-GATr carries the whole 16-dimensional Clifford algebra G(1,3) — scalars, vectors, bivectors, axial-vectors, pseudoscalars. The slim variant keeps only scalars and vectors. Linear layers multiply every component of a vector by the same learnable scalar coefficient (the only Lorentz-allowed vector-to-vector map), nonlinearity comes from a GLU that takes the inner product of two vectors and multiplies it back onto the vector output of the linear layer, and attention is implemented by folding the Minkowski metric into a list of prefactors applied to the query vector so a standard Euclidean fused-attention kernel can be used. Result: same accuracy as full L-GATr with 6x less training time and 2x less memory.

**evidence.** arXiv:2512.17011, Table 1: L-GATr-slim 2.0M params, acc 0.9420, AUC 0.9869, 1/eps_B = 546±7 at eps_S=0.5 and 2264±93 at eps_S=0.3; full L-GATr 1.1M params, acc 0.9423, AUC 0.9870, 540±20 and 2240±70. Identical within error. Abstract: slim + quantized reaches 'an order-of-magnitude reduction in energy cost for a moderate performance decrease, down to 1000-parameter taggers.' Paper states slim cuts training time 6x and memory 2x versus full L-GATr with no performance loss, and that it 'can be compiled with torch.compile'.

**source.** arXiv:2512.17011v2, Sections 2.1-2.4, Table 1. https://arxiv.org/html/2512.17011v2

**applies to us.** Partly, and as a warning. Your z_a = bivec(p_a, P) is a BIVECTOR, i.e. exactly the grade L-GATr-slim found expendable — and the companion interpretability paper measured bivector channels as near-worthless. Two concrete transfers that do apply: (1) the GLU gating pattern — replace your fixed scalar weight w_b with sigma(MLP(invariants)) multiplying the vector output, giving a learned per-channel nonlinear gate instead of a linear combination; (2) the prefactor trick — if you move to attention (see finding 7), you can use PyTorch's fused SDPA on your complex/bivector dot products by folding the metric into the query, which is where most of your 1272 s/epoch goes.

**expected gain.** The GLU gating alone is a cheap +0.0003-0.0008 AUC class change. The larger implication is strategic: the bivector representation your whole model is built on is the grade two independent papers found least useful, which argues for adding an explicit vector (4-momentum-like) channel alongside z_a rather than replacing it.

**test cost gpu hours.** GLU gating is a drop-in replacement for the w_b aggregation weights with negligible FLOP change (~1.02x): 20 epochs at K=64 = 7.2 GPU-h, or ~1.8 GPU-h as a K=32 pilot. The torch.compile/fused-kernel work is an engineering change with no training cost and potentially negative net cost (it may cut your 1272 s/epoch).

**confidence.** medium

### 21. Bivector-only representation is the weakest choice — add explicit vector channels alongside z_a

**mechanism.** Two independent ablations attack the same question — which tensor grade should carry covariant information between nodes. LLoCa ablates message representations at fixed budget; the L-GATr interpretability paper ablates grades of a trained tagger by zeroing them. Both find vector-like channels dominant and bivector/rank-2 channels weak. The physical reading: the jet's prong structure is encoded in the constituents' 4-momentum directions, and a bivector z_a = p_a ∧ P has already projected out the component along P, discarding the very longitudinal information that separates a 3-prong top from a 1-prong QCD jet.

**evidence.** arXiv:2505.20280 (LLoCa) Table 2, message-representation ablation at fixed budget: 16 scalars only = 40±4 MSE (x1e-5); single rank-2 tensor = 2.0±0.4; 4 vectors = 1.4±0.2; 8 scalars + 2 vectors = 1.0±0.1. Vectors beat rank-2 by 2x, mixed scalar+vector beats both. arXiv:2606.21790 'What Do Lorentz-Equivariant Jet Taggers Learn?' grade ablations on L-GATr: 'Bivector (G2) channels are negligible (Delta AUC ~ 0.001)', while 'vector-like (G1+G3) channels are dominant'. Caveat: the LLoCa table is amplitude regression (Z+4g), not tagging, so transfer is indirect; the grade ablation IS on top tagging.

**source.** arXiv:2505.20280 Table 2 (https://arxiv.org/html/2505.20280v1); arXiv:2606.21790 Sec. on grade ablations (https://arxiv.org/html/2606.21790v1)

**applies to us.** Directly and uncomfortably — this is a critique of your central design choice. The cheap test is additive, not a rewrite: keep z_a, and add a second covariant channel v_a carrying the raw 4-momentum (or its boost into the jet rest frame) updated by the same exp-flow, with cross-terms z_a.v_b and v_a.v_b added to the edge invariant set. Your existing beam particles already break the symmetry that makes this non-degenerate.

**expected gain.** If the ablation transfers, the vector channel supplies information your bivectors structurally cannot represent. The LLoCa 8-scalar+2-vector vs rank-2 gap is 2x in MSE, which in tagging terms is the difference between the ParT tier (~1600) and the equivariant tier (~2200). Uncertain but high-ceiling.

**test cost gpu hours.** Adding one covariant channel roughly doubles the bilinear invariant count per edge (z.z, z.v, v.v) but the K^2 dot-product structure is unchanged: ~1.3-1.5x per epoch. At K=64, 20 epochs: 1272*1.4*20/3600 ≈ 9.9 GPU-h. Pilot at K=32, 20 epochs: ~2.5 GPU-h.

**confidence.** medium

### 22. JetClass pretraining is the single largest published gain: +29% rejection, and it is the only thing that beat PELICAN

**mechanism.** Pretrain on JetClass (100M jets, 10 classes) then fine-tune on the 1.2M top-tagging set. The pretraining task forces the network to learn generic substructure features (W/Z/H/t decay topologies) that the 1.2M-jet supervised set is too small to pin down. Every number above PELICAN's 2250 on this benchmark is a pretrained number.

**evidence.** arXiv:2411.00446 'A Lorentz-Equivariant Transformer for All of the LHC', Table 2: L-GATr trained from scratch = acc 0.9423±0.0002, AUC 0.9870±0.0001, 540±20 @ eps_S=0.5, 2240±70 @ 0.3. L-GATr-f.t. (JetClass-pretrained) = acc 0.9446±0.0002, AUC 0.98793±0.00001, 651±11 @ 0.5, 2894±84 @ 0.3. That is +654 rejection (+29%) and +0.0009 AUC from pretraining alone at identical parameter count (1.1M). arXiv:2512.17011 reproduces it: L-GATr-slim-f.t. = acc 0.9442, AUC 0.9879, 2927±70. MIParT (arXiv:2407.08682) independently reports fine-tuning from 100M JetClass improved top-tagging background rejection by 39%.

**source.** arXiv:2411.00446v2 Table 2 (https://arxiv.org/html/2411.00446v2); arXiv:2512.17011v2 Table 1; arXiv:2407.08682 abstract

**applies to us.** Yes, and it is the only mechanism in this literature with a gain large enough to close your gap in one step. Full-scale reproduction is out of budget — L-GATr used 1e6 iterations at batch 512 = 512M samples; at your measured throughput (1.211M samples / 1272 s = 952 samples/s) that is ~149 GPU-hours, five weeks of your allocation. But a truncated version is affordable: JetClass is downloadable in shards, and the question 'does a reduced pretraining budget capture a useful fraction of the +29%' is unanswered in the literature and worth one week.

**expected gain.** Full-budget published gain is +29% rejection (2240→2894). A 20M-sample, 3-pass truncated pretrain is ~5% of their sample budget; even capturing a third of the effect would take you from 1131 to ~1250, and it stacks with the architectural changes above rather than competing with them.

**test cost gpu hours.** One pass over a 20M-jet JetClass subset at 952 samples/s = 5.8 GPU-h; 3 passes = 17.5 GPU-h; plus fine-tuning 20 epochs on top tagging = 7.1 GPU-h. Total ~25 GPU-h = one full week's allocation for one shot. Cheaper probe: 10M subset, 2 passes = 5.8 GPU-h pretrain + 7.1 GPU-h fine-tune = ~13 GPU-h, enough to see whether the curve moves at all before committing the full week.

**confidence.** high

### 23. Add the time axis (1,0,0,0) as a reference particle, not just the two beams

**mechanism.** L-GATr-slim appends BOTH beam axes and the time axis as extra input particles. The beams break Lorentz invariance down to the longitudinal-boost-and-transverse-rotation subgroup; the time axis further breaks the residual longitudinal boost, letting the network represent detector-frame quantities (energy, pT thresholds, calorimeter granularity) that are genuinely not Lorentz-invariant because the detector is not. You measured that beams alone were worth +0.0004 AUC and unlocked the channel-8 capacity gain — the time axis is the same lever, unpulled.

**evidence.** arXiv:2512.17011v2: the model uses 'beam and time axes as additional input particles' to enable symmetry breaking and 'account for detector effects that violate Lorentz invariance'. This configuration is the one producing AUC 0.9869 / 2264 rejection in their Table 1.

**source.** arXiv:2512.17011v2, architecture/inputs section. https://arxiv.org/html/2512.17011v2

**applies to us.** Directly — it is a one-line change to your beam-appending code, one extra node from 66 to 67. Note the subtlety for your parameterization: z = bivec(p, P) with p = (1,0,0,0) and P the jet total is generically non-degenerate (it encodes the jet's boost direction relative to the lab), so it will not collapse to zero the way a reference parallel to P would.

**expected gain.** Your own beam measurement (+0.0004 AUC, and it unlocked +0.0004 more from capacity) is the best available prior for a second symmetry-breaking reference. Expect a similar or somewhat smaller increment, ~+0.0002-0.0004 AUC, possibly with a second capacity unlock.

**test cost gpu hours.** (67/66)^2 = 1.03x, so ~1310 s/epoch. 20 epochs = 7.3 GPU-h; 30 epochs = 10.9 GPU-h. Or hold cost exactly fixed by using 61 constituents + 3 references instead of 64 + 2: 10.6 GPU-h for 30 epochs.

**confidence.** medium

### 24. Lion optimizer at lr 3e-4 with weight decay 0.2 — not the AdamW recipe you already ruled out

**mechanism.** Every L-GATr result on this benchmark uses Lion, not Adam/AdamW: lr 3e-4, weight decay 0.2, batch 128, cosine annealing, 2e5 iterations. Lion's update is a sign function of the momentum, so its effective step size is decoupled from gradient magnitude and the conventional scaling is ~10x smaller lr and ~10x larger weight decay than Adam. Your measured optimum (Adam lr 3e-3) maps to Lion lr 3e-4 under exactly that scaling — which is evidence your lr search was correct, and that the wd 0.01 you tested is 20x below what this architecture class actually uses.

**evidence.** arXiv:2411.00446v2 Appendix A: 'Lion with learning rate 3e-4', batch size 128 (512 for JetClass), 2e5 iterations standard (1e6 for JetClass), CosineAnnealingLR, weight decay 0.2. arXiv:2512.17011v2 confirms the same: 200,000 iterations, Lion, batch 128, on an H100.

**source.** arXiv:2411.00446v2 Appendix A; arXiv:2512.17011v2 training section

**applies to us.** Yes, and it is distinct from the negative result you recorded. You tested AdamW wd 0.01 + dropout 0.2 + lr 1e-3 + warm restarts as a bundle and got 0.97769. The Lion recipe is a different optimizer with no dropout and 20x the weight decay. Note also the epoch budget: 2e5 iterations x batch 128 = 25.6M samples = 21 epochs of your 1.211M set — so their budget is close to your 20-30, and your 'we underfit at 35 epochs' observation is consistent with theirs rather than contradicting it.

**expected gain.** Optimizer swaps in this literature are worth a few 1e-4 in AUC, not a tier change. Its real value is as a control: it isolates whether your 0.98333 ceiling is an architecture limit or a training limit, before you spend a week on pretraining. Run it concurrently with the depth test.

**test cost gpu hours.** Zero architectural cost. 20 epochs at K=64 = 7.1 GPU-h; 30 epochs = 10.6 GPU-h. Cheapest informative version: K=32, 20 epochs ≈ 1.8 GPU-h.

**confidence.** medium

### 25. Replace the fixed aggregation weights w_b with softmax attention over the pair scores you already compute

**mechanism.** Your covariant update aggregates as sum_b w_b z_b with w_b a learned function of per-pair invariants. Every architecture above 2000 rejection instead normalizes the aggregation: L-GATr/LLoCa-ParT use softmax attention over all pairs, and PELICAN uses learned permutation-equivariant aggregators over the full K x K invariant matrix. The normalization matters because K varies per jet (you pad to 64 from up to 200 constituents) — an unnormalized sum makes the update magnitude scale with multiplicity, which is a QCD-vs-top discriminant the network then has to un-learn rather than a geometric feature.

**evidence.** arXiv:2411.00446v2 Table 2 stratifies cleanly by aggregation type: fixed-weight/kNN message passing (ParticleNet 1615±93, ParT 1602±81) sits ~600 below full-pairwise normalized aggregation (LorentzNet 2195±173, PELICAN 2250±75, L-GATr 2240±70). arXiv:2606.21790 Table 1 reproduces the same ordering on AUC: L-GATr 0.9869±0.0001, LLoCa-T 0.9867±0.0001, ParT 0.9857±0.0001, vanilla Transformer 0.9856±0.0001.

**source.** arXiv:2411.00446v2 Table 2; arXiv:2606.21790v1 Table 1 (https://arxiv.org/html/2606.21790v1)

**applies to us.** Yes and it is nearly free, because you already pay the K^2 cost to build every pair's invariants — you are computing the attention logits and then throwing away the normalization. Change w_b to softmax_b(MLP(invariants_ab)) per receiver a per channel. Combine with the prefactor trick from finding 2 to run it through a fused SDPA kernel and you may get the epoch time back.

**expected gain.** Small on its own (+0.0002-0.0005 AUC), but it is a prerequisite for the depth increase in finding 1: at 9-10 rounds an unnormalized sum will drift in scale and the deep model will not train stably. Treat it as part of the depth test, not a separate experiment.

**test cost gpu hours.** ~1.05x FLOPs, or faster if fused-attention-compatible. 30 epochs at K=64 = 11.1 GPU-h standalone, but ~0 marginal if bundled into the depth pilot.

**confidence.** medium

### 26. Nothing published as of Sept 2026 beats 2894 — and K=64 constituents is not your bottleneck

**mechanism.** Landscape check across every post-PELICAN entrant. From-scratch SOTA on the reference dataset has been flat since 2023 at 1/eps_B ~ 2240-2264, AUC 0.9870: PELICAN 2250±75 (208k), L-GATr 2240±70 (1.1M), L-GATr-slim 2264±93 (2.0M), LorentzNet 2195±173 (224k), CGENN 2172. The only genuine advance is pretraining (2894). Non-equivariant transformers remain a full tier below at 1602-1615. LLoCa (local canonicalization — predicting per-particle reference frames via polar decomposition of equivariantly-combined 4-momenta, then passing tensorial messages between frames) reaches 2150±130 at 2.0M params, i.e. it matches but does not exceed direct equivariance, and costs 60-100% more training time than its non-equivariant base.

**evidence.** arXiv:2608.02735 'Virtues and Vices of Equivariant Transformers' (Aug 2026) Table 4 is the current consolidated comparison; arXiv:2512.17011v2 Table 1 and arXiv:2411.00446v2 Table 2 give the numbers above. LLoCa overhead: arXiv:2505.20280 reports 'FLOPs rising by 10-50% and training time by 60-100%', split evenly between frame prediction and frame-to-frame transformation. 2608.02735 on the compute trade-off: 'For very low cost the baseline transformer performed best, but above a very basic threshold equivariant transformers outperform', with L-GATr-slim giving 'the best performance at fixed cost'.

**source.** arXiv:2608.02735 Table 4 (https://arxiv.org/html/2608.02735); arXiv:2512.17011v2 Table 1; arXiv:2411.00446v2 Table 2; arXiv:2505.20280 (https://arxiv.org/html/2505.20280v1)

**applies to us.** Two conclusions. (1) Do not chase LLoCa — it is a way to make an arbitrary non-equivariant network equivariant, which you do not need since you are already equivariant by construction, and it would cost you 60-100% more time for a number below PELICAN's. (2) Do not spend GPU-hours raising K past 64: 2608.02735 standardizes inference cost at N=50 particles and L-GATr's headline numbers come from a comparable regime, so your K=64 truncation of up to 200 constituents is not what separates you from 2250. Your gap is depth, aggregation normalization, and pretraining — all in the findings above.

**expected gain.** Negative-result value: rules out two expensive directions (K scaling, canonicalization) that look attractive and would cost 20-40 GPU-h to disconfirm empirically. Note the K^2 saving — staying at K=64 rather than K=128 avoids a 4x epoch cost (1272 → ~5000 s/epoch, 42 GPU-h for 30 epochs).

**test cost gpu hours.** 0 — this is the literature check itself. It frees ~20-40 GPU-h that would otherwise go to K scaling and canonicalization experiments.

**confidence.** high

