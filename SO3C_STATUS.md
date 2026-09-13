# SO3C status

Last updated 2026-09-13. This is the single place to start from. Commit hashes point at the evidence for every number.

## Scope and goal

- **Scope.** This repository holds two separate papers. `paper/main.tex` is the SO(3,3) preprint and is **out of scope** here. SO3C — the complexified-SO(3) construction in `so3c/` and `benchmarks/so3c_models.py` — is a different paper with no draft yet. Work only on SO3C.
- **Goal.** State of the art on the Kasieczka top-tagging reference dataset, canonical splits (1.211M train / 403k val / 404k test). Not a repositioning to "construction plus methodology": that was proposed after the capacity sweep saturated and was overridden on 2026-09-12.
- **Targets**, verified against the source tables (`paper/figures/literature_reference.csv`):

  | model | params | AUC | 1/ε_B at ε_S=0.3 |
  |---|---|---|---|
  | PELICAN (arXiv:2307.16506, table 1) | 208k | 0.9870 | 2250 ± 75 |
  | LorentzNet (arXiv:2201.08187, tables 1, 5) | 224k | 0.9868 | 2195 ± 173 |
  | PELICAN, size sweep (table 2) | 1k | 0.9835 | 1145 ± 74 |
  | PELICAN, size sweep (table 2) | 11k | 0.9858 | 1879 ± 103 |

## Current best

`so3c_message_set` with **beams + channels 8**, K=64, canonical protocol, 30 epochs, Adam 3e-3 cosine, P100 float32, 22834 parameters (`b72437f`):

| | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 |
|---|---|---|---|
| seed 0 | 0.98326 | 1147.2 | 287.6 |
| seed 1 | 0.98340 | 1115.5 | 297.4 |
| **mean** | **0.98333 ± 0.00010** | **1131 ± 22** | **292 ± 7** |
| **two-seed ensemble** | **0.98373** | **1174** | **306** |

- **Gap to PELICAN:** 0.0037 AUC and a factor 1.99 in rejection (from 0.0063 and 2.6 before beams).
- **The sobering comparison (verified):** PELICAN's own 1k-parameter model (0.9835 / 1145) matches this 22.8k-parameter model. The gap is architectural, not a matter of size.
- **Exact equivariance, measured** (float64, `87206e7`): with the beams transformed along with the jet, AUC stays 0.9684 at every boost scale (drop −0.0001); with the beams pinned to the lab it falls to 0.9332. That is exact covariance, the correct physics, and the same property PELICAN has.
- **Reproduce:** `notebooks/kaggle_beams_k64_seed{0,1}.ipynb`, then `notebooks/kaggle_beams_k64_finish.ipynb` (both seeds stop at epoch 24 on the session cap and resume). Results are in `results_message/beams_k64/`. Ensemble: `python -m benchmarks.ensemble_scores --results-dir results_message/beams_k64`.

## How we got here (canonical protocol)

| model | params | K | AUC | 1/ε_B@0.3 | commit |
|---|---|---|---|---|---|
| `eta_invariants` anchor | 4802 | 32 | 0.9478 | 49 | — |
| `so3c_invariant_set` (no flow) | 5122 | 32 | 0.9689 | 183 | — |
| `so3c_covariant_set` (1 covariant round) | 9078 | 32 | 0.9746 ± 0.0001 | 312 | `c3a98d0` |
| `so3c_covariant_set` | 9078 | 64 | 0.9772 ± 0.0001 | 638 ± 1 | `6b4e7b5` |
| `so3c_message_set` (3 rounds + scalar channel) | 13862 | 64 | 0.98073 ± 0.00006 | 850 ± 56 | `d77fb9c` |
| + beams + channels 8 | 22834 | 64 | 0.98333 ± 0.00010 | 1131 ± 22 | `b72437f` |

## Closed — measured, do not re-run without a reason

| question | result | commit |
|---|---|---|
| Connection built from invariants (`so3c_equivariant_set`) | **Not equivariant once trained**: 0.9424 falls to 0.4414 (below chance) at boost scale 3. Replaced by the covariant connection `z_a × Σ_b φ z_b` | `c3a98d0`, `f305ae4` |
| ODE solver instead of closed form (`so3c_interaction_set`) | 0.9735, 9.5× slower | `c3a98d0` |
| K for the covariant model | K=32/64/128 → 0.9746/0.9772/0.9781, rejection 312/638/639: saturates at 64 | `6b4e7b5` |
| **K beyond 64, by energy content** (raw 200-slot parquet, 100k test jets) | K=64 keeps 99.75% of jet energy on average (p1 95.3%); 82% of jets are complete and have exact jet mass. K=128 adds only the soft tail. Combined with the K sweep above and 19.3 h per seed even with kNN: **closed** | `1259ff7` |
| **Per-constituent mass m² as an input** | rounding noise: \|m²\|/E² median 4.6e-8, max 2.3e-7, 50% negative, log\|m²\| tracks log E² with slope 0.998. The constituents are massless | `1259ff7` |
| Channels on the broken model | flat, 0.9743–0.9745 across 22× parameters | — |
| kNN(16) graph at K=32 | −0.0011 AUC for 2% wall clock saved | `1f01ddd` |
| Capacity **without** beams (K=32 probe) | flat: scalar_dim 24, hidden 256, channels 8/16, rounds 6 all within 0.0006 AUC of the anchor | `abd5660` |
| Scalar channel | load-bearing: −0.0025 AUC and −40% rejection without it | `abd5660` |
| Published LorentzNet/PELICAN recipe (AdamW wd 0.01, dropout 0.2, lr 1e-3, 35-epoch warm restarts) | **negative**: 0.97769 alone (below anchor 0.9786), and beams + recipe 0.98065 < beams alone 0.98088 | `837c567` |
| Learning rate | 3e-3 beats both 6e-3 (−0.0003) and 1e-3 | `c325950` |
| Longer training | 35 epochs: +0.00015 AUC, −14 rejection, 75% more compute | `c325950` |
| Regularisation | dropout 0.05 + wd 1e-4: −0.0011 AUC, −133 rejection. The model **underfits** | `c325950` |
| Capacity **with** beams | channels 8: +0.0004 AUC, +67 rejection; **channels 16: worse than 8** (0.98111 / 811) at 1.5× cost | `c325950`, `90f2ead` |
| K=128 wall clock | dense 602 s/epoch on 100k jets (60.8 h full protocol); kNN(16) 191 s (19.3 h), 3.15× faster | `90f2ead` |
| Two-seed ensemble | +0.0004 AUC; member scores correlate at 0.994 | `49e0c61`, `b72437f` |
| Epochs 25–30 vs 24 (session cap) | +0.00022 AUC | `b72437f` |

**Lesson worth keeping:** a closed question can re-open when the inputs change. Capacity was flat without beams and paid once beams supplied the information. kNN was useless at K=32 and 3.15× faster at K=128.

## Open — candidates

### Verified defects in the current model (free to fix)

Both are now switchable and default to the old behaviour: `--no-mass-input --no-self-edges` (22634 params on beams + channels 8, CPU cost 1.00x the headline, `6771158`). Not yet run on GPU.

- **m² fed as noise.** `h_init` takes `asinh(m²)` per node, and three of the seven `_minkowski_stats` readout features are per-particle m² moments. All of it is rounding noise (see Closed).
- **Self-edges in the dense graph.** The dense path includes `a = b` in the scalar message sum and counts N, not N−1, in the denominator. The flow itself is unaffected, since `z_a × w_aa z_a = 0`. LorentzNet masks the self-edge; our sparse path already excludes it.

### Best-motivated architectural candidates

- **Rank-2 pair latent** with a reduced Eq2→2 aggregator basis (7 of PELICAN's 15). PELICAN carries a [B,N,N,C] state through its blocks; we collapse pairs to nodes every round. The verified table-2 comparison — 1k PELICAN parameters match our 22.8k — is consistent with this being the dominant carrier. Outcome likely bimodal: a real step toward 1400–1800 rejection, or under +0.0005 if our flow already supplies it. Agent estimate: screen at 20% data × 3 seeds ≈ 11 GPU-h.
- **A vector channel alongside the bivector.** Two reasons, one measured elsewhere and one structural:
  - arXiv:2606.21790 (abstract, verified) finds that in L-GATr "bivector channels are negligible for top-quark tagging while vector-like channels are dominant". The LLoCa (2505.20280) and slim L-GATr (2512.17011) abstracts do not address grades; their bodies are unchecked.
  - Our lift `z_a = bivec(p_a, P)` is **unchanged under p_a → p_a + λP**, so it discards each constituent's component along the jet axis. The scalar channel recovers ⟨p_a, P⟩ only as an invariant. A covariant vector v_a = p_a keeps it — the same kind of missing-information problem that beams fixed. **Implemented** as `--vector-channel`: 531 parameters on beams + channels 8, CPU cost 1.10x the headline, exactly covariant to 7e-15 with and without beams, and the vector update moves the logits by 7e-2 to 9e-2. Not yet run on GPU.

### Cheap candidates (unverified gain)

| candidate | mechanism | note |
|---|---|---|
| True multiplicity as a jet scalar | multiplicity separates classes (signal mean 54.7, background 43.4 constituents) and is clipped at 64 for 17.8% of jets | Lorentz invariant; count from the K=128 data before slicing |
| Relative-norm edge feature d_ab = s_aa + s_bb − 2 s_ab | LorentzNet Eq 3.2 feeds ‖x_i − x_j‖²; not recoverable after asinh | implemented, `--relnorm-edge` (`6771158`); CPU cost 1.30x the headline |
| f_α multi-resolution embedding, learnable exponents | PELICAN Sec 3.1: ((1+x)^(α²) − 1)/α², α initialised over [0.05, 0.5] | implemented, `--falpha n` (`6771158`). **Not** zero-runtime as the agent claimed: at n=3 on beams + channels 8, K=64, CPU cost is **3.54x the headline** (~37.5 h per 30-epoch seed). CPU ratios misled before (kNN), so measure on GPU before any full run |
| N^α / N̄^α aggregation rescaling | sum-vs-mean semantics; our flow angle scales with Σ_b w_b z_b | ~0 runtime |
| Time-axis reference particle (1,0,0,0) | a third symmetry-breaking reference, as in L-GATr | 1.03× cost |
| Softmax attention over w_b | normalised aggregation; needed for depth | ~1.05× |
| Depth over width | ~10 blocks with fewer channels at small budgets (slim L-GATr) | our rounds 6 was −0.0006, but **without** beams |
| Best-validation checkpoint | LorentzNet reports the best-val checkpoint | protocol match; +0.0002–0.0005 claimed |
| Lion optimiser, lr 3e-4, wd 0.2 | a different optimiser, not the ruled-out AdamW bundle | a training-limit control |
| JetClass pretraining | the only published result above PELICAN (L-GATr 2894, arXiv:2411.00446) | ≥ 25 GPU-h truncated; changes the comparison class |

Full agent text: `so3c_notes/gap_to_sota_2026-09-13.md`. Three of the workflow's six agents failed on the usage limit, so the findings there are unranked.

## Infrastructure traps

| trap | what happened | fix |
|---|---|---|
| Kaggle derives the kernel slug from the **title** | polling the wrong slug returns HTTP 403 "Permission kernels.get denied", which reads like auth | `push` prints the real slug |
| Duplicate kernel titles | HTTP 409 | unique titles |
| P100 is sm_60 | preinstalled torch lacks the arch; every CUDA op fails | `GPU_SETUP` cell reinstalls torch 2.5.1 (~150 s) |
| Proxy `127.0.0.1:10808` is dead | every API call refused | run the Kaggle client with `HTTPS_PROXY` / `HTTP_PROXY` unset |
| Weekly GPU quota resets **Saturday** | a Monday-based running total said "spent" when ~24 h were available | test by pushing, not by arithmetic |
| Kernel output (including the log) is unavailable until the kernel finishes | cannot read per-epoch cost mid-run | budget from completed runs |
| A new session starts with an empty `/kaggle/working` | `--resume` alone restarts from epoch 1 | finish kernels mount the earlier run via `kernel_sources` and copy the checkpoint (`build_*_finish`) |
| `max_seconds` compared the cumulative clock | a run resumed past the cap trained one epoch per session, forever | per-session cap (`c030d17`) with a regression test |
| Code bundle ships only `CODE_PACKAGES` | the on-card pytest guard found no tests and stopped the session | `tests` added (`a4a0c61`); replay the bundle locally before pushing |
| CUDA-only resume bug (RNG state on GPU) | invisible to CPU tests | on-card `pytest -k cuda_resume` guard must report "1 passed" |
| Eval OOM at large K | the **eval chunk** (B, K, K, C), not the training batch | `--eval-chunk-size` 2048 at K=32, 1024 at K=64, 256 at K=128 |
| Result filenames carry no K or variant | K-sweep runs overwrote each other; a beams boost run would have overwritten the beamless one | one results directory per variant; `--tag` in the boost script; the figure groups by model **and** tag |
| Session cap vs cost | beams + channels 8 at K=64 is 1272 s/epoch = 10.6 h; runs stop at epoch 24 with `--max-seconds 30000` | plan a finish kernel for any run over ~8 h |
| Cost extrapolated from a cheaper setting | K=32 ratio 3.05× became 4.7× at K=64; the kNN saving and the K slope also missed | measure at the target setting before committing a window |
| float32 in the boost diagnostic | fakes a 0.023 AUC symmetry break at boost scale 2 | `run_boost_robustness.py` defaults to float64 |
| Beams change what "invariant" means | with fixed beams the output must change | measure with the beams moved along with the jet |
| Zero-initialised connection heads | equivariance tests pass vacuously on the identity flow | tests excite the heads and assert the logits move |
| Local npz cache stores only K=32 | local CPU checks cannot see past 32 constituents (verified: shape `(n, 32, 4)`) | K=64/128 data exists only on Kaggle (`nsanya/so3c-data-prep-k-128`); the raw 200-slot parquet is local in `data/toptagging/` |
| Bash tool collapses `\\` to `\` | `\times` became TAB + "imes" in a LaTeX edit | write files containing backslashes with the Write tool, or build them with `chr(92)` |
| Large subagent workflows | 45 agents: 38 failed on the usage limit; 6 agents: 3 failed | keep workflows small; verify bounded claims directly |
| Unverified agent output | a findings file said "zero runtime" for f_α and cited abstracts that do not make the claim | verify before building on it; record the verification |
| Separate commands in a Bash script keep running after one fails | a status-doc edit raised and exited non-zero, and the `git commit` on the next line ran anyway (`6771158` has the code but not the doc update it was meant to carry) | chain every dependent step with `&&`, or `set -e` |

## Compute and quota facts

| configuration | seconds per epoch, full train set, P100 | 30 epochs |
|---|---|---|
| `so3c_covariant_set`, K=64 | ~199 | 1.66 h |
| `so3c_message_set`, K=64 | ~930 | 7.75 h |
| + beams + channels 8, K=64 | 1272 | 10.6 h |
| K=128 dense (from 100k jets) | ~7290 | 60.8 h |
| K=128 kNN(16) (from 100k jets) | ~2313 | 19.3 h |

- The K=32 probe protocol (400k train jets, 20 epochs) costs 0.6–1.2 h per row.
- Quota is 30 GPU-h per week. The window ending 2026-09-19 is essentially spent (~29 h); the next opens Saturday 2026-09-19.
