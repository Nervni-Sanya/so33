# Boost-OOD, seed 0, on the corrected code

Per-model records of `python -m benchmarks.run_boost_ood --seeds 0` (commit
77ba122, see each record's `provenance`), run after
the adaptive-scale gradient fix and with the Architecture-B connection cap
(`--connection-cap auto`). They answer the rewrite's TODO "re-run Table 1 and
state whether any number changes" for seed 0, and are the data behind
`paper/figures/fig_boost_ood_{rapidity,readout,coeff_norms}.pdf` (the readout figure
is drawn from the per-sample `__samples.npz` files of the same run, which are
git-ignored and not kept here; a re-run writes them again).

Every record has `git_dirty: true`: the working tree had uncommitted changes when
the run started. The next commit, 1143514, changes only
`benchmarks/figure_boost_ood.py`, which the run does not import. The recorded `argv`
shows the temporary `--results-dir` the run wrote to.

OOD AUC, seed 0 — independent re-run on the pre-fix code (referee) vs this run:

| model | pre-fix | corrected |
|---|---:|---:|
| eta_invariants (Readout A) | 1.000 | 1.000 |
| so33_equivariant_unbounded / _eta_bounded | 1.000 | 1.000 |
| so33_equivariant (beta_euc) / _frozen | 0.662 | 0.662 |
| relu_mlp | 0.949 | 0.949 |
| so33_multi | 0.892 | 0.892 |
| so33 | 0.887 | 0.887 |
| so33_signature_only | 0.888 | 0.888 |
| relu_bottleneck | 0.881 | 0.881 |

No number changes at the quoted precision. The trained connections are small
(||c|| <= 0.05 for the flat models), where the dropped derivative of the scale
is a small correction, and for Architecture B (one particle per sample) the
readout does not depend on c at all: ||c|| decays to <= 2e-16 (1e-16 for
`so33_equivariant`, <= 2e-18 for `_eta_bounded` / `_unbounded`) under the Frobenius
penalty (`fig_boost_ood_coeff_norms.pdf`); the frozen variant keeps its initial
||c|| ~ 0.04. Rerun seeds 0-2 with
`python -m benchmarks.run_boost_ood --seeds 0-2` for the table itself.

Note: these records predate commit 6968f5d, which seeds every model right
before it is built (so its initial weights no longer depend on the models
trained before it in the same process). A re-run on the current code starts
the SO33 models from different initial weights, so their last digits can
differ from the table above.
