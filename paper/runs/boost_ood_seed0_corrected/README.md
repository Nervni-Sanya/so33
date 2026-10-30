# Boost-OOD, seed 0, on the corrected code

Per-model records of `python -m benchmarks.run_boost_ood --seeds 0` (commit
1143514, see each record's `provenance`), run after
the adaptive-scale gradient fix and with the Architecture-B connection cap
(`--connection-cap auto`). They answer the rewrite's TODO "re-run Table 1 and
state whether any number changes" for seed 0, and are the data behind
`paper/figures/fig_boost_ood_{rapidity,readout,coeff_norms}.pdf`.

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
readout does not depend on c at all: ||c|| decays to ~1e-16 under the
Frobenius penalty (`fig_boost_ood_coeff_norms.pdf`). Rerun seeds 0-2 with
`python -m benchmarks.run_boost_ood --seeds 0-2` for the table itself.

Note: these records predate commit 6968f5d, which seeds every model right
before it is built (so its initial weights no longer depend on the models
trained before it in the same process). A re-run on the current code starts
the SO33 models from different initial weights, so their last digits can
differ from the table above.
