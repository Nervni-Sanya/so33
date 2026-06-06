# `paper/` — arXiv preprint workspace

Skeleton for the SO(3,3) preprint, set up per the 6-week plan in
`/root/.claude/plans/i-was-planning-to-cached-quokka.md`.

## Layout

- `main.tex` — compilable skeleton with section stubs (`% TODO` markers
  indicate week of plan in which each section is to be written).
- `refs.bib` — BibTeX with the ~7 references the related-work section
  must address. Entries marked `% verify:` need their arXiv ID / DOI
  double-checked against the canonical source before submission.
- `figures/` — empty, to be populated:
    - `equivariance_vs_norm.pdf` (Week 4, Figure 1).
    - `ood_vs_rapidity.pdf` (Week 4, Figure 2).
- `runs/week1.sh` — shell pipeline to launch the seeds×3 runs that
  back the headline tables. Run from anywhere; the script `cd`s to
  the repo root itself.
- `notes/related_work.md` — annotated bibliography and the positioning
  argument, feeds into Section 6.
- `notes/equivariant_diagnosis.md` — Week-2 measurement protocol for
  understanding why `so33_equivariant` loses OOD generalization.

## Build

```bash
cd paper/
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.tex` uses stock `article` class so it builds with any TeX Live
installation. For arXiv submission, switch to NeurIPS 2024 style by
replacing the `\documentclass` line and the small preamble block marked
`[SUBMIT]` in `main.tex`.

## Status (filled in as work progresses)

- [x] Week 1: seeds×3 runs (`runs/week1.sh`) — done, aggregated in
      `notes/results_week1.md`.
- [ ] Week 1: literature scan.
- [x] Week 2: `so33_equivariant` diagnosis -- bound_input identified as
      cause; `so33_equivariant_unbounded` and the principled
      `so33_equivariant_eta_bounded` both recover OOD 1.000 +/- 0.000
      (3 seeds). See `notes/diagnosis_findings.md`.
- [x] Bonus: real-data top-tagging finding -- eta_invariants reaches
      AUC 0.944 +/- 0.0005 on 100k constituents (3 seeds) vs 0.759 relu
      / 0.750 signature-only on the identical split (apples-to-apples
      gap confirmed). Arch B fails (0.504) due to a readout without
      pairwise eta-products. See `notes/top_tagging_finding.md`.
- [x] Week 3: `\section{Method}` written (3.1 algebra+basis, 3.2 geodesic
      ODE activation incl. the three input bounds, 3.3 eta_invariants,
      3.4 so33_equivariant). Corrected the connection to fixed learnable
      scalars (not a hypernetwork). LaTeX structurally validated
      (env/brace/math balance); not yet compiled (no local TeX).
- [x] Week 4: `\section{Experiments}` written (4.1 empirical
      equivariance with the clean ||c||=0 / sweep numbers, 4.2 boost-OOD
      table, 4.3 HIGGS matched+natural-width tables, 4.4 Adult
      parameter-efficiency, 4.5 top-tagging with the honest canonical-
      split caveat). Figure 1 (equivariance vs ||c||) generated via
      `benchmarks/figure_equivariance.py` into `paper/figures/`.
      Figure 2 (OOD AUC vs rapidity) deferred -- the boost-OOD table
      already carries the story; can add in week 6 polish if needed.
- [x] Week 5: Introduction, Related Work, Discussion, Conclusion written.
      All 7 \citep keys resolve to refs.bib (no dangling refs). The
      Discussion uses the correct diagnosis (non-invariant bound, not
      the earlier "Gamma drift" hypothesis).
- [x] Week 6 (content): Appendix A (hyperparameters) and Appendix B
      (reproduction commands) written. ~4080 words total, LaTeX
      structurally validated (474/474 braces, 686/686 inline math, 7/7
      env types balanced, 0 dangling \citep). Paper content COMPLETE.
- [ ] Week 6 (mechanical): compile locally
      (`pdflatex && bibtex && pdflatex && pdflatex`), check refs/tables
      /Figure 1 render, optionally verify the `% verify:` arXiv IDs in
      refs.bib, then upload to arXiv (suggested categories: cs.LG
      primary, hep-ph cross-list).
