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
      (3 seeds). See `notes/diagnosis_findings.md`. Pending: top-tagging
      constituents stability confirmation for the eta-bound.
- [ ] Week 3: write `\section{Method}`.
- [ ] Week 4: write `\section{Experiments}` + figures.
- [ ] Week 5: write `\section{Introduction}`, `\section{Related work}`,
      `\section{Discussion}`.
- [ ] Week 6: polish + arXiv upload.
