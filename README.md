# COMP5331 Group 7 · Pedestrian trajectory forecasting

This repository is the collaboration workspace for six independent ETH/UCY
model implementations and the group report and presentation. The current
assembly contains the audited Trajectron++ contribution. The five remaining
model slots are marked pending in the model registry.

## Start here

- **Model owner or agent** — follow the [clone-to-merge SOP](CONTRIBUTING.md).
  Use the [registry](submission/models.json) to select your assigned model.
- **Group editor** — follow the [review and release checklist](submission/editor/README.md).
- **Experiment contract** — use the [shared benchmark](benchmark/README.md)
  and [course specification](GROUP7_COMMON_DATA_EVALUATION_SPEC.md).
- **Existing contribution** — [Trajectron++](submission/contributions/trajectronpp/README.md)
  provides working code, audited results and merge-ready presentation sources.

## Repository map

| Path | Purpose |
|---|---|
| `benchmark/` | Common protocol entry points and lightweight manifest, corruption and runtime fixtures |
| `models/<model_id>/` | Model-owned code, configs, environment, tests and attribution |
| `li_junhao_trajectronpp/` | Existing Trajectron++ implementation and technical reproduction entry points |
| `submission/contributions/<model_id>/` | One model's English report subsection, slide frames, notes, figures and audited result snapshot |
| `submission/editor/` | Group comparison and synthesis |
| `submission/build/` | Generated group PDFs, kept locally |

## Build the group draft

From the repository root, run `make check-docs` and `make docs`. Building the
PDFs uses Python 3, pdfLaTeX, BibTeX and Beamer Metropolis. Open
`submission/build/report.pdf` and `submission/build/slides.pdf`.
A clean checkout can build these documents from the small committed
snapshots. `make final-docs` also requires all six ready contributions and
the editor's report and slides synthesis.

## Prepare the shared repository

Keep the root `.gitignore` when transferring these sources to the group's
checkout. Review the staged files there before pushing. Local `data/`,
`third_party/`, model `outputs/`, course originals and generated PDFs stay
outside Git. Large reproduction archives can be referenced by filename
and SHA-256, with a download location once the group approves one. The
current workspace keeps course originals under `local/course/` and earlier
drafts and reference workflows under `local/legacy/`.
