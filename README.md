# COMP5331 Group 7 · Pedestrian trajectory forecasting

This repository is the collaboration workspace for six independent ETH/UCY
model implementations and the group report and presentation. The current
assembly contains the audited Trajectron++ contribution. The five remaining
model slots are marked pending in the model registry.

## Join the project

`group7/project` is the shared integration branch for the COMP5331 project.
Clone it directly so your checkout starts with the project sources and its
own Git history.

```bash
git clone --single-branch --branch group7/project https://github.com/Nine-Three-Cattery/comp5331.git comp5331-group7
cd comp5331-group7
make check-docs
```

Model owners work on `model/<model_id>` branches created from `group7/project`.
Open each model pull request with `group7/project` as its base. The group editor
reviews and merges contributions there and assembles the final report and deck.

- **Model owner or agent** — follow the [clone-to-merge SOP](CONTRIBUTING.md).
  Use the [registry](submission/models.json) to select your assigned model
  and read [AGENTS.md](AGENTS.md) for repository conventions.
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

## Keep local material out of Git

Review `git diff --cached --name-status` before each commit. Keep raw data,
course originals, Canvas exports, model weights, full predictions, generated
documents and full reproduction archives in ignored paths. Use `local/` for
course material and `artifacts/` for reproduction packages. Commit the small
audited tables and source figures used by the report and deck. Record each
model's artifact filename and SHA-256 in its contribution provenance and
share the artifact through the group's agreed storage location.
