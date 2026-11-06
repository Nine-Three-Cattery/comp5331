# Group report and slides

`report.tex` and `slides.tex` are the shared English masters. Model order,
owner and readiness come from [models.json](models.json). Each contribution
can be edited in its own directory. `make docs` builds a draft from ready
contributions. `make final-docs` requires all six ready contributions and
[editor synthesis](editor/README.md). The group masters share
a typography and color palette. LaTeX intermediates are built in a
temporary directory and removed automatically. `build/` holds the two
published PDFs.

## Model contribution format

| File | Contract |
|---|---|
| `report.tex` | One English `\subsection`. Cover implementation and deviations, causal or validation checks, results and limits. Use the group benchmark and metric definitions. |
| `slides.tex` | English Beamer frames. Aim for three frames covering model, evidence, and example or runtime. Give each frame one claim and its evidence. Follow the shared Beamer styles and allotted timing. |
| `speaker_notes.md` | One English block per frame in order. Include the transition to the next speaker. |
| `assets/` | Small, final figures only. Cite origin in prose and name files relative to the submission directory. |
| `results.csv` | Audited fold–seed–condition result snapshot with the shared header. |
| `provenance.json` | Protocol, run and bundle counts, reproduction artifact filename and SHA-256, runtime summary and supporting evidence hashes. |

Use [templates](templates/) or the
[Trajectron++ contribution](contributions/trajectronpp/README.md).
Fragments contain no preamble, title page, frame numbering or local
bibliography. Add paper citations to `references.bib`. Prefix labels with
your model ID. Use meters for distances, milliseconds for timing and
three decimals for reported displacement and Energy Score. Shared macros
read the same evidence used by the comparison table. Show how an estimate
varies across the relevant folds, seeds or conditions. Label paired changes
and uncertainty, and distinguish observations from possible explanations.

```latex
\ModelMetric{trajectronpp}{clean_0}{min_ade_mean}
\ModelMeanADEChange{trajectronpp}{noise_020}
\ModelRuntime{trajectronpp}{end_to_end_ms}
```

## Results schema

Copy only the header of the
[Trajectron++ CSV](contributions/trajectronpp/results.csv) or use the
[empty results template](templates/results.csv). Each ready model exports
one row per fold, seed and condition, for 90 rows. Use `causal_v2`, seeds
`0`, `1` and `2`, and condition IDs `clean_0`, `noise_005`, `noise_010`,
`noise_020`, `missing_1` and `missing_2`. Metric columns are within-fold
window means. Report the same K across all runs, and record it in the
registry. With K=1, minimum and mean errors coincide and diversity is
zero. The same Energy Score definition is used for either budget.

Each row identifies the ordered test manifest and the exact configuration,
checkpoint and prediction SHA-256. The runtime fields in `provenance.json`
are `preprocessing_ms`, `prediction_sampling_ms` and `end_to_end_ms`.
Use the median of per-run medians and preserve the full timing table in
the reproduction artifact. The builder checks structure and manifest
agreement. The editor must inspect the run audit and reproduction evidence
before accepting cross-model comparisons.

Keep the constant-velocity sanity check as supplementary evidence with
Individual LSTM. The learned-model matrix contains the six registered
models. The editor owns the final comparison and synthesis.

## Beamer presentation conventions

The shared master uses the standard Metropolis theme and compiles with
pdfLaTeX. On Debian or Ubuntu install `texlive-latex-extra` and
`texlive-fonts-recommended`. Use `make preview MODEL=<id>` to build a pending
owner's contribution before editor approval. The preview cover labels its
status. Commit frame source and vector figures; keep generated PDFs local.

Write a short result sentence as each frame title. Make the chart or
diagram the primary evidence, with axes and units readable when projected.
Use the report for the full per-fold result and the notes for verbal
reasoning. Use the shared palette, margins and frame numbering. Avoid
custom title pages and per-model slide themes.

[Metropolis](https://ctan.org/pkg/beamertheme-metropolis) ·
[Beamer presentation guidelines](https://ctan.net/macros/latex/contrib/beamer/doc/beameruserguide.pdf) ·
[MIT research presentation guidance](https://mitcommlab.mit.edu/nse/2021/08/27/claim-evidence-reasoning-more-value/)

Final assembly requires a current `benchmark_audit.json` beside each
model's CSV and [editor sign-off](editor/README.md). A draft or preview
can be built before those final gates clear.
