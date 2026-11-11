# Social-LSTM contribution — WANG Yifan

The deterministic K=1 causal_v2 matrix is complete: 15 validation-selected runs, 90 prediction bundles, and a passed common interface/score audit on the recorded Mac environment. Registry status remains pending for group editor review and reference-environment reproduction; it no longer indicates missing local formal experiments.

- `report.tex`: English model subsection using shared metric macros.
- `slides.tex`: three shared-theme Beamer frames.
- `social_lstm_3min.pptx`: supplemental editable three-slide PowerPoint.
- `speaker_notes.md`: approximately three-minute English script.
- `results.csv`, `benchmark_audit.json`, `pilot_audit.json`: actual common-format results and audit receipts.
- `evidence/causal_*`: current summaries, fixed-rule example, crowd/turn analysis and runtime.
- `evidence/legacy_*`: historical K20 exploration, excluded from current results and comparisons.
- `provenance.json`: archive hash, runtime boundaries, local environment and review limitations.

Code, exact commands and source attribution are under `models/social_lstm/` from the repository root. Run `python3 submission/build.py --preview social_lstm --check` to validate this contribution's data/provenance schema. The standard preview uses pdflatex/bibtex; local PDFs were rendered with Tectonic using the shared palette, theme and macros.

The release archive named in provenance retains the checkpoint and full prediction data locally; generated PDFs, raw data and large artifacts are excluded from Git. Transfer that archive through the group's agreed storage before peer reproduction. Mac MPS timing starts from cached neutral histories and must be labeled separately from Linux/reference-hardware results.

This is a declared deterministic adaptation of Social-LSTM, not an exact reproduction of the original Gaussian model. Forecast quality is reported as measured, including the constant-velocity comparison. AI assistance contributed to code, diagnosis and draft writing; the owner is responsible for reviewing the assessed work and disclosures.
