# Social-LSTM contribution — WANG Yifan

Status: **pending**. This package contains an adapted independent implementation and exploratory evidence. It is not an audited causal_v2 release.

- `report.tex`: English subsection for the group technical report.
- `slides.tex`: three frames following the shared Beamer theme.
- `social_lstm_3min.pptx`: editable supplemental PowerPoint version.
- `speaker_notes.md`: approximately three-minute English talk.
- `evidence/`: explicitly legacy local summaries and selected qualitative example.
- `results.csv`: shared schema, intentionally no formal result rows.
- `provenance.json`: zero formal runs/bundles, with exploratory counts separated.
- Model commands and attribution: `models/social_lstm/README.md` from repository root.

The legacy study used seeds 11/22/33 and K=20. Group integration requires deterministic K=1, seeds 0/1/2, the shared manifests/loader, target-only perturbations, RMS-norm Energy Score, and formal audit. Do not map legacy values into shared metric macros or the comparison table. The registry remains pending.

`make check-docs` can check the existing assembled ready contributions. `make preview MODEL=social_lstm` is expected to reject missing formal results. The owner-only PDFs are layout previews and cannot replace that audit gate. The report PDF has two pages. The talk has three slides and a 371-word English script. Generated PDFs, datasets, checkpoints and full predictions are intentionally excluded from Git.

This section is intended to merge into the six-part report after editor review. Its present scientific conclusion is that sampling coverage and average error diverge, and rollout limitations remain unresolved. AI assistance contributed to implementation, diagnosis, packaging and draft writing; the owner must review the final assessed work and declarations.
