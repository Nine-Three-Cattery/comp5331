# Packaging validation

- 19 owner unit tests passed (10 causal-contract tests and 9 retained legacy tests). Independent model checks cover translation, padding and the zero-residual anchor.
- The common setup gate, full ETH seed-0 pilot, 15 formal runs and all 90 prediction bundles completed. The common scorer/auditor passed; receipts and results are included.
- Saved-checkpoint replay from the extracted release archive passed for ETH seed 0 under all six conditions (364 windows each). Inputs, masks, labels, records and metadata matched exactly; maximum prediction difference was 4.77e-7 m, within absolute tolerance 1e-5 m. This is same-host MPS inference reproduction, not retraining or Linux validation.
- All 15 fold/seed runtime measurements completed, with three warmups and ten repetitions of 128 targets. Full repetitions and timing boundaries are recorded.
- Contributor preview schema validation passed (two contributions included with Social-LSTM preview). `make check-docs` passed for the single registry-ready contribution; it does not change Social-LSTM editor status.
- Owner report compiled to three pages; shared-theme Beamer preview compiled to three frames with Tectonic, without overflow/citation warnings. All pages were visually inspected. Standard pdflatex execution was unavailable.
- Supplemental three-slide PPTX passed package integrity, layout/font, native chart/workbook and first-party import checks. All final slide previews were inspected. The error chart starts at zero. Native Microsoft PowerPoint execution was not tested.
- English script: approximately 376 spoken words, about three minutes at 125 words/minute; rehearse before presenting.
- Full datasets, weights, prediction bundles, generated document PDFs and archive remain local and excluded from Git. Portable source figure PDFs are included.
- Group editor acceptance, reference Linux reproduction, hardware comparison policy and large-artifact transfer remain outstanding. The local formal experiments and audit are complete.
