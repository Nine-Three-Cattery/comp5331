# Trajectron++ · LI Junhao

The group-ready sources are [report.tex](report.tex), [slides.tex](slides.tex)
and [speaker notes](speaker_notes.md). Build from the root with `make docs`.
This contribution uses the shared report subsection format and three
presentation frames. Individual technical materials and reproduction
commands live in the [model guide](../../../li_junhao_trajectronpp/README.md).

[Results](results.csv) are the 90 audited `causal_v2` fold–seed–condition
rows from the formal matrix. [Provenance](provenance.json) records the
release archive digest, evidence hashes and timing summary. The [evidence snapshots](evidence/) retain clean sample-budget results,
paired bootstrap estimates and per-run timings from the audited release.
Regenerate the analysis macros and vector figures by running
`python submission/contributions/trajectronpp/refresh_charts.py` with
Matplotlib. Group PDF builds use the committed figures and require no GPU
or Python plotting dependencies. Keep the prose, figures and notes
aligned when refreshing results.

[Shared benchmark audit](benchmark_audit.json) binds the 90 verified
prediction bundles to the shared code and this contribution CSV.
