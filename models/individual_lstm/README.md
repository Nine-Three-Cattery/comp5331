# Individual LSTM · Group 7 causal_v2

Owner: LU Yichong. Local alignment of the previously delivered deterministic individual-motion baseline. The old module1-v1 numbers are historical, not submission results. This owner implementation uses only the eight target xy observations. See independent_design.md and implementation_provenance.md.

## Run from the repository root

Use the Python 3.12 CPU environment in CONTRIBUTING.md. This local run uses the same torch/numpy/pandas versions on Apple silicon, not the reference Linux x86-64 machine. Recheck on the reference host before claiming reference-environment reproduction. Keep model status pending until editor review.

    make benchmark PYTHON=python
    python models/individual_lstm/own_unit_tests/test_causal.py
    python models/individual_lstm/run.py pilot
    make pilot MODEL=individual_lstm RUNS=artifacts/individual_lstm/pilot/runs PYTHON=python
    python models/individual_lstm/run.py formal
    make score MODEL=individual_lstm RUNS=artifacts/individual_lstm/formal/runs PYTHON=python
    make audit MODEL=individual_lstm RUNS=artifacts/individual_lstm/formal/runs PYTHON=python

The owner script refuses to overwrite existing runs. Run a fresh training experiment in a separate checkout. It reloads the winning checkpoint and exports the six conditions, then records ten runtime repetitions per run. Runtime excludes model/load/disk setup; includes constructing target-only observation windows, tensor conversion, batch prediction and CPU output. CPU timing is not comparable with the GPU numbers in other contributions.

Formal predictions sit under artifacts/individual_lstm/formal/runs. The supplemental CV rows use the actual training-seed-specific noisy observations, despite CV having no trainable weights. CV is never entered as a seventh group model. Supplementary headline metrics are computed by the group's vectorized metrics_for_budget_batch function; the 90-bundle make audit receipt is for the LSTM.

## Review and package locally

After make audit, run `python models/individual_lstm/handoff.py` to regenerate the paired CV table, cross-model macro comparison, localized alignment note and the local artifact. The comparison verifies the checked-in Trajectron++ result CSV against its receipt and matches each fold/seed/condition to the same manifest before calculating any differences. This is a comparison to published group evidence, not a rerun of Trajectron++.

Use `python submission/build.py --preview individual_lstm --check` for the group contribution schema; building the optional PDF previews requires the group's pdfLaTeX/BibTeX toolchain. The artifact extracts into a checkout of group7/project at df016def5890898b70d372fb0e8e2bc142121330. Generate neutral data with make benchmark before rescoring; model-specific training data are intentionally not embedded.
