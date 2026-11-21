# COMP5331 Group 7 Pedestrian Trajectory Forecasting

This project implements and compares six pedestrian trajectory forecasting
models on ETH/UCY. Each model predicts the next 12 positions from eight
observed positions sampled 0.4 seconds apart. The shared `causal_v2`
benchmark measures displacement error, probabilistic forecast quality, and
robustness to noisy or missing observations across five held-out scenes.

[Project GitHub repository](https://github.com/Nine-Three-Cattery/comp5331) ·
[Final report](submission/FINAL_REPORT.md) ·
[Benchmark specification](benchmark/README.md)

## Setup and compilation

The forecasting code runs directly in Python. The reference CPU setup uses
Linux x86-64, Git, Make, and Python 3.12. Run these commands in a Bash-compatible
shell.

```bash
git clone --branch main https://github.com/Nine-Three-Cattery/comp5331.git
cd comp5331
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
python -m pip install -r benchmark/requirements.txt
```

The following command compiles the Python sources to bytecode as a syntax check.

```bash
python -m compileall -q benchmark models li_junhao_trajectronpp submission
```

For the optional report and slides, install a TeX distribution with
pdfLaTeX, BibTeX, Beamer, and the Metropolis theme. The build writes
`submission/build/report.pdf` and `submission/build/slides.pdf`.

```bash
python submission/build.py --final --check
make final-docs PYTHON=python
```

## Run an example on the CPU

Run the Individual LSTM pilot from the repository root with the virtual
environment active. The example prepares the benchmark, trains for two
epochs, selects a checkpoint on validation data, and forecasts all 364 ETH
test windows under six observation conditions. Use a fresh checkout for each
pilot because the runner requires a new output directory.

```bash
make benchmark PYTHON=python
python models/individual_lstm/run.py pilot
make pilot MODEL=individual_lstm RUNS=artifacts/individual_lstm/pilot/runs PYTHON=python
```

The first command fetches or reuses the six pinned ETH/UCY coordinate files,
verifies their SHA-256 hashes, and builds all five ordered holdout splits.
It prints `BENCHMARK_OK` when the data and metric checks pass. The final
command checks the six prediction bundles and prints
`BENCHMARK_AUDIT_OK artifacts/individual_lstm/pilot_audit.json`.

The clean prediction bundle is at
`artifacts/individual_lstm/pilot/runs/eth/seed_0/evaluation/clean_0/predictions_test_clean_0.npz`.
Its `samples` array has shape `(364, 1, 12, 2)`, representing 364
windows, one forecast per window, 12 future steps, and x/y coordinates in
meters.

## Run the full experiments

Run `make benchmark PYTHON=python` from the repository root first. Each model
guide gives the exact training, prediction, and replay commands, including
any separate dependencies or CUDA requirements. Each audited formal result
table has 90 rows from five holdouts, three seeds, and six observation
conditions. `K` is the number of forecasts per test window.

| Model | K | Run guide | Audited results |
| --- | ---: | --- | --- |
| Individual LSTM | 1 | [Guide](models/individual_lstm/README.md) | [CSV](submission/contributions/individual_lstm/results.csv) |
| Social-LSTM | 1 | [Guide](models/social_lstm/README.md) | [CSV](submission/contributions/social_lstm/results.csv) |
| Social-GAN | 20 | [Guide](models/social_gan/README.md) | [CSV](submission/contributions/social_gan/results.csv) |
| Social-STGCNN | 20 | [Guide](models/social_stgcnn/README.md) | [CSV](submission/contributions/social_stgcnn/results.csv) |
| Trajectron++ | 20 | [Guide](li_junhao_trajectronpp/README.md) | [CSV](submission/contributions/trajectronpp/results.csv) |
| DSPIGCN | 20 | [Guide](models/dspigcn/README.md) | [CSV](submission/contributions/dspigcn/results.csv) |

To audit an archived formal run, obtain the archive named in the model's
`submission/contributions/<model_id>/provenance.json`, verify its SHA-256,
and extract it according to the model guide. The command below audits an
archived Social-GAN run tree at its default path.

```bash
make audit MODEL=social_gan RUNS=artifacts/social_gan/formal/runs PYTHON=python
```

For newly generated formal predictions, run `make score` with the same
`MODEL`, `RUNS`, and `PYTHON` arguments before auditing. Scoring writes
detailed metrics under `artifacts/<model_id>/metrics/` and replaces the
model's tracked `results.csv`. The prediction bundle schema, metrics, and
split rules are in the [benchmark contract](benchmark/README.md).

Raw data and derived splits live under `data/`. Checkpoints, full predictions,
and reproduction archives live under `artifacts/` or in the location listed
in the relevant [contribution folder](submission/contributions/). The
[model registry](submission/models.json) records ownership and acceptance.
For development, use `group7/project` as the base for model branches and
pull requests. [Contributor SOP](CONTRIBUTING.md)

## Tested operating systems

| Operating system | Verified scope |
| --- | --- |
| macOS 26.6.2 (Apple silicon, arm64) | Python 3.14.7, PyTorch 2.14.0, NumPy 2.5.3, and pandas 3.0.6. Verified Python bytecode compilation, benchmark setup with cached raw data, the Individual LSTM ETH pilot and its six-bundle audit, and the final submission metadata check. |
| Linux x86-64 | Group-recorded experiment and replay environments for Social-STGCNN and DSPIGCN. See the [Social-STGCNN replay](submission/contributions/social_stgcnn/evidence/extracted_reproduction_check.json) and [DSPIGCN provenance](submission/contributions/dspigcn/provenance.json). |

Native Windows execution is unverified. The [contributor SOP](CONTRIBUTING.md)
documents the Linux environment for Windows users running WSL.

## Source file reference

The tables describe each Python and shell source file, the Makefile, and
the LaTeX sources. Paths are relative to the directory named above each table.

<details>
<summary>Root, shared benchmark, and submission tooling</summary>

Paths are relative to the repository root.

| File | Purpose |
| --- | --- |
| `Makefile` | Provides the benchmark, pilot, scoring, audit, document, and release commands. |
| `benchmark/contract.py` | Defines the files and hashes that identify the shared protocol. |
| `benchmark/setup.py` | Fetches and verifies coordinates, builds manifests, checks fixtures, and computes training-only motion thresholds. |
| `benchmark/validate.py` | Validates prediction bundles and checkpoint identities, recomputes metrics, and writes score tables or audit receipts. |
| `submission/build.py` | Validates the contribution records and builds the shared report and slides. |
| `submission/release.py` | Packages a clean, committed checkout with the PDFs, score table, speaker notes, and checksums. |
| `submission/contributions/trajectronpp/refresh_charts.py` | Rebuilds the Trajectron++ charts from audited summaries. |

</details>

<details>
<summary>Individual LSTM</summary>

Paths are relative to `models/individual_lstm/`.

| File | Purpose |
| --- | --- |
| `model.py` | Implements the deterministic encoder-decoder LSTM and constant-velocity baseline. |
| `run.py` | Trains the model, selects the checkpoint, exports forecasts, and measures CPU runtime. |
| `summarize.py` | Builds comparison and runtime tables from the group audit and baseline results. |
| `handoff.py` | Prepares the contribution snapshots and experiment archive. |
| `own_unit_tests/test_causal.py` | Checks that target history and predictions are invariant to changes in futures, neighbors, and stored derivatives. |

</details>

<details>
<summary>Social-LSTM</summary>

Paths are relative to `models/social_lstm/`.

| File | Purpose |
| --- | --- |
| `causal/__init__.py` | Marks the deterministic Social-LSTM package. |
| `causal/data.py` | Builds visible target and neighbor inputs over the shared neutral loader. |
| `causal/model.py` | Implements the deterministic recurrent predictor and social-grid pooling. |
| `causal/run.py` | Trains, selects, predicts, exports, and times the benchmark model. |
| `analysis/figures.py` | Generates accuracy, robustness, trajectory, and motion-strata evidence. |
| `analysis/summarize.py` | Summarizes formal results and computes the constant-velocity comparison. |
| `legacy/data.py` | Builds and loads windows for the earlier exploratory protocol. |
| `legacy/diagnostics.py` | Computes motion and proximity diagnostics with training-derived thresholds. |
| `legacy/download_data.py` | Downloads or verifies the pinned legacy coordinates. |
| `legacy/metrics.py` | Computes trajectory, motion, and close-approach metrics for legacy exports. |
| `legacy/model.py` | Implements the probabilistic Social-LSTM and Gaussian loss used in exploratory runs. |
| `legacy/reproduce.py` | Replays the legacy CPU-training and MPS-evaluation workflow. |
| `legacy/run.py` | Trains, evaluates, exports, and times legacy experiments. |
| `legacy/smoke.py` | Runs a synthetic training and inference check. |
| `legacy/verify_exports.py` | Recomputes metrics from saved legacy predictions. |
| `own_unit_tests/test_causal.py` | Tests future invariance, feature construction, and forecast behavior for the deterministic model. |
| `own_unit_tests/test_legacy.py` | Exercises legacy preprocessing, model behavior, and metrics. |

</details>

<details>
<summary>Social-GAN</summary>

Paths are relative to `models/social_gan/`.

| File | Purpose |
| --- | --- |
| `data.py` | Builds causal social histories and samples scene-level latent inputs. |
| `model.py` | Implements the recurrent generator, social pooling, discriminator, and variety training. |
| `run.py` | Trains, selects, exports, times, and replays Social-GAN forecasts. |
| `analysis.py` | Produces tables and figures for robustness, sample budgets, motion, and runtime. |
| `check_metrics.py` | Independently recomputes the six headline scores from exported forecasts. |
| `check_training.py` | Compares a fresh ETH pilot with the saved pilot experiment. |
| `handoff.py` | Packages, verifies, and rescores the owner archive. |
| `own_unit_tests/test_causal.py` | Tests visible-history inputs, transformations, and forecast output. |
| `own_unit_tests/test_model.py` | Tests social max pooling and adversarial variety training. |

</details>

<details>
<summary>Social-STGCNN</summary>

Paths are relative to `models/social_stgcnn/`.

| File | Purpose |
| --- | --- |
| `windows.py` | Builds unpadded scene graphs and maps test targets to manifest order. |
| `model.py` | Implements full-scene graph and temporal convolutions, Gaussian loss, and forecast sampling. |
| `run.py` | Trains, selects, exports, and times one fold and seed. |
| `export.py` | Writes full-scene predictions in the shared neutral bundle format. |
| `timing.py` | Reconstructs model inputs from in-memory coordinates for runtime measurement. |
| `retime.py` | Measures all selected checkpoints on one host. |
| `check_reference.py` | Compares network and sampling outputs with the pinned external reference using matched weights. |
| `check_selection.py` | Replays the selected checkpoints on uncached validation windows. |
| `check_trained_reference.py` | Replays selected forecast paths with the reference network and the group's trained weights. |
| `review_metrics.py` | Independently checks displacement errors and computes heading diagnostics. |
| `analyze.py` | Summarizes formal results and renders contribution figures. |
| `handoff.py` | Associates the public evidence with its audited review archive. |
| `package_review.py` | Packages the audited experiment matrix for review. |
| `verify_review.py` | Verifies the archive and reproduces selected ETH clean forecasts. |
| `preview.py` | Builds the model's review PDFs with the shared preamble. |
| `own_unit_tests/test_analysis.py` | Checks evidence joins, sample counts, and sample-budget calculations. |
| `own_unit_tests/test_model.py` | Checks graph operators, full-scene inference, and the forecast distribution. |
| `own_unit_tests/test_package_review.py` | Checks archive identities and how receipts identify source and results. |
| `own_unit_tests/test_timing.py` | Checks that the timed loader reconstructs the expected clean inputs. |
| `own_unit_tests/test_training.py` | Checks scene loss and validation weighting against hand calculations. |
| `own_unit_tests/test_windows.py` | Checks causality, manifest order, and neutral export on synthetic scenes. |

</details>

<details>
<summary>Trajectron++ and shared trajectory utilities</summary>

Paths are relative to `li_junhao_trajectronpp/`.

| File | Purpose |
| --- | --- |
| `independent_tpp/__init__.py` | Marks the independent pedestrian model package. |
| `independent_tpp/data.py` | Loads neutral splits and builds, caches, and batches target and neighbor windows. |
| `independent_tpp/model.py` | Implements the graph-conditioned CVAE and pedestrian dynamics for Trajectron++. |
| `independent_tpp/protocol.py` | Defines fold and corruption rules, history-only derivatives, and data hashes. |
| `scripts/fetch_causal_raw.py` | Retrieves and verifies the six pinned ETH/UCY coordinate files. |
| `scripts/prepare_neutral_data.py` | Converts raw trajectories into metric coordinates and ordered split manifests. |
| `scripts/validate_neutral_data.py` | Validates the neutral trajectory representation and split contents. |
| `scripts/cache_independent_data.py` | Materializes clean neutral inputs for training. |
| `scripts/verify_causal_data.py` | Checks cached tensors against reconstructed inputs and exports a fixture. |
| `scripts/compute_independent_thresholds.py` | Computes motion and spatial diagnostic thresholds from training data. |
| `scripts/train_independent.py` | Trains the independent Trajectron++ core and saves candidate checkpoints. |
| `scripts/select_independent_checkpoint.py` | Ranks candidate checkpoints on validation forecasts. |
| `scripts/predict_independent.py` | Produces metric-coordinate prediction bundles from a selected checkpoint. |
| `scripts/evaluate_independent_run.py` | Predicts and scores all six test conditions for one run. |
| `scripts/run_independent_formal.py` | Runs the formal matrix with verified restart points. |
| `scripts/evaluate.py` | Scores prediction bundles and writes window-level, aggregate, and spatial metrics. |
| `scripts/metrics.py` | Implements displacement, distribution, motion, and spatial metrics. |
| `scripts/tiny_metric_check.py` | Verifies metrics and the evaluator with small reference cases. |
| `scripts/aggregate_independent_results.py` | Combines run-level results across folds, seeds, and conditions. |
| `scripts/aggregate_spatial.py` | Combines close-approach and pairwise-distance summaries. |
| `scripts/analyze_independent_diagnostics.py` | Computes trajectory strata and heading-change diagnostics. |
| `scripts/analyze_sample_budget.py` | Compares scores at different forecast sample counts. |
| `scripts/analyze_statistics.py` | Computes paired robustness effects and fold-level uncertainty. |
| `scripts/generate_failure_analysis.py` | Selects and plots trajectory examples and failure cases. |
| `scripts/benchmark_independent_runtime.py` | Measures runtime on the benchmark's fixed 128-window sample. |
| `scripts/audit_independent_formal.py` | Verifies the formal matrix and writes a relocatable prediction index. |
| `scripts/prepare_causal_report.py` | Generates report tables, figures, and speaker notes from audited results. |
| `scripts/package_causal_release.py` | Archives the audited source, results, and replay inputs. |
| `scripts/verify_causal_release.py` | Replays a selected CUDA forecast and recomputes headline scores after extraction. |
| `scripts/finalize_causal_results.sh` | Runs the post-training audit, analyses, and release workflow. |
| `scripts/verify_independent_core.py` | Runs a small acceptance check for the Trajectron++ core. |
| `own_unit_tests/test_causal_history.py` | Tests causal fill, derivatives, and future invariance for targets and neighbors. |
| `report/build_latex.sh` | Compiles the standalone Trajectron++ report and slides. |

</details>

<details>
<summary>DSPIGCN</summary>

Paths are relative to `models/dspigcn/`.

| File | Purpose |
| --- | --- |
| `dspigcn/__init__.py` | Marks the registered DSPIGCN package. |
| `dspigcn/model.py` | Implements the dual-stream graph network, physics loss, and anchor sampling. |
| `dspigcn/data.py` | Converts shared neutral trajectories into graph batches. |
| `dspigcn/adapter.py` | Exposes the model through the group's fit, select, and predict interface. |
| `dspigcn/experiment.py` | Handles resumable training, validation, checkpoint loading, and forecast export. |
| `run.py` | Provides the train, predict, and experiment-matrix commands. |
| `launch_formal.py` | Launches formal training by seed and records run status. |
| `runtime.py` | Measures latency on the fixed benchmark windows. |
| `analyze.py` | Summarizes scores and produces fixed-example and failure plots. |
| `package.py` | Archives audited runs with hashes for replay. |
| `verify_release.py` | Verifies the extracted release and replays its forecast bundles. |
| `own_unit_tests/test_core.py` | Tests causal inputs, graph geometry, physics loss, sampling, and training behavior. |
| `own_unit_tests/test_scoring.py` | Checks displacement units and separate best samples for ADE and FDE. |

</details>

<details>
<summary>Supplemental DSPIGCN candidate</summary>

The [candidate guide](models/dspigcn/provisional_variant/README.md) documents
the separate experiment protocol. Paths are relative to
`models/dspigcn/provisional_variant/`.

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Declares the candidate package and its dependencies. |
| `dspigcn_candidate/__init__.py` | Marks the candidate package. |
| `dspigcn_candidate/data.py` | Builds candidate splits, manifests, and trajectory batches. |
| `dspigcn_candidate/model.py` | Implements the candidate's dual-stream attention and graph network. |
| `dspigcn_candidate/losses.py` | Computes the Gaussian likelihood and polynomial motion penalties. |
| `dspigcn_candidate/sampling.py` | Draws temporally coherent Gaussian anchor samples. |
| `dspigcn_candidate/runner.py` | Prepares data, trains, evaluates, measures runtime, and runs the matrix. |
| `dspigcn_candidate/evaluation.py` | Applies deterministic perturbations and computes candidate metrics. |
| `dspigcn_candidate/protocol.py` | Compares manifests to check that experiment populations agree. |
| `dspigcn_candidate/interface.py` | Exposes inference on observed metric trajectories. |
| `dspigcn_candidate/group_interface.py` | Adapts the candidate to the registered model's graph batches. |
| `dspigcn_candidate/aggregate.py` | Aggregates metrics across scenes, conditions, and seeds. |
| `dspigcn_candidate/audit.py` | Checks saved bundles against manifests and recomputes accuracy scores. |
| `dspigcn_candidate/plotting.py` | Draws trajectory plots from saved predictions. |
| `dspigcn_candidate/report.py` | Generates the candidate report and figures from saved evidence. |
| `tests/test_core.py` | Tests data handling, model behavior, and metric calculations. |
| `tests/test_group_interface.py` | Checks candidate inference on the group's graph batches. |
| `tests/test_integration.py` | Exercises the candidate workflow on synthetic data. |

</details>

<details>
<summary>Report, slides, and bibliography</summary>

Paths are relative to the repository root. The [submission guide](submission/README.md)
describes how these sources are assembled.

| File | Purpose |
| --- | --- |
| `submission/report.tex` | Master source for the group report. |
| `submission/slides.tex` | Master source for the Beamer deck. |
| `submission/common.tex` | Loads generated model metrics and declares the shared metric macros. |
| `submission/palette.tex` | Defines the shared document colors. |
| `submission/references.bib` | Bibliography for the project report. |
| `submission/editor/report_synthesis.tex` | Cross-model interpretation for the report. |
| `submission/editor/slides_synthesis.tex` | Cross-model comparison and conclusion frames. |
| `submission/templates/report.tex` | Starting template for a model report subsection. |
| `submission/templates/slides.tex` | Starting template for a model's frames. |
| `submission/contributions/individual_lstm/report.tex` | Describes the target-only model and constant-velocity comparison. |
| `submission/contributions/individual_lstm/slides.tex` | Presents the deterministic baseline and its robustness results. |
| `submission/contributions/individual_lstm/evidence/paired_table.tex` | Paired LSTM and constant-velocity report table. |
| `submission/contributions/individual_lstm/evidence/paired_frame.tex` | Condensed paired comparison table for slides. |
| `submission/contributions/social_lstm/report.tex` | Describes deterministic social pooling and the formal K=1 results. |
| `submission/contributions/social_lstm/slides.tex` | Presents the social-grid model and representative results. |
| `submission/contributions/social_gan/report.tex` | Describes the adversarial model, sample budgets, and robustness. |
| `submission/contributions/social_gan/slides.tex` | Presents social pooling and evidence on sampling and robustness. |
| `submission/contributions/social_gan/analysis.tex` | Generated macros for audited Social-GAN results. |
| `submission/contributions/social_stgcnn/report.tex` | Documents the full-scene graph network and its audited results. |
| `submission/contributions/social_stgcnn/slides.tex` | Presents the full-scene decoder and forecast evidence. |
| `submission/contributions/social_stgcnn/contribution_statement.tex` | Records the owner's implementation and evaluation contributions. |
| `submission/contributions/trajectronpp/report.tex` | Describes the graph CVAE, dynamics, and audited results. |
| `submission/contributions/trajectronpp/slides.tex` | Presents the model, sample-budget effects, and robustness. |
| `submission/contributions/trajectronpp/analysis.tex` | Generated macros for audited Trajectron++ results. |
| `submission/contributions/dspigcn/report.tex` | Describes the physics-informed model and shared benchmark results. |
| `submission/contributions/dspigcn/slides.tex` | Presents the dual-stream design and forecast robustness. |
| `li_junhao_trajectronpp/report/TrajectronPP_LI_Junhao_Technical_Report.tex` | Standalone Trajectron++ technical report. |
| `li_junhao_trajectronpp/report/TrajectronPP_5min_Slides.tex` | Standalone five-minute Trajectron++ deck. |
| `li_junhao_trajectronpp/report/causal_results.tex` | Generated metrics and tables for the standalone documents. |

</details>
