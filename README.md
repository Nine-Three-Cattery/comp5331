# COMP5331 Group 7 Pedestrian Trajectory Forecasting

Six independent model implementations for forecasting pedestrian motion on
ETH/UCY. The project studies single-forecast and ensemble accuracy, sensitivity
to corrupted histories, motion diagnostics, and inference cost.

Start with the [group final report](submission/FINAL_REPORT.md) for the study
and the [benchmark contract](benchmark/README.md) for data, evaluation and run
formats. The [model registry](submission/models.json) records ownership,
sampling budgets and editor acceptance.

## Find a model and its results

| Model | Owner | Implementation and run commands | Results and evidence |
| --- | --- | --- | --- |
| Individual LSTM | LU Yichong | [Code](models/individual_lstm/) | [Common results](submission/contributions/individual_lstm/results.csv) · [Constant velocity and other evidence](submission/contributions/individual_lstm/) |
| Social-LSTM | WANG Yifan | [Code](models/social_lstm/) | [Common K=1 results](submission/contributions/social_lstm/results.csv) · [Validation, figures and earlier K=20 exploration](submission/contributions/social_lstm/) |
| Social-GAN | Cheung, Alan Pak To | [Code](models/social_gan/) | [Common results](submission/contributions/social_gan/results.csv) · [Sampling, robustness and replay evidence](submission/contributions/social_gan/) |
| Social-STGCNN | Zongmin Zhang | [Code](models/social_stgcnn/) | [Full-scene common results](submission/contributions/social_stgcnn/results.csv) · [Audit, replay and release links](submission/contributions/social_stgcnn/) |
| Trajectron++ | LI Junhao | [Code](li_junhao_trajectronpp/) | [Accepted common results](submission/contributions/trajectronpp/results.csv) · [Audit, provenance and figures](submission/contributions/trajectronpp/) |
| DSPIGCN | YAN Zhengyang | [Registered code](models/dspigcn/) · [Candidate variant](models/dspigcn/provisional_variant/) | [Common ETH pilot](submission/contributions/dspigcn/evidence/) · [Candidate results and audit](models/dspigcn/provisional_variant/results/provisional_main/) |

The five common result tables each contain 90 fold–seed–condition rows with
a benchmark audit receipt. Trajectron++ is marked ready. Four audited
implementations await editor acceptance. Registered DSPIGCN has pilot
evidence and awaits a formal common result table. The DSPIGCN candidate has
its own 90-row audit under a different protocol.

## Repository map

| Path | What to find |
| --- | --- |
| [benchmark/](benchmark/) | Shared data setup, scoring, audit, locked manifests, corruption fixture and runtime window IDs |
| [models/](models/) and [li_junhao_trajectronpp/](li_junhao_trajectronpp/) | Model implementations, configurations, dependency files, causal tests and reproduction commands |
| [submission/contributions/](submission/contributions/) | Owner result tables, evidence, archive identities, report subsections, Beamer frames and speaker notes |
| [submission/](submission/) | Markdown final report, shared LaTeX masters, bibliography and [editor checklist](submission/editor/) |
| `data/` | Local raw coordinates, generated neutral datasets and training-derived thresholds |
| `artifacts/` | Local checkpoints, complete predictions, detailed diagnostics and release archives |
| `submission/build/` | Local PDFs compiled from the shared LaTeX masters |

## Reproduce the shared benchmark

Use the Linux x86-64 reference environment in the [clone-to-merge SOP](CONTRIBUTING.md).
For a new reference setup, run the following from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
python -m pip install -r benchmark/requirements.txt
make benchmark PYTHON=python
```

The gate verifies the six pinned raw coordinate files, regenerates all
five ordered holdouts, checks the target corruption and metric fixtures,
and creates training-only motion thresholds. Expect `BENCHMARK_OK`. Use the
model's guide for its separate training environment and commands. Each
`causal_v2` formal run uses seeds 0–2, a clean-validation-selected checkpoint
per fold and seed, and six test conditions with the registered K.

With an extracted owner run tree and the benchmark prepared, check or
recompute results with the shared evaluator. Run the export command when
you intend to regenerate the contribution CSV.

```bash
MODEL=social_gan
make pilot MODEL="$MODEL" RUNS="artifacts/$MODEL/pilot/runs" PYTHON=python
make audit MODEL="$MODEL" RUNS="artifacts/$MODEL/formal/runs" PYTHON=python
make score MODEL="$MODEL" RUNS="artifacts/$MODEL/formal/runs" PYTHON=python
```

### Data and full experiment archives

The repository tracks the benchmark locks, recipes and compact evidence.
Keep raw data, model weights, complete predictions and generated release
files in ignored paths or the group's approved artifact store. To recover
full runs, obtain the archive named in the relevant `provenance.json` and
verify its SHA-256 before extraction. The Social-STGCNN contribution links
its published full-scene and earlier target-readout releases. For archives
without a download link, request the named archive from its owner. The
DSPIGCN candidate records its archive in
[publication.json](models/dspigcn/provisional_variant/results/provisional_main/publication.json).
The [common specification](GROUP7_COMMON_DATA_EVALUATION_SPEC.md) covers
population, coordinates, causal features and attribution.

## Review and submission

Create future model branches and pull requests from `group7/project` using
[CONTRIBUTING.md](CONTRIBUTING.md). The source integration on `main` brings
together owner contributions. Editor acceptance remains governed by the
[review checklist](submission/editor/README.md) and registry.

`make check-docs` checks the ready contribution. Run
`python submission/build.py --preview MODEL_ID --check` to validate a pending
contribution's complete common-format data. Run `make preview MODEL=MODEL_ID`
to compile its report and slides, or `make docs` to compile the ready
contributions. PDF builds require pdfLaTeX, BibTeX and Beamer Metropolis.
`make final-docs` and `make release` require all six models to clear the
editor gates. Add student IDs and member-confirmed declarations to a private
submission copy of the Markdown report.
