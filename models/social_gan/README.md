# Social-GAN

Owner: Cheung Alan Pak To. Model ID: `social_gan`. The contribution uses
the pooled Social GAN architecture adapted to the Group 7 benchmark.
See the [design](independent_design.md), [paper notes](paper_reading_notes.md)
and [implementation provenance](implementation_provenance.md).

## Setup and pilot

Run from the repository root. Use the reference Linux x86-64 CPU
environment in [the SOP](../../CONTRIBUTING.md). To reproduce the local
macOS arm64 run, create a Python 3.14 virtual environment and install
`models/social_gan/requirements-macos-arm64.lock`. The benchmark pins
are the same in both environments. On Linux, install the CPU build of
PyTorch as specified in the SOP before installing the model requirements.

```sh
python -m pip install -r models/social_gan/requirements.txt
make benchmark PYTHON=python
python models/social_gan/own_unit_tests/test_causal.py
python models/social_gan/run.py pilot
make pilot MODEL=social_gan RUNS=artifacts/social_gan/pilot/runs PYTHON=python
```

The pilot uses two epochs and the full ETH seed 0 test manifest. The
regression suite poisons or removes post-anchor target and neighbor
positions, checks the coordinate transform and displacement calculation,
and exercises the isolated-target path and an adversarial update.

## Formal runs and scoring

```sh
python models/social_gan/run.py formal --jobs 5
make score MODEL=social_gan RUNS=artifacts/social_gan/formal/runs PYTHON=python
make audit MODEL=social_gan RUNS=artifacts/social_gan/formal/runs PYTHON=python
python models/social_gan/run.py formal --replay eth 0
python models/social_gan/handoff.py analyze
python models/social_gan/handoff.py package
python models/social_gan/handoff.py verify
python models/social_gan/handoff.py rescore
make preview MODEL=social_gan PYTHON=python
```

Use `--jobs 1` for a single training worker. Each uses two CPU threads.
Training and export run in the same command. Each fold and seed reloads
the best validation checkpoint for all six test conditions. Use
`--runs PATH` for a fresh output tree or `--resume` to finish a stopped
run with the original settings. Timing runs serially after training and
export. Use `--timing-only` to repeat that pass without training.
The local CPU run did not need a GPU reservation.

The main [contribution](../../submission/contributions/social_gan/)
contains the audited results, evidence and English LaTeX sources. The
full archive is under `artifacts/social_gan/`. Raw data are fetched
separately by `make benchmark`. Full predictions, weights, diagnostic
logs, the archive and built document PDFs stay in ignored paths.
The replay checks the saved arrays bit for bit in the same environment.
The archive rescore regenerates the neutral data and validates the
archived bundles in a fresh copy of the pinned group base.
