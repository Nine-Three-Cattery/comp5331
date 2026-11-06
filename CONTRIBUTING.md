# Clone-to-merge SOP

Run commands from the repository root in a Linux x86-64 checkout. On macOS
or Windows, use the group's Linux host or WSL for the reference benchmark.
The six assignments and stable IDs are in [the model registry](submission/models.json).

## 1. Set up the shared benchmark

Clone the group's repository, then run the following. Replace the example
ID with your assigned ID before creating the branch.

```bash
export MODEL=social_lstm
git switch -c "model/$MODEL"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
python -m pip install -r benchmark/requirements.txt
make benchmark PYTHON=python
```

Expected result is `BENCHMARK_OK`. This fetches the pinned coordinates,
builds all five ordered train/validation/test splits, checks their hashes,
tests corruption and metrics, and derives motion thresholds from training.
Use this CPU environment for shared scoring and audit. Record a separate
model environment if training requires CUDA or additional packages.
[PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu/torch/)

## 2. Implement and complete the ETH pilot

Use the code directory in the registry. Commit a README with exact setup,
smoke, train, predict and reproduce commands. Record the model inputs,
normalization, paper differences, optimizer, maximum epochs, checkpoint
rule, budget `K` and GPU estimate before the formal runs. The
[benchmark contract](benchmark/README.md) defines allowed inputs, bundle
fields and the exact run tree. Reuse the shared loader and evaluator. Write
the model, loss and training code independently. Register references,
licenses and any permitted minimal reuse.

The pilot uses **ETH, seed 0, all six conditions and the full test manifest**
with a short training budget. Verify your model's future-invariance test and
coordinate round-trip test. Place the pilot in a distinct run directory and
run:

```bash
make pilot MODEL="$MODEL" RUNS="artifacts/$MODEL/pilot/runs" PYTHON=python
```

Expected result is `BENCHMARK_AUDIT_OK` with `pilot_audit.json`. Record its
elapsed time and peak memory, then reserve GPU time for the formal matrix.

## 3. Run, score and hand off the formal matrix

Use the fixed five folds, seeds 0/1/2 and six test conditions. Keep the
formal validation-selected checkpoint fixed across conditions. From the
CPU environment, run:

```bash
make score MODEL="$MODEL" RUNS="artifacts/$MODEL/formal/runs" PYTHON=python
make audit MODEL="$MODEL" RUNS="artifacts/$MODEL/formal/runs" PYTHON=python
make preview MODEL="$MODEL"
```

Make the required provenance, report, slides and notes before building the
preview. `make score` runs the common evaluator, saves full motion and spatial
diagnostics locally, and writes the 90-row contribution CSV. `make audit`
independently checks its headline values and binds the receipt to the
benchmark code and CSV. Copy `artifacts/$MODEL/benchmark_audit.json` into
your contribution directory. Keep raw bundles, checkpoints, timing logs and
the release archive in ignored paths. Share the full release through the
group's agreed location and record the filename and SHA-256 in provenance.

Use the [submission guide](submission/README.md) and templates for English
report, Beamer frames and notes. Submit one model branch for review. The
editor confirms attribution, the causal and validation tests, an independent
artifact reproduction, and runtime comparability before marking it ready.
Inspect staged files and keep course originals, credentials, raw data and
large artifacts local.

## 4. Integrate and release

The editor merges accepted model branches, writes the comparison and
synthesis, and records one peer review per model. Follow the
[editor checklist](submission/editor/README.md). From the clean committed
final branch, `make release` exports the combined code archive, report,
slides, score table, notes and checksums to `artifacts/group7/`.
