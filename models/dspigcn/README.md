# DSPIGCN — YAN Zhengyang

Independent paper-based implementation for COMP5331 Group 7. Core code, loss,
trainer and anchor sampler are group-owned; shared data/evaluation follow
`causal_v2`. Registered sampling budget: **K=20**. Integration status remains
pending until the complete matrix, artifact reproduction and editor review.

## Reproduce

Run from the repository root on Linux x86-64. Create an isolated Python 3.12
environment. Use PyTorch 2.14 CPU for common scoring. The A800 host has a
CUDA 12.8 compatibility runtime, so train in a separate environment with
PyTorch 2.7.0+cu128 and `requirements-cuda.txt`. Do not upgrade a
shared machine's driver. The macOS environment lock is for local smoke tests.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r models/dspigcn/requirements.txt
make benchmark PYTHON=.venv/bin/python
.venv/bin/python -m pytest models/dspigcn/own_unit_tests -q

# Full ETH manifest, two epochs, six conditions; separate from formal runs.
.venv/bin/python models/dspigcn/run.py matrix \
  --config models/dspigcn/configs/pilot.json \
  --runs artifacts/dspigcn/pilot/runs --folds eth --seeds 0 --device cuda
make pilot MODEL=dspigcn RUNS=artifacts/dspigcn/pilot/runs PYTHON=.venv/bin/python

# All 15 fits, full manifests, 200 epochs, validation-selected checkpoints.
.venv/bin/python models/dspigcn/run.py matrix \
  --runs artifacts/dspigcn/formal/runs --device cuda
make score MODEL=dspigcn RUNS=artifacts/dspigcn/formal/runs PYTHON=.venv/bin/python
make audit MODEL=dspigcn RUNS=artifacts/dspigcn/formal/runs PYTHON=.venv/bin/python
.venv/bin/python models/dspigcn/runtime.py \
  --runs artifacts/dspigcn/formal/runs --output artifacts/dspigcn/formal/runtime.csv \
  --device cuda
```

`train`, `predict` and `matrix` are separate CLI commands. `--folds` and
`--seeds` schedule specific runs. Repeating a training command resumes the
last completed epoch with optimizer/scheduler/RNG state; config or source
changes are rejected. Prediction always loads `best.pt`, never the last epoch.
Do not overwrite the formal directory with pilot configurations.

The neutral prediction bundles contain metric positions `[N,20,12,2]`,
unaltered future labels, perturbed histories/masks, ordered JSON records and
config/checkpoint/manifest digests. Shared evaluation imports no model code.

For this A800 training environment:

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128
.venv-train/bin/pip install -r models/dspigcn/requirements-cuda.txt
```

Use `.venv-train/bin/python` for train/predict/runtime and `.venv/bin/python`
for shared score/audit. Checkpoint weights are ordinary state dictionaries.

## Design and evidence

- [Paper equations and artifact discrepancies](paper_reading_notes.md)
- [Causal inputs, units, registered choices and departures](independent_design.md)
- [Source inventory](source_inventory.md) and [assistance/provenance](implementation_provenance.md)
- `own_unit_tests/`: target/neighbor future invariance, derivative units,
  polynomial derivative, exact likelihood, graph normalization, neighbor
  permutation/padding/translation, anchor conservation, batching/seeds and
  tiny optimization with gradients through both streams and the PINN.

The model uses degree-four physics regularization, not a hard safety
constraint. Close approaches are spatial diagnostics. Anchor samples are
dependent; neither low minimum error nor Energy Score proves calibration.
Reports must distinguish completed pilot checks from formal measurements and
must not compare runtime across dissimilar hardware as an architecture effect.

Large raw data, reference checkouts, weights, predictions and release archives
stay under ignored `data/`, `local/` or `artifacts/`. English contribution
sources belong to `submission/contributions/dspigcn/`.
