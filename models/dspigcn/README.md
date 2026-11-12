# DSPIGCN — YAN Zhengyang

Independent paper-based implementation for COMP5331 Group 7. Core code, loss,
trainer and anchor sampler are group-owned; shared data/evaluation follow
`causal_v2`. Registered sampling budget: **K=20**. The 15 formal fits are complete and the
90 forecasts passed the common numerical audit. The group editor records
acceptance in the [registry](../../submission/models.json).

## Reproduce

Run from the repository root on Linux x86-64 with Python 3.12. The formal
experiment used an NVIDIA GeForce RTX 4090 and PyTorch 2.7.0+cu128. A
separate CPU environment with PyTorch 2.14.0 scored the saved predictions.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
.venv/bin/python -m pip install -r models/dspigcn/requirements.txt
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128
.venv-train/bin/python -m pip install -r models/dspigcn/requirements-cuda.txt

make benchmark PYTHON=.venv/bin/python
.venv/bin/python -m pytest models/dspigcn/own_unit_tests -q

# Full ETH manifest, two epochs, six conditions; separate from formal runs.
.venv-train/bin/python models/dspigcn/run.py matrix \
  --config models/dspigcn/configs/pilot.json \
  --runs artifacts/dspigcn/pilot/runs --folds eth --seeds 0 --device cuda
make pilot MODEL=dspigcn RUNS=artifacts/dspigcn/pilot/runs PYTHON=.venv/bin/python

# All 15 fits, full manifests, 200 epochs, validation-selected checkpoints.
.venv-train/bin/python models/dspigcn/run.py matrix \
  --runs artifacts/dspigcn/formal/runs --device cuda
make score MODEL=dspigcn RUNS=artifacts/dspigcn/formal/runs PYTHON=.venv/bin/python
make audit MODEL=dspigcn RUNS=artifacts/dspigcn/formal/runs PYTHON=.venv/bin/python
.venv-train/bin/python models/dspigcn/runtime.py \
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

Exact validated Linux dependency closures are in
`requirements-linux-cuda.lock` and `requirements-linux-cpu.lock`. The A800
pilot passed all six conditions; its lightweight evidence is in the matching
submission folder. The supplementary five-fold results, reference environment and audit are in the [group contribution](../../submission/contributions/dspigcn/README.md).

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

## Supplemental candidate experiments

The isolated [candidate variant](provisional_variant/README.md) preserves an
additional implementation and its 15-fit / 90-row local experiment matrix. Its
`dspigcn_candidate` namespace can coexist with this registered model, and its
inference bridge accepts this model's graph batches. Those saved experiments use
a different protocol and must remain outside the common causal_v2 result table.

## Replay the supplementary run

The shared formal experiment used the registered configuration at revision `020466c` and NVIDIA GeForce RTX 4090, with 2.7.0+cu128 for model execution. A separate Linux x86-64 CPU environment performs shared scoring. The contribution's [provenance](../../submission/contributions/dspigcn/provenance.json) identifies the downloadable archive. Extract it into an empty directory, activate the CPU environment from the group SOP, and run this command from the extracted root. Set `PREDICTION_PYTHON` to the absolute path to the matching model interpreter.

```bash
python models/dspigcn/verify_release.py --prediction-python "$PREDICTION_PYTHON" --device cuda --receipt extracted_replay.json
```

The verifier rebuilds the shared data, checks the complete score table, exports forecasts again for every selected checkpoint, and compares all 90 bundles. It reports the measured coordinate difference and checks the observed inputs, masks, targets and ordered records for exact equality.
