# Social-LSTM — deterministic K=1 causal_v2 interface

Owner: WANG Yifan. **The deterministic K=1 causal_v2 matrix and local common audit are complete.** All 15 runs used the frozen 12-epoch budget and validation-only selection, and all 90 prediction bundles passed the shared interface/score audit. Registry/editor status remains pending for independent review, reference-environment reproduction and runtime comparison policy. No shared registry, contract or fixture was modified.

Two distinct branches are retained:

- `causal/`: independent deterministic social-grid recurrent predictor, common loader/corruption and neutral K=1 export; the branch for the registered shared comparison.
- `legacy/`: preserved exploratory Gaussian K=20 pipeline, seeds11/22/33, different splits/corruption. Its numbers are excluded from the shared comparison and cannot be relabeled as K=1.

## Shared K=1 design and frozen budget

Eight observed metric positions predict twelve future positions at dt=.4s. Hidden64, embeddings32, grid4x4 in a 4m square, self excluded. Admit neighbors only if all eight observed positions exist and they come within6m of the target at any observed time after target corruption. Neighbor admission never uses future availability. All admitted actors roll forward jointly; only target labels supervise loss.

The deterministic mean uses the fixed average of the final three observed displacement intervals plus a learned recurrent residual. It does not accumulate a sampled displacement into the next mean. There is no Gaussian sampling or teacher forcing. Training uses full closed-loop metric position MSE, Adam.001, weight decay.0001, clipping10, batch32, seeded actor-count bucketing without dropping windows. The formal default is12 epochs; the separate full-population ETH seed0 pilot uses1 epoch. Select lowest clean validation meanADE, with earlier epoch for exact ties. Keep the selected checkpoint fixed across all six conditions. The design record is `configs/causal_v2.json`.

This is an **adapted deterministic Social-LSTM**, not an exact reproduction of the paper's Gaussian position model. Grid dimensions/hidden size and displacement anchor differ. A completed interface or a passed unit test does not establish good forecasting quality; report actual shared validation/test results and any limits.

## Environment and paths

Follow the repository benchmark SOP for the reference Linux/Python3.12/PyTorch2.14 environment and common dependencies. The available local exploratory environment is Python3.13/PyTorch2.11.0/NumPy2.4.4 on macOS arm64; this is an explicit environment deviation. The CLI records environment, device/chip and normalized process peak RSS. Local CPU/MPS timing must be labeled separately from reference-host evidence.

To reproduce the recorded Mac environment, use Python 3.13.12 and install `models/social_lstm/requirements-causal-macos.txt` in a separate virtual environment. Do not mix those exact local pins with the different reference pins in one environment.

Run commands below from the repository root, with an interpreter containing the shared `benchmark/requirements.txt` dependencies and PyTorch. CUDA, CPU and MPS devices can be passed explicitly. Do not infer that MPS is always faster. Fixed seeds do not guarantee bitwise-identical MPS training across runs or software versions. Deterministic K=1 describes the forecast rule (no sampling); saved-checkpoint replay and numeric tolerances should be used to assess reproduction.

```bash
make benchmark PYTHON=python
python -m unittest discover -s models/social_lstm/own_unit_tests -v
```

The shared gate validates the unchanged data/manifests. The owner tests cover both branches; causal checks exercise actual target/neighbor future and stored-derivative invariance, future-presence invariance, exact target-only corruption/index seeds, causal fill, metric translation/roundtrip, deterministic K1, padding/batch isolation, grid boundaries/gradient, closed-loop backward and complete bucket coverage.

## Full ETH pilot, prediction and formal matrix

```bash
python models/social_lstm/causal/run.py pilot --runs artifacts/social_lstm/pilot/runs --device cpu
make pilot MODEL=social_lstm RUNS=artifacts/social_lstm/pilot/runs PYTHON=python
```

`pilot` trains ETH seed0 for one epoch using full shared train/validation populations and exports the **entire** test manifest for all six conditions. It never uses a test subset as a shared pilot. Choose `--device mps` only on a supported local host; a runtime estimate comes from the actual pilot and training log, not an assumed GPU speed.

The completed MPS ETH seed0 pilot used one full epoch: 35.18s, clean-validation ADE .46917m, process peakRSS769MiB. Its six full-test exports passed the common pilot audit. These local Mac measurements are not Linux reference-host timing. The formal12-epoch matrix has completed and passed the local shared audit. Reproduction command:

```bash
python models/social_lstm/causal/run.py formal --runs artifacts/social_lstm/formal/runs --device cpu
make score MODEL=social_lstm RUNS=artifacts/social_lstm/formal/runs PYTHON=python
make audit MODEL=social_lstm RUNS=artifacts/social_lstm/formal/runs PYTHON=python
```

The formal default is five folds x seeds0/1/2 x12 epochs, with six exports per selected model. Any pre-formal budget revision must be explicit in the command/config and documented before starting formal training. The output tree is `<runs>/<fold>/seed_<seed>/` with config, checkpoint selection/candidate table, selected checkpoint, training log/summary, and `evaluation/<tag>/predictions_test_<tag>.npz`.

For one run or prediction on CPU from a selected checkpoint trained on another device:

```bash
python models/social_lstm/causal/run.py train --runs artifacts/social_lstm/formal/runs --fold eth --seed 0 --epochs 12 --device cpu
python models/social_lstm/causal/run.py predict --runs artifacts/social_lstm/formal/runs --fold eth --seed 0 --device cpu
python models/social_lstm/causal/run.py runtime --runs artifacts/social_lstm/formal/runs --fold eth --seed 0 --device cpu
```

`predict` writes all six common-schema bundles `[N,1,12,2]` in manifest order, with masks, original labels, records, loader identities, config hash and selected checkpoint hash. Deterministic sampling is absent; each row still records an integer sampling seed. `runtime` uses the exact ordered128-target fixture, four batches of32, three warmups and ten measurements, device synchronization and metric CPU output. Run it for every fold/seed on the agreed host/device. It records coupled deterministic prediction time and does not invent a Gaussian sampling stage. Training checkpoints and full predictions stay in ignored root `artifacts/`.

## Legacy reproduction only

The old environment is pinned in `requirements-legacy.txt`. Legacy artifacts default to ignored `models/social_lstm/artifacts/legacy/`; optionally set `SOCIAL_LSTM_LEGACY_WORKDIR` to another local directory. Do not point it at shared data or an immutable release.

```bash
python models/social_lstm/legacy/smoke.py
python models/social_lstm/legacy/download_data.py
python models/social_lstm/legacy/download_data.py --verify-only
python models/social_lstm/legacy/data.py
python models/social_lstm/legacy/run.py train --scene eth --seed 11 --epochs 1 --tag _exploratory_pilot
python models/social_lstm/legacy/run.py evaluate --scene eth --seed 11 --tag _exploratory_pilot --conditions clean --limit 3 --device cpu
```

The legacy download uses curl, fetches only six pinned coordinate files and verifies SHA256. The limited export above is only a synthetic/partial workflow check, explicitly marked partial, and is not a common benchmark pilot. Full old reproduction uses `python models/social_lstm/legacy/reproduce.py --evaluation-device cpu`; full arrays are not committed.

The old residual sampled-velocity feedback amplified innovations; typical sampled error exceeded CV. A validation-only anchored control did not improve, so it was not a validated repair. That old experiment is separate from the new deterministic model and its future shared evidence.

Read [contract_gaps.md](contract_gaps.md), [paper_reading_notes.md](paper_reading_notes.md), [independent_design.md](independent_design.md), [source_inventory.md](source_inventory.md) and [implementation_provenance.md](implementation_provenance.md). `reused_fragments.csv` is header-only: no copied upstream model code. Report, slides and three-minute notes live in `submission/contributions/social_lstm/`.

## Regenerate contribution evidence

After formal scoring and all fifteen runtime measurements, run `python models/social_lstm/analysis/summarize.py` and `python models/social_lstm/analysis/figures.py`. These write lightweight macro, constant-velocity, runtime, crowd/turn and figure evidence under the matching contribution directory. The CV baseline uses the same clean test manifests and the common headline metric function. The qualitative selection rule is based on ground-truth turning among the fixed runtime IDs and is disclosed as descriptive, not representative.
