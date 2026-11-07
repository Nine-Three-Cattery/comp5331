# Social-LSTM — pending shared contribution

Owner: WANG Yifan. **Status: pending. This is a legacy/exploratory K=20 implementation, not the registry's deterministic K=1 contribution and not a causal_v2 adapter.** Its results must not enter the common comparison. No shared registry, protocol or fixture is changed, and no shared audit receipt is claimed.

This directory preserves the owner's independently written social-pooling model, Gaussian loss and old training/evaluation pipeline for inspection and reproduction. Model quality remains unresolved: sampled velocity residuals amplify innovations over the horizon; typical sampled error was worse than CV, and a one-fold anchored-skip control did not improve validation. Engineering checks do not establish faithful paper reproduction or a validated repair.

## Environment and artifact paths

Run from the repository root. Use a separate exploratory environment:

```bash
python3.13 -m venv models/social_lstm/.venv
models/social_lstm/.venv/bin/python -m pip install -r models/social_lstm/requirements-legacy.txt
```

Below, `python` means that environment's interpreter. The old experiment used Python3.13, PyTorch2.11.0 and NumPy2.4.4; these differ from the shared Linux/Python3.12/PyTorch2.14.0 environment. CPU is the portable default. MPS is optional local exploration, not shared runtime evidence.

Data, processed windows, checkpoints and predictions default to ignored `models/social_lstm/artifacts/legacy/`. Set `SOCIAL_LSTM_LEGACY_WORKDIR` to another local artifact root if necessary. Configs stay in this source directory. Do not point the override at the shared causal_v2 store or an immutable historical release.

## Tests and smoke — no dataset needed

```bash
python -m unittest discover -s models/social_lstm/own_unit_tests -v
python models/social_lstm/legacy/smoke.py
```

Expected tag: `LEGACY_SMOKE_OK_NOT_SHARED_AUDIT`. Nine owner tests cover Gaussian NLL, grid isolation/gradient, metric oracles, causal missing fill, target-and-neighbor future-coordinate invariance, identity-metric/translation roundtrip and backward/inference shape. They test the actual legacy visible-history boundary, not an unimplemented shared adapter or the shared benchmark gate.

## Download, prepare, train and predict

The downloader needs `curl`; it fetches only six coordinate text files from a pinned release and checks every SHA256. It fetches no upstream model code, weight or paper PDF.

```bash
python models/social_lstm/legacy/download_data.py
python models/social_lstm/legacy/download_data.py --verify-only
python models/social_lstm/legacy/data.py
python models/social_lstm/legacy/run.py train --scene eth --seed 11 --epochs 1 --tag _exploratory_pilot
python models/social_lstm/legacy/run.py evaluate --scene eth --seed 11 --tag _exploratory_pilot --conditions clean --limit 3 --device cpu
```

`evaluate` is also the prediction/export command. The example is a three-window smoke export, marked `is_partial=true`; it is not the required full shared ETH seed0 pilot. Drop `--limit 3` for the full legacy ETH population and omit `--conditions clean` for all six legacy conditions. Bundles use `pred[K,N,12,2]`, not the shared schema.

For the complete old five-fold, seeds11/22/33, 12-epoch workflow:

```bash
python models/social_lstm/legacy/reproduce.py --evaluation-device cpu
python models/social_lstm/legacy/diagnostics.py
python models/social_lstm/legacy/verify_exports.py models/social_lstm/artifacts/legacy/runs/eth_seed11/evaluation/clean_predictions.npz
```

With an artifact-root override, adjust the final export path accordingly. `reproduce.py` trains on CPU, evaluates on the requested device, and reuses matching completed runs. Interrupted runs restart from their seed. Run the small pilot first. This upload started no full retraining and contains no raw data, weights or full predictions.

## Frozen exploratory settings and pending work

Eight observed metric positions, twelve future positions, dt=.4s, no maps or fitted normalization. Hidden128, embeddings64, 8x8 social sum grid in a 4m square, correlated Gaussian displacement head, residual sampled-velocity mean, Adam .001, dropout .1, clipping10, 12 epochs and lowest clean-validation teacher-forced NLL checkpoint. Exact settings are in `configs/legacy.json`. K=20 and seeds11/22/33 are legacy choices. Asking the Gaussian sampler for one random path does not define the registered deterministic K=1 model.

Read [contract_gaps.md](contract_gaps.md). Pending: deterministic K=1 design, shared loader/ordered IDs, target-only corruption and masks, shared bundle/evaluator, causal_v2 ETH seed0 pilot, formal seeds0/1/2, score/audit receipt and reference-host runtime. Old local timings are not the future shared GPU estimate.

Read [paper_reading_notes.md](paper_reading_notes.md), [independent_design.md](independent_design.md), [source_inventory.md](source_inventory.md) and [implementation_provenance.md](implementation_provenance.md). The external-code table [reused_fragments.csv](reused_fragments.csv) is header-only: **no copied code** from upstream model repositories. Report, slides and three-minute notes are maintained in `submission/contributions/social_lstm/`.
