# Shared ETH/UCY contract

The scope and integrity rules are in the
[common specification](../GROUP7_COMMON_DATA_EVALUATION_SPEC.md). This
document is the executable `causal_v2` interface. Start with
[the clone-to-merge SOP](../CONTRIBUTING.md).

## Fixed settings

| Item | Value |
|---|---|
| Data | Pinned raw files and SHA-256 under the Trajectron++ config. `make benchmark` generates `data/causal_v2`. |
| Splits | ETH, Hotel, Univ, Zara1 and Zara2 leave-one-scene-out. Use the generated ordered manifests. Development splits are time-disjoint with a 20-step exclusion on each side of the boundary. |
| Input | Eight observed positions, twelve future positions, 0.4 s spacing, metric coordinates. Target and observed neighbors only. Derive velocity and acceleration after perturbation from the visible window. |
| Formal matrix | Five folds × seeds 0/1/2 × six conditions. Select one checkpoint on clean validation per fold/seed. Record the criterion and tie-break before training. Trajectron++ uses Energy Score, then meanADE, then the earlier epoch. |
| Sampling | `K=1` deterministic; `K=20` probabilistic. Register `K` before the pilot. Use deterministic sampling for a fixed seed and record the actual sampling seed per row. |
| Corruption | `clean_0`, `noise_005`, `noise_010`, `noise_020`, `missing_1`, `missing_2`. Perturb target history only. NumPy `RandomState` seed is `100003*training_seed + 1009*zero_based_manifest_index`. Draw time-major x then y. Missing suffixes use constant-velocity extrapolation from the last two visible points. |
| Scoring | Use the common evaluator. Report min/mean ADE and FDE, Energy Score and diversity, plus the motion and proximity diagnostics. Average within fold and seed, then across seeds, then equally across holdouts. Thresholds come from training only. |

[Manifest lock](fixtures/manifests.json) · [Synthetic corruption fixture](fixtures/corruption.json) ·
[Runtime window IDs](fixtures/runtime_windows.json)

The public fixture uses synthetic coordinates. The manifest lock and
runtime lists carry hashes, counts and window IDs. Fetch benchmark
coordinates separately with the shared setup command.

## Input and output files

Use `independent_tpp.data.load_split(fold, split, data_root)` to access
ordered records and neutral metric positions, and
`independent_tpp.protocol.perturb_history` to create the target observations.
Add `li_junhao_trajectronpp` to your Python import path for those shared
helpers. Model-specific transforms and social context stay in the owner
implementation. Invert transforms before exporting predictions.

For each fold and seed, write the following tree under the `RUNS`
argument. Use the same tree for the separate ETH pilot.

```text
<runs>/<fold>/seed_<seed>/
  config.json
  checkpoint_selection.json
  <selected checkpoint filename>
  evaluation/<condition>/predictions_test_<condition>.npz
```

`config.json` stores the frozen training parameters plus `train_identity`
and `val_identity` from the shared loader. Its digest is
`independent_tpp.protocol.digest_json(config)`. `checkpoint_selection.json`
must include `selection_split: "val"`, `test_used_for_selection: false`,
and `selected.checkpoint` as a filename in that run directory. Include the
candidate validation table and tie-break in the release.

Write the prediction bundle with these arrays:

| Key | Format |
|---|---|
| `samples` | Finite `[N,K,12,2]` metric forecast positions in manifest order |
| `ground_truth` | `float32 [N,12,2]` metric futures, identical across all conditions |
| `observed` | `float32 [N,8,2]` model input positions after perturbation and any fill |
| `observed_mask` | `bool [N,8]` with false for filled positions |
| `records` | `N` JSON strings. Include `window_id`, `scene_name`, `scene_id`, `source_file`, `scene_t`, `anchor_t`, string `target_id`, `fold`, `split`, `seed`, `sample_count`, integer `sampling_seed`, `dt`, `coordinate_unit: "meter"`, `condition`, `sigma` and `missing`. |
| `metadata` | Scalar JSON string. Spread the test `store.identity`, then add `seed`, `sample_count`, `condition`, `level`, `config_hash`, `checkpoint` and `checkpoint_sha256`. |

In JSON, `condition` is `clean`, `noise` or `missing`. The numeric `level`
is 0, noise standard deviation in meters, or suffix length. Each record's
`sigma` and `missing` are zero outside their respective conditions.
The path's condition tag comes from the fixed six-item matrix above.
For a working exporter see
[Trajectron++ prediction entry point](../li_junhao_trajectronpp/scripts/predict_independent.py).

## Runtime and diagnostics

Use the ordered runtime window list, clean inputs, 128 targets, batch 32,
registered sample budget, three warmups and ten measured repetitions for
each fold/seed. Synchronize CUDA at stage boundaries. Measure preprocessing
from in-memory neutral positions, model prediction including sampling,
and end-to-end through metric CPU output. Retain per-repetition records
and the exact host, GPU/CPU, driver and software versions. Compare runtime
only on the same host and device, or present separately labeled device
results. The editor records this policy before synthesis.

`make score` uses the training-derived thresholds in
`data/benchmark_thresholds` and saves per-window, per-scene and spatial
summaries under `artifacts/<model>/metrics`. Proximity is a descriptive
distance check. `make audit` validates all 90 bundles and independently
recomputes their six headline scores. A receipt establishes the checked
interface. The editor separately reviews causal feature use, source
attribution and reproduction on a clean checkout.
