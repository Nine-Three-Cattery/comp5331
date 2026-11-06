# LI Junhao — independent Trajectron++ causal_v2 release

This is the Group 7 trajectory-only pedestrian implementation. The release
uses window-local causal motion features, the locked ETH/UCY manifests, and
the model-independent evaluator. The reference code and pre-causal results
are historical materials in the course workspace. They are outside this
release.

## Start here

In the shared repository, use the
[group report and slide contribution](../submission/contributions/trajectronpp/README.md)
and build it with the [group masters](../submission/README.md). The committed
contribution carries the lightweight result snapshot and artifact checksum.
Model code remains at this path because the audited workflow and archive
use it as their entry point.

The full reproduction archive includes the following technical companion
materials. Generated PDFs and the full output tree are local artifacts in
a source checkout.

- `report/TrajectronPP_LI_Junhao_Technical_Report.pdf` and
  `report/TrajectronPP_5min_Slides.pdf`
- [Technical speaker notes](report/TrajectronPP_5min_Speaker_Notes.md) and
  [defense notes](report/TrajectronPP_Defense_Notes.md)
- `outputs/causal_v2/results/group7_trajectronpp.csv`,
  `outputs/causal_v2/prediction_index.csv` and
  `outputs/causal_v2/corruption_fixture.json`
- [Independent design](independent_design.md), [sources](source_inventory.md),
  [paper notes](paper_reading_notes.md), and [provenance](implementation_provenance.md)

## Verify the extracted release

Run commands from the extracted project root, one level above
`li_junhao_trajectronpp`. The package includes the causal neutral data and
cache, selected and candidate checkpoints, and all 90 test predictions.

The recorded environment is Linux x86-64, Python 3.12.14 and PyTorch
2.14.0+cu130. The package list is in `requirements-causal-v2.lock`.
Use a CUDA-capable NVIDIA GPU and a driver compatible with the locked
PyTorch build for the exact prediction check. The stored tables can be
recomputed on CPU. A different GPU or kernel may change the last bits of
a newly sampled prediction.

```bash
python -m pip install -r li_junhao_trajectronpp/requirements-causal-v2.lock
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python li_junhao_trajectronpp/scripts/verify_causal_release.py
```

This verifies the internal SHA-256 index, runs the metric and causality
fixtures, repeats the ETH seed 0 clean prediction, and recomputes the six
headline metrics from all stored predictions. It writes the reconstructed
table and a verification record to the project root. The original audit is
available separately at `outputs/causal_v2/audit.json`.

To run just the prediction or the common evaluator:

```bash
causal_scripts=li_junhao_trajectronpp/scripts
causal_run=li_junhao_trajectronpp/outputs/causal_v2/runs/eth/seed_0
python "$causal_scripts/predict_independent.py" --fold eth --seed 0 \
  --checkpoint selected --run-dir "$causal_run" --device cuda --output /tmp/causal_eth.npz
python "$causal_scripts/evaluate.py" --predictions /tmp/causal_eth.npz \
  --thresholds li_junhao_trajectronpp/outputs/causal_v2/thresholds/eth.json \
  --output-dir /tmp/causal_eth_metrics
```

## Rebuild data and the matrix

Fetch the six coordinate files at the pinned data release, using the SHA-256
hashes in `configs/causal_v2_raw_sha256.json`. The fetcher downloads only
the text files. Preserve the shipped cache if using the stage signatures to
resume the delivered runs. For a new run, use a fresh output directory.

```bash
causal_scripts=li_junhao_trajectronpp/scripts
python "$causal_scripts/fetch_causal_raw.py" data/eth_ucy_raw
python "$causal_scripts/prepare_neutral_data.py" --raw-root data/eth_ucy_raw
python "$causal_scripts/validate_neutral_data.py" --data-root data/causal_v2
python "$causal_scripts/cache_independent_data.py"
python "$causal_scripts/compute_independent_thresholds.py"
python "$causal_scripts/run_independent_formal.py" --device cuda \
  --output-dir li_junhao_trajectronpp/outputs/causal_v2_reproduction
```

The driver uses five holdouts, seeds 0–2, 100 epochs, batch 256 and epoch-10
through epoch-100 candidates. Each stage only reuses files whose hashes and
input signature match. Leave the delivered result root intact when
retraining. To rebuild the published analysis and LaTeX from the delivered
matrix, run the finalizer below. It uses the delivered fold checkpoints for
runtime, so GPU timing will be remeasured.

```bash
bash li_junhao_trajectronpp/scripts/finalize_causal_results.sh
```

Use `bash li_junhao_trajectronpp/report/build_latex.sh` to compile just the
English PDFs with the delivered result macros and figures. A LaTeX setup
with pdfLaTeX, Beamer and the packages listed in the source is required.

## Common data and evaluation contract

- Forecast twelve future positions from eight observed points at 0.4 s.
  Coordinates are meters. Windows are ordered exactly as in the manifest.
- The five holdouts are ETH, Hotel, Univ, Zara1 and Zara2. The development
  splits are time-disjoint with a 20-step exclusion on each side of the
  temporal boundary. There is one manifest per fold and split.
- Perturb only the target history. The fixture gives five fixed window
  IDs, the six conditions, exact seed inputs, masks and positions.
  The seed for a window is `100003 * training_seed + 1009 * manifest_index`.
  Use NumPy `RandomState`, time-major x then y. The standard deviations
  are 0.05, 0.10 and 0.20 m. Missing suffixes have length 1 or 2 and are
  filled from the last two visible positions. Neighbor history stays
  available through the anchor.
- NPZ arrays are `samples[N,K,12,2]`, `ground_truth[N,12,2]`,
  `observed[N,8,2]` and boolean `observed_mask[N,8]`. The observed
  positions include any causal fill. The `records` JSON strings identify
  windows and per-batch sampling seeds. The scalar `metadata` JSON
  carries the condition, split, configuration, manifest, input and
  checkpoint identities. Forecast sampling uses
  `training_seed + floor(manifest_index / 128)` for the model-specific
  PyTorch batch RNG.
- The common evaluator reads neutral bundles, including deterministic
  single-sample bundles with the same scene/time records. Include
  `scene_name` at source-file granularity and `scene_t` for spatial
  grouping, and `condition`, `sigma` (meters, otherwise 0), and
  `missing` (suffix length, otherwise 0) in each record.
  The independent
  pipeline exposes fitting, validation-only selection, metric prediction
  and export through separate CLI steps. Shared corruption and kinematic
  conventions live in `independent_tpp/protocol.py`.
- For the runtime comparison, use the fixed 128 target windows per fold
  listed in `runtime.json`, batch size 32, K=20, three warmups and ten
  synchronized measurements. Preprocessing includes window, graph,
  collation and GPU transfer. Inference includes sampling and integration.
  End-to-end also includes CPU readback. Disk and one-time setup are
  outside the timed interval.
- Report per-window metrics and per-fold means across seeds. Headline
  values average the fold means equally. Best-of-20 error, mean error,
  Energy Score and diversity describe different properties of the
  forecast. Spatial proximity is descriptive for separately sampled
  targets.

To join other models, match ordered window IDs, masks, metric definitions,
checkpoint selection policy and runtime boundaries. The handoff CSV
includes protocol, prediction path, checkpoint digest and manifest hash.
