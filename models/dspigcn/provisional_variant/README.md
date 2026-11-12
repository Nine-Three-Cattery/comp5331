# DSPIGCN candidate variant — Zhengyang Yan (COMP5331)

Supplemental implementation and experiments prepared before repository access.
The [registered DSPIGCN workflow](../README.md) remains the group's `causal_v2`
implementation. This directory preserves the separate candidate implementation
and its evidence. Its Python namespace is `dspigcn_candidate`, so both models
can be imported in one process. Use a separate environment for its dependencies.

**Current status:** all five folds × three seeds have completed under the
candidate protocol, producing 90 evaluation rows. The full saved-artifact audit
and original 17 unit/integration tests pass; each checkpoint reproduces its first clean
window exactly. See the [report](docs/REPORT.md) and
[audit evidence](results/provisional_main/audit.json). Repository inspection and
paper review have now identified material differences from both `causal_v2`
and the paper. These 90 rows must not enter the common comparison table.
See [integration and protocol differences](docs/TEAM_INTEGRATION.md),
[paper equation review](docs/PAPER_CHECK.md), and
[implementation notes](docs/IMPLEMENTATION.md). The original audit covers the
candidate protocol only; the additional integration test exercises the actual
group loader, history-only context, batch ordering and namespace coexistence.
The published package passes 18 tests including that new regression.

## Environment

From this directory, use a separate environment (the CUDA wheel must match your
machine). Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

`requirements-validated.txt` records the versions used for the local experiments.
Do not install dependencies into a teammate's environment. `python -m
dspigcn_candidate.runner --help` and the installed `comp5331-dspigcn-candidate` command are equivalent.

## Data and protocol

The loader accepts a directory containing `biwi_eth.txt`, `biwi_hotel.txt`,
`students001.txt`, `students003.txt`, `crowds_zara01.txt`, `crowds_zara02.txt`, or
the author's layout `<scene>/test/<filename>`. Files contain whitespace-separated
`frame pedestrian x y` in meters; optional appended `vx vy` are ignored.

The experiment data come from the released artifact, pinned to commit
`d38fd905e51d401a5e812ae561185ac4a423c2e0`. To obtain that release:

```bash
git clone https://github.com/shangfeng2000/PI-STGCN.git data/author
git -C data/author checkout d38fd905e51d401a5e812ae561185ac4a423c2e0
```

Raw files/checkpoints from the author's repository are reference material; this
workflow does not import its Python modules or load its pretrained models.

The provisional protocol uses 8 observations, 12 future steps, 0.4 s sampling,
all complete trajectories including single pedestrians, stride 1, chronological
80% training segments and a 19-frame boundary gap before validation segments.
Only the other four scenes enter a fold's training/validation sets. See
[PROTOCOL.md](docs/PROTOCOL.md) before combining results with other models.

```bash
python -m dspigcn_candidate.runner prepare --data-root data/author/datasets --scene eth --output outputs/manifests/eth
python -m dspigcn_candidate.protocol outputs/manifests/eth/manifest_test.json /path/to/team/manifest_test.json
```

The second command must pass after the team exports the same manifest schema.
Equal counts alone are insufficient: it checks files, settings, sample IDs and
order. The team can adapt its existing manifest into this schema without changing
any model code. Protocol differences must be resolved before comparing scores.

## Train, evaluate and combine

```bash
# Short end-to-end pilot only (stride 5 and two epochs, not a final benchmark).
python -m dspigcn_candidate.runner train --data-root data/author/datasets --scene eth --seed 0 --config configs/pilot.json --output outputs/pilot_eth_seed0
python -m dspigcn_candidate.runner evaluate --data-root data/author/datasets --checkpoint outputs/pilot_eth_seed0/best.pt --output outputs/pilot_eth_seed0/evaluation
python -m dspigcn_candidate.runner benchmark --checkpoint outputs/pilot_eth_seed0/best.pt --pedestrians 32 --output outputs/pilot_eth_seed0/runtime.json

# Candidate experiment matrix, separate from the group's causal_v2 runs: 15 fits,
# each evaluated on all 6 conditions with 20 samples (90 rows in results.csv).
python -m dspigcn_candidate.runner run-all --data-root data/author/datasets --config configs/default.json --output outputs/main
python -m dspigcn_candidate.aggregate outputs/main/results.csv --output outputs/main/seed_summary.csv
python -m dspigcn_candidate.audit outputs/main --matrix --data-root data/author/datasets --output outputs/main/audit.json
python -m dspigcn_candidate.report outputs/main --destination results/main --report docs/REPORT.md
```

The matrix audit checks every saved prediction against source trajectories and
corruptions, recomputes accuracy/motion scores, checks checkpoint identity and
validation selection, and requires all 15 scene/seed combinations. Passing it
does not certify that the provisional protocol matches the other five models.

Use a new output directory for a new training run; existing checkpoints are never
silently overwritten. CPU is supported with `--device cpu`. Checkpoints are
selected using validation ADE of the mean trajectory, never test performance.
The default limit is 100 epochs with patience 15; seeds are 0, 1, 2. No optional
ablations are included ahead of the mandatory experiment matrix.

`run-all --resume` reuses fully completed phases after checking configuration and
source manifests. Each train/evaluate/benchmark phase runs in its own process to
release native resources on shared filesystems. JSON/CSV/checkpoint writes use
temporary files followed by atomic replacement. An interrupted training run is
not silently treated as complete: its best-only checkpoint has no optimizer/RNG
resume state. Preserve that incomplete seed directory separately, then rerun only
that seed through `--resume`; completed seeds are retained.

Each run exports:

- `best.pt`: model, train-only GMM anchors, configuration, seed, validation epoch,
  training thresholds, train/validation manifest hashes.
- `run.json`, `history.csv`, `manifest_{train,val}.json`: provenance and training.
- `evaluation/results.csv`: one auditable row per scene, seed and condition.
- `evaluation/<condition>/predictions.zip`: numbered NPZ windows, each containing
  `observations`, `clean_observations`, `target`, `predictions`, `sample_ids`,
  `frames`, `pedestrians`. Predictions use `[20, 12, N, 2]`, in meters. Streaming
  archives avoid creating thousands of small files on shared storage.
- `per_trajectory.csv`, fixed-example plots, worst-window plots, `failures.json`.
- `runtime.json`: separate preprocessing, prediction and sampling timings at a
  fixed pedestrian count, with synchronization, warmup and hardware recorded.

NPZ files contain no pickled objects and can be loaded with `allow_pickle=False`.
Outputs are ignored by Git so checkpoints and prediction archives do not bloat
teammates' clones. Deliver them separately with the final experiment bundle;
small audited tables/figures can be copied into a named `results/` directory.

## Import interface

```python
from dspigcn_candidate.interface import Predictor

predictor = Predictor("outputs/pilot_eth_seed0/best.pt", device="cpu")
# obs_traj: NumPy or torch [8, total_pedestrians, 2], absolute meters.
# Optional seq_start_end isolates multiple scenes, as in Social-GAN collation.
predictions = predictor.predict(obs_traj, num_samples=20,
                                seq_start_end=seq_start_end, seed=0)
# torch [20, 12, total_pedestrians, 2], on the input tensor's device.
```

This bridge establishes an explicit contract. `GroupPredictor` in
`dspigcn_candidate.group_interface` additionally accepts the existing owner's
`GraphDataset`/`collate` batch and returns `[N,K,12,2]` metric predictions.
See [the integration example](docs/TEAM_INTEGRATION.md). This inference bridge
does not convert the saved candidate experiments into common-protocol results.
The package's local tests
verify scene isolation, no mutation of inputs/caller RNG and singleton-pedestrian
support. Importing the inference interface does not alter environment variables
or global deterministic-algorithm settings. Explicit train/evaluate commands set
their own seeds and deterministic settings.

## Attribution and report

See [THIRD_PARTY.md](docs/THIRD_PARTY.md), [IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
and [REPORT.md](docs/REPORT.md). The implementation and documentation were prepared
with AI coding assistance; this assistance must remain acknowledged in the final
course report. Each member must personally verify their proposal declaration.
