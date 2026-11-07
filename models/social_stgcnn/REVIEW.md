# Reproduce the Social-STGCNN review artifact

This private course artifact contains the 15 validation-selected formal checkpoints, 90 original test bundles, the ETH seed-0 pilot and its six bundles, both group audit receipts and the formal score table. Each run includes its training configuration, per-epoch likelihoods, checkpoint selection table, hardware timings and resource record. `inventory.json` binds these files to their SHA-256 digests. The contribution's published `provenance.json` records the SHA-256 of the completed archive.

## Extract and prepare the data

Use a Linux x86-64 CPU host with Python 3.12. Compare the first command's digest with the contribution's published archive digest before extracting. Run the following from the directory containing the archive.

```bash
sha256sum social_stgcnn_causal_v2_review.tar
tar -xf social_stgcnn_causal_v2_review.tar
cd social_stgcnn_causal_v2_review
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
python -m pip install -r benchmark/requirements.txt
make benchmark PYTHON=python
```

The shared target fetches six coordinate files from its pinned raw release, rebuilds the 15 ordered manifests and derives motion thresholds from each training split. It prints `BENCHMARK_OK` after checking the locked identities and common evaluator fixture. The raw source digests are in `li_junhao_trajectronpp/configs/causal_v2_raw_sha256.json`. Manifest hashes and counts are in `benchmark/fixtures/manifests.json`.

## Replay a selected checkpoint

```bash
python models/social_stgcnn/verify_review.py \
  --receipt artifacts/social_stgcnn/reproduction_check.json
```

The verifier checks the inventoried files and scored run identities, loads the formal ETH seed-0 selected checkpoint, and exports `clean_0` to a temporary directory. It uses the archived thread count and batch size. It compares all 20 sampled trajectories per target, ground truth, input history, mask, ordered records, manifest window IDs and metadata. The JSON receipt gives the number of different numeric entries and the maximum absolute position difference in meters. An exact replay reports `passed`. Differences produce a nonzero exit and their measurements remain in the receipt.

The formal training, forecasts and timing used `2b2d57de6d5ff009f1388e1f0d9baf948e9db4ac`. Pilot training and all six pilot forecasts used `c37d1211104d9c125775bd4c7b5044083c04c1f1`. The team reran the pilot timing at the formal revision to include on-demand window and neighbor discovery. The pilot's `runtime.json` contains that measurement. Its configuration retains the original `run.py` digest. The model, window builder and exporter are identical across these two revisions. `inventory.json` records both revisions. The shared project base is `df016def5890898b70d372fb0e8e2bc142121330`.

All fifteen reported formal timings use the same physical CPU node. The driver from revision `1987673` calls the frozen timing routine with the selected checkpoints and records the three stage boundaries. To remeasure on a new host, run `python -m models.social_stgcnn.retime --threads 2` on a separate extracted copy. The CPU, software and thread count accompany each measurement.

The prespecified analyses of the scored outputs are versioned separately at `9cb6ac43b8c13bfcc3e8d65de891030fc35b8b23`. To run with a fresh Git export, use a clone of the group repository containing the formal revision. From the extracted review directory with the environment active, run the following. Set `REVIEW_REPO` to the clone's directory.

```bash
mkdir -p ../social_stgcnn_clean_source
git -C "$REVIEW_REPO" archive 2b2d57de6d5ff009f1388e1f0d9baf948e9db4ac | tar -x -C ../social_stgcnn_clean_source
make -C ../social_stgcnn_clean_source benchmark PYTHON="$(command -v python)"
python models/social_stgcnn/verify_review.py --source ../social_stgcnn_clean_source \
  --receipt artifacts/social_stgcnn/fresh_source_reproduction_check.json
```

## Recheck the entire published score table

```bash
make pilot MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/pilot/runs PYTHON=python
make audit MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/formal/runs PYTHON=python
```

The shared audit validates all 90 formal bundles against the rebuilt raw-derived windows and independently recomputes the six headline metrics in `submission/contributions/social_stgcnn/results.csv`. The selected-checkpoint replay and full audit answer different questions. The replay checks inference from saved weights. The full audit checks all stored forecasts and scores.

To check the validation choice for every set of trained weights, run
`python models/social_stgcnn/check_selection.py --runs artifacts/social_stgcnn/formal/runs --data data/causal_v2`.
It re-evaluates clean validation trajectories with the full graph and a separate
double-precision Gaussian likelihood, then compares the stored winning epoch
and likelihood with the recorded candidates.
