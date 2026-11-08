# Reproduce the full-scene Social-STGCNN experiment

The full-scene archive contains the 15 selected formal checkpoints, 90 formal forecast bundles, the ETH pilot and its six bundles, the training and scoring source, the score table, and the review evidence. `inventory.json` identifies the numerical Git revision and the files in the archive. The contribution's `provenance.json` identifies the exact archive and its download location. The [earlier target-readout release](https://github.com/Nine-Three-Cattery/comp5331/releases/tag/social-stgcnn-causal-v2-review-v2) uses a different network and has its own weights and scores.

## Rebuild the group data

Use a fresh directory and the Linux x86-64 CPU environment from the group SOP. The published archive digest is in the contribution's `provenance.json`.

```bash
sha256sum social_stgcnn_full_scene_causal_v2_review.tar
tar -xf social_stgcnn_full_scene_causal_v2_review.tar
cd social_stgcnn_full_scene_causal_v2_review
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
python -m pip install -r benchmark/requirements.txt
make benchmark PYTHON=python
```

The benchmark fetches the pinned six raw files, reconstructs time-disjoint train and validation splits, checks every ordered manifest and computes fold-specific thresholds on training data. The pilot and formal results share the protocol fingerprint printed after `BENCHMARK_OK`.

## Reproduce results from the archive

```bash
python models/social_stgcnn/verify_review.py --receipt artifacts/social_stgcnn/reviewer_replay.json
make audit MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/formal/runs PYTHON=python
python models/social_stgcnn/check_selection.py --runs artifacts/social_stgcnn/formal/runs --data data/causal_v2 \
    --out artifacts/social_stgcnn/reviewer_selection.json
python -m models.social_stgcnn.review_metrics --out artifacts/social_stgcnn/reviewer_metrics.csv
```

The replay checks file and benchmark identities, then re-exports the selected ETH seed-zero clean forecast. It checks all 364 targets and all twenty sampled paths, along with observed histories, masks, ground truth and record order. The shared audit independently calculates all six headline scores for every formal condition. The checkpoint check reads all training epochs, identifies the winner among the 51 recorded candidates in each run, and re-evaluates each selected model on uncached validation scenes using a separate double-precision likelihood. It preserves the published density floor. The owner's displacement check computes min/mean ADE and FDE from the exported coordinates. The turning diagnostic uses the shared moving-step threshold of 0.05 m/s and the training 95th percentile. A reported window can have up to 220 eligible turns across twenty draws.

For the complete per-window motion and spatial tables run `make score MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/formal/runs PYTHON=python` in a separate extracted copy. Then use `python -m models.social_stgcnn.analyze --phase extract`. With Matplotlib installed, `python -m models.social_stgcnn.analyze --phase render` draws the figures from the resulting small tables. The matched ETH plot uses pedestrian 216 at anchor 875, the same seed-zero example as the group reference handoff.

## Compare the actual reference operators

This optional check uses the authors' complete checkout solely as a test oracle. It loads the pinned code and dataset, initializes matched test weights, compares every predicted node, and checks input and parameter gradients, normalization state, a floored Gaussian likelihood and fixed-noise sampled paths. The raw forecast comparison uses the two reshapes and final axis order implemented by the authors. The receipt records the smaller standalone environment used with the historical NetworkX API. From the group source or the extracted archive:

```bash
python3.11 -m venv .oracle-env
.oracle-env/bin/python -m pip install torch==2.2.1 numpy==1.26.4 scipy==1.11.4 pandas==2.3.3 networkx==2.8.8 tqdm
mkdir -p local
git clone https://github.com/abduallahmohamed/Social-STGCNN.git local/authors-social-stgcnn
git -C local/authors-social-stgcnn checkout --detach 333d3a57b4d2705e129b21aefefa09c79b2b9ae1
.oracle-env/bin/python models/social_stgcnn/check_reference.py --reference local/authors-social-stgcnn \
    --out artifacts/social_stgcnn/reviewer_source_parity.json
```

The six cases include a recorded ETH sequence with three pedestrians. The release's loader excludes a lone pedestrian and its graph helper removes the singleton axis. For the course singleton case we check the single-node Laplacian and run the authors' neural operators with that dimension retained. Formal training and export use only the independent model and the group data.
