# Social-STGCNN

Zongmin Zhang, COMP5331 Group 7. This model implements the executed operators of Mohamed et al.'s full-scene network. Its temporal convolutions see features for all pedestrians before a target is selected. Training starts from new weights. The earlier target-readout model and its review are preserved in the [previous release](https://github.com/Nine-Three-Cattery/comp5331/releases/tag/social-stgcnn-causal-v2-review-v2). [Paper and source comparison](paper_reading_notes.md) · [Experiment plan](independent_design.md) · [Attribution](source_inventory.md)

## Run the experiment

Use a fresh checkout with the Linux x86-64 Python 3.12 CPU environment from the [group SOP](../../CONTRIBUTING.md). Formal training uses one CPU thread per process. Final timing uses two threads on the same physical host. Run these commands from the repository root. Each training entry creates its own output directory.

```bash
make benchmark PYTHON=python
python -m unittest discover -s models/social_stgcnn/own_unit_tests -v
python models/social_stgcnn/run.py pilot --device cpu --threads 1
make pilot MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/pilot/runs PYTHON=python
```

After the full ETH pilot passes, run each fold/seed once. Jobs for different pairs can run in parallel.

```bash
for fold in eth hotel univ zara1 zara2; do
  for seed in 0 1 2; do
    python models/social_stgcnn/run.py formal --fold "$fold" --seed "$seed" --device cpu --threads 1
  done
done
make score MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/formal/runs PYTHON=python
make audit MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/formal/runs PYTHON=python
python -m models.social_stgcnn.retime --threads 2
python models/social_stgcnn/check_selection.py --runs artifacts/social_stgcnn/formal/runs --data data/causal_v2
python -m models.social_stgcnn.analyze --phase extract
python -m models.social_stgcnn.review_metrics
```

To replay a forecast without overwriting the original, use the training environment and original thread count:

```bash
python models/social_stgcnn/run.py formal --phase export --fold eth --seed 0 \
    --condition clean_0 --export-to artifacts/social_stgcnn/replay --threads 1
```

## Code and evidence

- `model.py` implements degree-normalized displacement graphs, all active network layers, Gaussian loss and sampling. `windows.py` builds scenes in numeric ID order and records each scored target's slot. `export.py` evaluates equal-width scenes without padding and writes neutral bundles in manifest order.
- `run.py` fits full scenes, averages scene gradients and selects checkpoints on clean validation targets. `timing.py` rebuilds visible scenes from neutral records for the latency measurement. `retime.py` measures all selected weights on one host. The [configuration](configs/experiment.json) records the optimization, graph order and choice of checkpoint.
- `check_reference.py` compares the independent code with a separate checkout of the [pinned authors' release](https://github.com/abduallahmohamed/Social-STGCNN/tree/333d3a57b4d2705e129b21aefefa09c79b2b9ae1). Only this check loads their network. It uses deliberately matched test weights and includes a recorded ETH window. The [reference receipt](../../submission/contributions/social_stgcnn/evidence/source_parity.json) lists the software versions and per-case differences.
- `check_selection.py` uses a second Gaussian likelihood implementation on uncached validation windows. `review_metrics.py` computes ADE/FDE independently and applies the shared turning diagnostic. `analyze.py` writes per-scene and overall accuracy, paired robustness, sampling budgets, physical diagnostics, crowd strata, timing and example paths.
- The [English contribution](../../submission/contributions/social_stgcnn/README.md) contains the results, report, Beamer frames and speaker notes. Use `python -m models.social_stgcnn.analyze --phase render` with Matplotlib, `make preview MODEL=social_stgcnn` for the group preview and `python models/social_stgcnn/preview.py` for the standalone PDFs. [Review and replay instructions](REVIEW.md)
