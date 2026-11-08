# Social-STGCNN

**Owner:** Zongmin Zhang. This target-readout variant uses the Group 7 `causal_v2` windows and exports 20 forecasts per target. Each observed frame contributes one target-centered graph summary to the decoder. The authors' released decoder retains features for all pedestrians. [Paper and architecture notes](paper_reading_notes.md) · [Registered experiment](independent_design.md) · [Attribution](source_inventory.md)

## Reproduce a run

Use the Linux x86-64 CPU environment in the [group SOP](../../CONTRIBUTING.md). It pins Python 3.12, PyTorch 2.14.0 CPU, NumPy 2.5.3 and pandas 3.0.6. Run the following commands from the project root with that environment active.

```bash
make benchmark PYTHON=python
python -m unittest discover -s models/social_stgcnn/own_unit_tests -v
python models/social_stgcnn/run.py pilot --device cpu --threads 2
make pilot MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/pilot/runs PYTHON=python
```

Each formal fold/seed is an independent training job. Parallel jobs write separate run directories. The trainer creates a fresh directory and records its configuration, per-epoch likelihoods, validation checkpoint decision, neutral predictions, runtime measurements and resource use. Use a separate output root (`--runs`) to repeat training.

```bash
for fold in eth hotel univ zara1 zara2; do
  for seed in 0 1 2; do
    python models/social_stgcnn/run.py formal --fold "$fold" --seed "$seed" --device cpu --threads 2
  done
done
make score MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/formal/runs PYTHON=python
make audit MODEL=social_stgcnn RUNS=artifacts/social_stgcnn/formal/runs PYTHON=python
cp artifacts/social_stgcnn/benchmark_audit.json submission/contributions/social_stgcnn/
python -m models.social_stgcnn.retime --threads 2
python models/social_stgcnn/check_selection.py --runs artifacts/social_stgcnn/formal/runs --data data/causal_v2
python -m models.social_stgcnn.analyze --phase extract
```

To replay a selected checkpoint without overwriting the original predictions:

```bash
python models/social_stgcnn/run.py formal --phase export --fold eth --seed 0 \
  --condition clean_0 --export-to artifacts/social_stgcnn/replay --device cpu --threads 2
```

## Source map

- `model.py` implements the graph reduction, temporal network, Gaussian likelihood and path sampler.
- `windows.py` constructs a target-first graph of every pedestrian with a complete observed history.
- `export.py` records the forecast samples, metric history, missing mask, future labels and ordered IDs.
- `run.py` owns training, validation selection, export and synchronized timing.
- `configs/experiment.json` freezes the settings used for the pilot and formal matrix.
- `own_unit_tests/` covers graph geometry, causality, distribution calculations and the export interface.
- `analyze.py` pairs the formal outcomes and writes compact evidence; its render phase draws the figures from those tables alone. Rendering used Matplotlib 3.11.2.
- `timing.py` constructs runtime inputs directly from the neutral store; `retime.py` measures all selected weights on one physical host. `check_selection.py` recomputes the winning validation likelihoods.
- `review_metrics.py` calculates ADE and FDE independently from the 90 frozen bundles and applies the group's heading diagnostic. Run `python -m models.social_stgcnn.review_metrics` after preparing the bundles and shared data. This analysis was added after the formal run.
- `package_review.py`, `verify_review.py` and `handoff.py` package the original forecasts, verify the extracted files and prepare the submission provenance. [Reviewer instructions](REVIEW.md)

## Report and presentation

The [contribution](../../submission/contributions/social_stgcnn/README.md) includes the 90-row score table, English subsection, three Beamer frames, a 377-word script and the individual contribution paragraph. The group build takes its totals directly from the scored CSV. After numerical extraction, use `python -m models.social_stgcnn.analyze --phase render` on a machine with Matplotlib, then `make preview MODEL=social_stgcnn`. To compile the standalone report and three-slide deck with the same preambles, run `python models/social_stgcnn/preview.py`.

The [review archive](https://github.com/Nine-Three-Cattery/comp5331/releases/download/social-stgcnn-causal-v2-review-v1/social_stgcnn_causal_v2_review.tar) is a private repository asset. Its external size and SHA-256 are in the [provenance record](../../submission/contributions/social_stgcnn/provenance.json). The reference experiment passed the six-condition pilot, the 90-bundle common audit and the fresh-checkout checkpoint replay.
