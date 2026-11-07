# Social-STGCNN

**Owner:** Zongmin Zhang. This pedestrian-only implementation uses the Group 7 `causal_v2` windows and exports 20 forecasts per target. The [paper notes](paper_reading_notes.md) and [experiment design](independent_design.md) explain the graph, distribution and registered training budget. See the [source inventory](source_inventory.md) for attribution.

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
python -m models.social_stgcnn.retime --threads 2
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
