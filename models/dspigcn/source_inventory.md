# Sources and ownership

| Source | Version | Use |
|---|---|---|
| DSPIGCN paper | DOI 10.1145/3770854.3780329, KDD 2026, pp. 383–394 | Equations, architecture, hyperparameters |
| Public DSPIGCN-labelled reference | https://github.com/shangfeng2000/PI-STGCN/tree/d38fd905e51d401a5e812ae561185ac4a423c2e0 | Read model/train/test for discrepancies; MIT; no copied code |
| Author-linked archive | https://doi.org/10.5281/zenodo.18144387 | Access attempted; not obtained or verified |
| Group 7 shared benchmark | group7/project base; contract fingerprint in audit receipts | Neutral loader, corruption, metrics, thresholds, manifests and audit reused by agreement |
| ETH/UCY coordinate release | StanfordASL/Trajectron-plus-plus ef0165a93ee5ba8cdc14f9b999b3e00070cd8588 | Six raw files; shared SHA-256 lock |
| PyTorch / NumPy / pandas / Matplotlib | Locked environments | General libraries, no external model classes |

No copied third-party model, loss, trainer, sampler or evaluator code. Runtime
imports from `independent_tpp` are restricted to the group's neutral data and
protocol helpers, not its Trajectron++ model. Reference source, course papers,
raw data, weights and full prediction bundles remain outside tracked source.
