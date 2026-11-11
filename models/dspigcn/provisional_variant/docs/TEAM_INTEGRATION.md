# Repository integration and comparison boundaries

Inspected the group contract at `group7/project` (`df016de`) and the existing
DSPIGCN branch (`782673d`), plus the Individual LSTM, Social-GAN, Social-LSTM
and Social-STGCNN branch layouts. Models use independent owner directories;
shared scoring reads neutral bundles without importing model implementations.

The existing `models/dspigcn` implementation is preserved. This supplemental
package uses `dspigcn_candidate`, a distinct distribution and console command.
It does not alter the group registry, evaluator, fixtures, shared dependencies,
other owners' code, registered DSPIGCN model or submission results.

## Inference with the existing batch format

Install this package in a separate environment. When running from the repository
root, expose the existing group loader modules with
`PYTHONPATH=models/dspigcn:li_junhao_trajectronpp`.

```python
from independent_tpp.data import load_split
from dspigcn.data import GraphDataset, collate
from dspigcn_candidate.group_interface import GroupPredictor

store = load_split("eth", "test", "data/causal_v2")
dataset = GraphDataset(store, seed=0, condition="clean", level=0)
batch = collate([dataset.inputs(i) for i in range(2)])
predictor = GroupPredictor("path/to/candidate/best.pt", device="cpu")
samples = predictor.predict(batch, num_samples=20, seed=0)
# CPU tensor [2,20,12,2] in absolute meters, in input manifest order.
```

The adapter consumes only `positions`, `mask`, `origin` and manifest `indices`.
It removes padding, predicts each independent target graph and restores metric
origins. A fixed manifest index keeps its sampling seed across batching/order
changes. The regression uses the actual group loader, alters target and neighbor
futures and stored derivatives, adds future-only neighbors, and checks clean,
noise and missing-history predictions. It also checks namespace coexistence,
padding, batch-order invariance and caller RNG preservation.

## Why the saved candidate results remain separate

| Item | Candidate experiments | Shared causal_v2 contract |
|---|---|---|
| Sources and IDs | Author-artifact six-column files, candidate sample IDs and hashes | Group-pinned raw release and ordered manifest identities |
| Development split | 80% frames then 19 excluded frames | Shared segments with 20 excluded steps on each side of the boundary |
| Graph context | Nodes require all 20 observed/future positions | Neighbors selected from visible history, including neighbors with no future |
| Perturbation | All eligible graph nodes; PCG64 keyed by sample ID | Target history only; RandomState keyed by training seed and manifest index |
| Stored tensor layout | Per-scene `[K,12,A,2]` in window archives | Manifest-ordered `[N,K,12,2]` neutral bundles |
| Energy Score | Unnormalized flattened sequence norm; all K² pairs including the diagonal | Sequence RMS distance; distinct sample pairs only |
| Diversity | Mean of per-step Euclidean pair distances | RMS trajectory distance between distinct sample pairs |
| Runtime | Synthetic 32-agent scenes, 10 warmups/100 repeats | Locked 128-target windows, batch 32, 3 warmups/10 repeats |

An inference bridge fixes tensor compatibility only. It does not fix training
population, graph-context selection, corruption, scoring conventions or runtime
comparability retroactively. The candidate's 15 fits / 90 rows have a local audit,
**not** a shared `BENCHMARK_AUDIT_OK` receipt. Do not copy these rows into
`submission/contributions/dspigcn/results.csv` or register them as ready.
Use the registered parent's train/score/audit commands for the formal benchmark.

## Local evidence and publication

`results/provisional_main/` contains compact candidate tables, figures and audit
receipts. Full predictions and 15 checkpoints remain in the separately retained
`zhengyang_dspigcn_experiments.tar` archive; its SHA-256 is recorded in
`results/provisional_main/publication.json`. No weights, raw data or full archive
are committed. The original local code used namespace `dspigcn`; this publication
renames it to avoid collisions and adds the bridge and documentation. The model,
loss, data, sampling and metric implementations retain their original bytes.

The supplemental suite passes 18 tests on Python 3.10 / PyTorch 2.7.0. A smoke
run of the unchanged registered model passes 10 tests in that environment; its
resume test requires the documented Python 3.12 environment (`hashlib.file_digest`
is unavailable in 3.10). That environment check is recorded separately and is not
represented as a completed shared benchmark gate.
