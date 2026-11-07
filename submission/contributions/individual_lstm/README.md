# Individual LSTM and constant velocity — Group 7 results

Under the shared `causal_v2` protocol, constant velocity (CV) had lower clean error. The Individual LSTM had lower ADE and FDE at all three tested levels of Gaussian position noise. With two missing observations, their errors were close and the two metrics favored different models.

| Observations | LSTM ADE (m) | LSTM FDE (m) | CV ADE (m) | CV FDE (m) |
|---|---:|---:|---:|---:|
| Clean | 0.574 | 1.191 | 0.534 | 1.148 |
| Noise σ = 0.05 m | 0.791 | 1.526 | 0.900 | 1.748 |
| Noise σ = 0.10 m | 1.127 | 2.059 | 1.434 | 2.663 |
| Noise σ = 0.20 m | 1.871 | 3.271 | 2.607 | 4.714 |
| Last observation missing | 0.649 | 1.284 | 0.629 | 1.268 |
| Last two observations missing | 0.735 | 1.383 | 0.732 | 1.393 |

Both models produce one forecast per pedestrian. ADE is the mean distance over the twelve future positions; FDE is the distance at the last position. Lower is better. Each row averages seeds 0/1/2 within each held-out fold, then gives the five folds equal weight. The seed pairs the noise across models; CV has no fitted weights.

## Evidence and reproduction

- [主结果、协议变更与中文解读](对齐说明.md)
- [Formal 90-row LSTM result table](results.csv) and [group audit receipt](benchmark_audit.json). The receipt covers LSTM, while [CV results](evidence/cv_supplement.csv) are supplementary.
- [Per-fold results](evidence/fold_means.csv) and [cross-model comparisons with seed variation](evidence/macro_comparison.csv). Trajectron++ uses 20 forecasts: its best-of-20 error and mean error answer different questions.
- [Implementation and reproduction commands](../../../models/individual_lstm/README.md), [artifact identity](provenance.json), [report section](report.tex), and [presentation notes](speaker_notes.md).

The numerical audit passed on an Apple M5 CPU (two threads). Linux x86-64 reproduction and editor review are pending. Compare runtime only on matching hardware. Full checkpoints, raw data and prediction bundles are excluded from Git under the group policy.
