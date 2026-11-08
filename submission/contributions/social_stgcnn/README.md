# Social-STGCNN with a target-only readout · Zongmin Zhang

Every pedestrian visible for all eight observations enters the graph. The decoder receives one weighted sum of centered positions per frame and exports twenty future paths per target. The authors' released decoder retains a feature vector for every pedestrian and applies two-dimensional convolutions across the full embedding. The results below describe the target-readout variant on the group's five ETH/UCY holdouts. [Architecture and worked example](../../../models/social_stgcnn/paper_reading_notes.md#comparison-with-the-released-network-after-training)

- [English report section](report.tex), [Beamer frames](slides.tex), [speaker notes](speaker_notes.md) and [individual contribution statement](contribution_statement.tex).
- [Formal results](results.csv), [common audit receipt](benchmark_audit.json), [reproduction artifact and timing](provenance.json) and [analysis tables](evidence/).
- [Independent implementation and run commands](../../../models/social_stgcnn/README.md), [paper and implementation differences](../../../models/social_stgcnn/paper_reading_notes.md) and [private archive verification](../../../models/social_stgcnn/REVIEW.md).

## Formal results

Distances are in meters. Each row averages three seeds within a holdout, then weights the five holdouts equally. The experiment contains 34,161 distinct target windows. Each window receives twenty forecast draws.

| Target history | minADE | minFDE | meanADE | meanFDE | Energy Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.615 | 1.081 | 0.940 | 1.739 | 0.830 |
| Position noise, 0.05 m | 0.663 | 1.155 | 1.001 | 1.833 | 0.891 |
| Position noise, 0.10 m | 0.783 | 1.342 | 1.143 | 2.051 | 1.034 |
| Position noise, 0.20 m | 1.120 | 1.865 | 1.509 | 2.619 | 1.413 |
| Last point missing | 0.652 | 1.131 | 0.981 | 1.793 | 0.870 |
| Last two points missing | 0.702 | 1.189 | 1.035 | 1.853 | 0.922 |

At twenty draws, best-sample ADE was 0.615 m and typical-sample ADE (mean over draws) was 0.940 m. The largest clean seed-zero minADE was 4.67 m on a pedestrian who reversed direction. The nearest sampled endpoint missed by 9.52 m.

## Verification and build

The ETH seed-zero pilot used two epochs, 26,961 training windows and the complete 364-target holdout under six conditions. Training, export and the initial timing pass took 34.43 seconds, with peak process RSS of 534,355,968 bytes. A subsequent timing check included visible-neighbor discovery from in-memory neutral positions. The final latency measurements use that boundary on one physical CPU node.

All fifteen formal runs used 250 epochs. Among 51 clean-validation candidates per run, the selected checkpoints ranged from epochs 80 to 250. Re-evaluating each saved checkpoint with uncached graphs and a separate double-precision Gaussian likelihood reproduced its recorded validation loss within 7.5 × 10⁻⁸ nats per target step. See the [selection replay](evidence/selection_check.json).

During review, an owner-written calculation of min/mean ADE and FDE agreed with the shared score table on all 90 frozen bundles to within 8.9 × 10⁻¹⁶ m. On clean inputs the mean absolute turn between successive sampled steps was 0.785 rad (45.0°). In 98.5% of windows at least one sampled turn reached the fold-specific 95th percentile for training turns. This percentage uses the maximum over up to 220 turns per window. Folds receive equal weight. [Per-run values](evidence/review_metrics.csv) · [Definitions and reproduction](../../../models/social_stgcnn/REVIEW.md#review-additions-for-the-same-frozen-forecasts)

A separate plot follows the Trajectron++ handoff's fixed ETH example, window `eth:biwi_eth:216:875`, at seed zero. Its three panels show clean input, 0.20 m noise and two missing observations for the same pedestrian. [Matched ETH plot](assets/shared_eth_case.pdf)

Run `make preview MODEL=social_stgcnn` from the repository root to compile the group-formatted contributor preview. The editor controls integration in `submission/models.json`.
