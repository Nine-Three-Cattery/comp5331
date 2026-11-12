# DSPIGCN on the group benchmark

YAN Zhengyang implemented the registered dual-stream model, defined its training and sampling settings, and ran the original A800 pilot. Group 7 carried out this supplementary five-fold experiment using the same registered model and training configuration. Zongmin Zhang coordinated the runs, verification and reporting with generative AI assistance. [Implementation and original design](../../../models/dspigcn/independent_design.md) · [Source and attribution](../../../models/dspigcn/source_inventory.md)

## Accuracy and robustness

Each row averages three seeds within each held-out scene, then gives all five scenes equal weight. The test population contains 34,161 distinct target windows. Each receives 20 forecasts across a twelve-step horizon. Minimum ADE and minimum FDE choose their best sample separately. All distances are in meters.

| Target history | minADE | minFDE | meanADE | meanFDE | Energy Score | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.495 | 0.753 | 0.760 | 1.488 | 0.548 | 0.736 |
| Position noise, 0.05 m | 0.817 | 1.256 | 1.084 | 2.131 | 0.854 | 0.860 |
| Position noise, 0.10 m | 1.316 | 2.028 | 1.582 | 3.070 | 1.368 | 0.943 |
| Position noise, 0.20 m | 2.438 | 3.669 | 2.716 | 5.152 | 2.568 | 1.060 |
| Last observation missing | 0.575 | 0.835 | 0.832 | 1.582 | 0.621 | 0.731 |
| Last two observations missing | 0.665 | 0.935 | 0.912 | 1.682 | 0.702 | 0.721 |

Typical clean-path ADE was 0.760 m, with an across-seed standard deviation of 0.007 m. Mean ADE was 2.716 m with 0.20 m target-history noise and 0.912 m with two missing observations. The fixed ETH case uses the same window in all three panels. [Common result table](results.csv) · [Paired changes](evidence/paired_differences.csv) · [Fixed ETH example](assets/fixed_eth.png)

## Verification and runtime

Each of the 15 formal fits completed all 200 epochs, with the minimum clean-validation Energy Score, mean ADE and earlier epoch as the selection rule. The selected epochs ranged from 60 to 200 among 20 candidates per fit. All six conditions reused each run's best checkpoint. The shared audit covers 90 formal bundles. Direct recalculation of the four displacement errors from the stored coordinates agreed with the common table to within 1.8e-15 m. The median end-to-end latency for 128 registered targets in batches of 32 was 142.7 ms on NVIDIA GeForce RTX 4090. [Checkpoint selection](evidence/checkpoint_selection.csv) · [Common audit](benchmark_audit.json) · [Displacement and heading diagnostics](evidence/review_metrics.csv) · [Runtime measurements](evidence/runtime.csv)

On clean histories the within-window 99th percentile of sampled acceleration exceeded the training threshold in 99.6% of windows after equal-fold aggregation. The fraction was 100.0% with 0.20 m target noise. The heading and proximity tables use the same moving-step and distance conventions as the group benchmark. [Motion diagnostics](evidence/diagnostics.csv) · [Spatial diagnostics](evidence/spatial_diagnostics.csv)

The original A800 pilot and the supplementary pilot have separate records. The earlier candidate results used a different population, perturbation rule and Energy Score definition. [Original pilot](evidence/pilot_audit.json) · [Supplementary pilot](evidence/supplemental_pilot.json) · [Candidate protocol](../../../models/dspigcn/provisional_variant/docs/TEAM_INTEGRATION.md)

[Report section](report.tex) · [Presentation frames](slides.tex) · [Speaker notes](speaker_notes.md) · [Weights, predictions and release information](provenance.json) · [Replay instructions](../../../models/dspigcn/README.md)

## Weights and forecast replay

The [numerical archive](https://github.com/Nine-Three-Cattery/comp5331/releases/download/dspigcn-causal-v2-group-review-v1/dspigcn-causal-v2.tar.gz) contains the registered model, configurations, training histories, selected weights, and all 90 forecast bundles. On the extracted copy the reviewer regenerated 295,151,040 forecast coordinates. The maximum absolute difference from the stored outputs was 0 m. The observed histories, masks, future labels and ordered records agreed exactly. [Extracted replay](evidence/extracted_replay.json)
