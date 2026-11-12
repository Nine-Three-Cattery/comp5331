# DSPIGCN on the common benchmark

The five-fold, three-seed experiment is complete. The shared CPU evaluator scored 90 forecast files from the registered model and independently checked all six headline metrics. The five held-out scenes have equal weight, with 34,161 distinct target windows and twenty forecasts per target. Clean minADE/minFDE are 0.495/0.753 m and meanADE/meanFDE are 0.760/1.488 m. Mean ADE is 2.716 m with 0.20 m position noise and 0.912 m with two missing observations. [Formal score table](results.csv) · [Common audit](benchmark_audit.json)

YAN Zhengyang implemented the model and its registered training and sampling rules and verified the original A800 pilot. Zongmin Zhang coordinated the supplementary Group 7 runs on an RTX 4090 using the same model and formal configuration. [Registered design](../../../models/dspigcn/independent_design.md) · [Original pilot](evidence/pilot_audit.json)

The full archive and the extracted-checkpoint replay are being prepared. The earlier candidate experiment remains a separate analysis under its [own protocol](../../../models/dspigcn/provisional_variant/docs/TEAM_INTEGRATION.md).
