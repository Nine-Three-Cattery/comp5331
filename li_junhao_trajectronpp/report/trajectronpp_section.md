# Trajectron++ pedestrian implementation — LI Junhao

## Method and protocol

Our independent pedestrian implementation encodes target and 3 m neighbor
histories, samples a 25-category latent prior, and integrates Gaussian GRU
velocity samples into metric positions. The CVAE posterior observes future
labels only during training. The model uses one pedestrian edge type and
trajectory inputs. Complete neighbor histories, a hard radius mask and the
velocity NLL objective are documented deviations from the broader paper
configuration. [Trajectron++ paper](https://arxiv.org/abs/2001.03093)

The benchmark uses eight observations, twelve future steps and 0.4 s
spacing. We trained 100 epochs on five scene holdouts with three seeds and
selected checkpoints by clean validation Energy Score at K=20. A causal
input audit replaced whole-track centered derivatives with within-window
backward differences for targets and neighbors. Changing either agent's
positions after the anchor leaves predictions identical with a fixed seed.
The target-only noise and missing-suffix conditions use the same ordered
windows and unchanged test futures. Neighbor observations remain available.

## Results

The audit covers 15 fold–seed runs, 90 prediction bundles and 34,161
unique test windows. The clean scene macro is 0.330 m minADE,
0.578 m minFDE, 0.855 m meanADE and
0.540 m Energy Score. Each row below averages the three seeds.
SD is the across-seed minADE standard deviation.

| Holdout | Windows | minADE (m) | SD (m) | minFDE (m) | meanADE (m) | ES (m) |
|---|---:|---:|---:|---:|---:|---:|
| ETH | 364 | 0.600 | 0.021 | 1.037 | 1.366 | 0.935 |
| Hotel | 1,197 | 0.276 | 0.023 | 0.467 | 0.857 | 0.467 |
| Univ | 24,334 | 0.312 | 0.006 | 0.586 | 0.722 | 0.523 |
| Zara1 | 2,356 | 0.238 | 0.000 | 0.397 | 0.709 | 0.399 |
| Zara2 | 5,910 | 0.222 | 0.016 | 0.401 | 0.621 | 0.378 |

Each scene contributes equally to the following condition-level macro.
All distances are in meters.

| Condition | minADE | minFDE | meanADE | meanFDE | Energy Score | Diversity |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.330 | 0.578 | 0.855 | 1.823 | 0.540 | 0.991 |
| Noise 0.05 m | 0.485 | 0.744 | 1.482 | 2.978 | 0.768 | 1.966 |
| Noise 0.10 m | 0.683 | 0.984 | 2.094 | 4.111 | 1.029 | 2.851 |
| Noise 0.20 m | 1.054 | 1.426 | 3.033 | 5.804 | 1.510 | 4.051 |
| Missing one | 0.373 | 0.628 | 0.895 | 1.848 | 0.592 | 0.950 |
| Missing two | 0.437 | 0.697 | 0.947 | 1.881 | 0.662 | 0.894 |

0.20 m noise changes meanADE by +2.178 m relative to clean
(95% paired scene bootstrap interval [1.757, 2.586]).
Missing two changes it by +0.092 m
([0.032, 0.138]). These intervals describe
variation across five folds. The clean motion-flag fractions are
9.4% for speed, 13.8% for acceleration and
13.4% for jerk. The clean close-approach fraction is
87.2% under train-only distance thresholds. Equal-index matching
of separately sampled targets is a descriptive proximity diagnostic.

The seed-0 failure list contains a clean window with 4.767 m
meanADE on ETH. Fixed and rule-selected figures show observed
histories, samples, endpoints and ground truth under clean, strong noise
and missing-two conditions. The common GPU benchmark (128 targets, batch
32, K=20) reports 30.8 ms preprocessing, 7.4 ms
combined inference and sampling, and 38.7 ms end-to-end,
each a median of run medians. Preprocessing starts from in-memory neutral
rows and end-to-end ends with metric predictions on CPU.

## Group handoff

Use `group7_trajectronpp.csv`, the prediction index and corruption fixture
to match manifest hashes, sample budgets, masks, metric definitions and
runtime boundaries before comparing with the other five models. These
results characterize the project-specific pedestrian implementation and
five test scenes.
