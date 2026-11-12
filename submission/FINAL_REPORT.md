# Pedestrian Trajectory Forecasting Under Noisy and Missing Observations

- **Course** COMP5331 Knowledge Discovery in Databases, Fall 2026
- **Project type** Implementation
- **Group number** 7

## Group members and declarations

Complete the bracketed fields in the private course submission copy. For each
member with a research or FYP project, state its topic and explain the difference
in research question, data, methods or deliverables. Enter N/A where applicable.

| Student name        | Student ID | Research or FYP supervisor                                                                                      | Research or FYP topic                                                                                                                                                                    | Difference from this course project                                                                                                                                                                                                                                                                                                                                                                                                                                           | Declaration |
| ------------------- | ---------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| LU Yichong          | 21350771   | Qifeng Chen                                                                                                     | World Models                                                                                                                                                                             | My research focuses on how world models represent an environment. In this course project, I implemented and evaluated an LSTM that predicts a pedestrian’s future positions from their observed trajectory.                                                                                                                                                                                                                                                                   | Confirmed   |
| WANG Yifan          | 20945256   | Prof. Qifeng Chen                                                                                               | Controllable joint audio-video generation                                                                                                                                                | My research focuses on controllable joint audio-video generation. In this course project, I implemented and evaluated Social-LSTM for pedestrian trajectory forecasting under noisy and missing observations. The projects address different research questions and use different data and methods.                                                                                                                                                                           | Confirmed   |
| Cheung, Alan Pak To | 21055723   | Arpit Narechania                                                                                                | Analyzing Text in Videos                                                                                                                                                                 | The goal of my Final Year Project (FYP) is to extract text from movie opening and end credits and analyze the relationships and collaboration networks among film cast and crew members.<br>For this course project, I implemented and evaluated a Generative Adversarial Network (GAN), investigating the trade-offs among best-sample accuracy, typical-sample prediction error, and sample diversity, while also assessing the model's robustness and runtime performance. | Confirmed   |
| Zongmin Zhang       | 21041605   | Prof. Chaojian Li (FYT supervisor) and Prof. Lixue Cheng (research supervisor and collaborating FYT supervisor) | *Embodied Multi-Agent System for High-Stakes Environment Safety: Transferring Autonomous Driving Architectures to Edge-Deployed Labs* (FYT). My broader research area is AI for Science. | My FYT studies how a laboratory safety system can use camera images and temperature readings to decide whether to continue or pause a liquid-handling procedure. For COMP5331, I implemented and evaluated Social-STGCNN on ETH/UCY pedestrian tracks to predict the next 12 positions from eight observations.                                                                                                                                                               | Confirmed   |
| LI Junhao           | 21352303   | Song Guo                                                                                                        | Agentic Robotics                                                                                                                                                                         | My research focuses on implementing agentic harnesses for robotic tasks. In this course project, I implemented and evaluated Trajectron++ for time-series forecasting of pedestrian trajectories on ETH/UCY.                                                                                                                                                                                                                                                                  | Confirmed   |
| YAN Zhengyang       | 21320594   | Song Guo                                                                                                        | Reinforcement Learning for Robotics                                                                                                                                                      | My research focuses on reinforcement learning methods for robotic tasks. In this course project, I implemented and evaluated DSPIGCN for pedestrian trajectory forecasting from ETH/UCY position histories, studying prediction accuracy, robustness and motion consistency. The course project concerns supervised trajectory prediction, while my research concerns learning robotic behavior through reinforcement learning.                                               | Confirmed   |

Each member confirms the following statement for their own submission.
“I declare that this project was carried out solely for COMP5331. I did not
submit this work for another course or count it toward any research project or
final-year project. I have acknowledged the external papers, datasets, software
and assistance used in my contribution.”

## Abstract

We evaluated six pedestrian forecasting models on a shared ETH/UCY benchmark
with 34,161 test windows across five held-out scenes.
The benchmark pairs clean inputs with three levels of Gaussian noise and two
missing-history conditions. The deterministic Individual LSTM and Social-LSTM
produced one forecast, and Social-GAN, Social-STGCNN, Trajectron++ and DSPIGCN
produced twenty. The constant-velocity baseline achieved the lowest clean
single-forecast ADE at 0.534 m. At 0.20 m noise, Social-LSTM achieved 1.128 m ADE
and the constant-velocity error rose to 2.607 m. Among the twenty-draw models,
Social-GAN had the lowest clean minimum ADE at 0.322 m, while Trajectron++ had
the lowest minimum FDE and Energy Score. DSPIGCN had the lowest clean mean ADE
at 0.760 m among the twenty-draw models, rising to 2.716 m at 0.20 m noise
and 0.912 m with two missing observations. The shared audit passed for all
six models.

## Introduction

Eight positions can describe how a person has been moving, but several
different futures may follow from the same short observation. A person may
continue at the current speed, turn, slow down or change course in response
to another pedestrian. A forecasting experiment must distinguish errors due
to the ambiguity of future motion from sensitivity to how the past was
observed. A common test set also matters. Counting an extra group of short
or easy trajectories can shift reported error even when a model makes the
same predictions on all overlapping examples.

We built a trajectory-only benchmark to study three questions. How much
does a learned model improve on extrapolating the final observed displacement?
What is the difference between the best available path in a forecast ensemble
and its typical path? How do forecasts change when observed coordinates are
noisy or the final observations are missing? The project uses independent
model code and one shared input and scoring contract so that the answers are
traceable to the recorded forecasts and evaluation populations.

## Related work

The ETH tracking study and the UCY crowds study provide the pedestrian
coordinates used in the benchmark [1, 2]. Our Individual LSTM uses the
recurrent sequence-modeling approach of Hochreiter and Schmidhuber [3].
Social-LSTM adds a spatial pooling grid to aggregate recurrent neighbor
states [4]. Social-GAN combines recurrent prediction with adversarial and
variety losses to model multiple plausible futures [5]. Social-STGCNN
represents interactions using a spatiotemporal graph [6]. Trajectron++ uses a
graph-structured recurrent generative model and integrates motion dynamics
[7]. DSPIGCN couples displacement and velocity streams through a
physics-informed graph model [8]. These methods motivate the implementations
below. The course runs use shared data rules and declared adaptations, so
the paper scores correspond to their own published experiments.

## Data and experimental design

We used the pinned ETH/UCY world-coordinate release. The five leave-one-scene-out
folds hold out ETH, Hotel, Univ, Zara1 and Zara2, respectively. Their ordered test
manifests contain 364, 1,197, 24,334, 2,356 and 5,910 target windows. In each fold,
the other scenes supply train and validation segments. The segments are split
in time before windows are generated, with 20 resampled steps excluded on each
side of the boundary. Every target contributes eight observed positions and
twelve future positions at 0.4-second intervals. The forecast covers 4.8 seconds
after the final observation. Model-specific local coordinate transforms are
inverted before scoring in meters. Neighbor histories and derived motion
features use positions available through the observation anchor. The benchmark
does not supply maps, images or future neighbor positions to the models.
[Ordered manifests](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/benchmark/fixtures/manifests.json) · [Shared contract](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/benchmark/README.md)

The six conditions are clean observations, zero-mean Gaussian position noise
at standard deviations of 0.05, 0.10 and 0.20 m, and missing suffixes of one
or two positions. Perturbations are applied to the target history alone.
Noise draws use the common seed derived from the run seed and the target's
manifest index. Missing positions are filled by constant-velocity
extrapolation from the last two visible observations. The mask records those
filled positions. Model inputs that use velocity or acceleration derive them
from the resulting history. Future targets and neighbor observations remain
fixed across conditions. This pairs each corrupted input with its clean
counterpart at the same checkpoint. [Corruption fixture](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/benchmark/fixtures/corruption.json)

Formal common runs use seeds 0, 1 and 2. Each model selects one checkpoint per
fold and seed using clean validation and applies it to all six test conditions.
Each complete matrix therefore consists of 15 fits and 90 prediction bundles.
The common auditor verifies ordered IDs, metric coordinates, target histories,
masks, future labels, run identity and headline scores. Checkpoints,
configuration and predictions have recorded digests. Feature regressions
alter future target or neighbor records and check that model inputs and
fixed-seed predictions remain invariant where those inputs are used.
[Audit and bundle contract](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/benchmark/README.md#input-and-output-files)

## Implementations

**Individual LSTM and constant velocity.** The learned baseline encodes the
seven observed displacements with a 32-unit LSTM, then rolls out a sequence of
future displacements with a recurrent decoder. The prediction accumulates
those increments from the last input point. Training uses Adam and squared
position error with a 40-epoch ceiling and eight-epoch patience. The selected
checkpoint minimizes clean validation ADE. The supplementary constant-velocity
forecast extends the displacement between the last two input positions by the
same amount at each prediction step. [Individual LSTM implementation](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/models/individual_lstm)

**Social-LSTM.** At each observed step, the deterministic adaptation pools
neighbor hidden states in a spatial grid centered on the target. The decoder
uses the encoded context to predict target displacements. Fifteen runs used
a 12-epoch training budget and selected the checkpoint with the lowest
clean validation ADE. The earlier Gaussian K=20 exploration used different
data and training settings and remains in the owner evidence. The causal
K=1 adaptation is the version evaluated in the common tables.
[Social-LSTM implementation](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_lstm)

**Social-GAN.** A recurrent generator encodes the target and eligible neighbors
from their observed displacements. At the anchor, a learned function combines
neighbor hidden states and relative offsets, and a maximum pools up to twelve
neighbors within 6 m. A Gaussian latent draw conditions a residual recurrent
decoder. The recurrent discriminator scores real and generated continuations.
The generator minimizes best-of-twenty trajectory loss plus a weighted
adversarial term. Selection uses clean validation Energy Score, then mean
ADE and the earlier epoch. [Social-GAN implementation](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_gan)

**Social-STGCNN.** The full-scene graph represents observed displacements.
The spatial operator weights edges by the reciprocal distance between
distinct displacement vectors and uses a normalized Laplacian. The
temporal decoder retains per-pedestrian features and produces Gaussian
displacements, which are sampled and integrated into metric coordinates.
Observed neighbors can enter the graph even if they leave before the forecast
ends. The model selected the minimum clean-validation displacement negative
log likelihood from the 250-epoch run. An owner replay of trained weights
against the reference network checked 45 fixed-window cases.
[Social-STGCNN implementation](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_stgcnn)

**Trajectron++.** The pedestrian-only model encodes observed target and
neighbor histories with a recurrent graph encoder and a 3 m radius.
The 25-category prior conditions a GRU decoder that samples and integrates
correlated velocities. The causal implementation computes motion
derivatives within the observed window. Checkpoint selection uses
validation Energy Score, then mean ADE and the earlier epoch.
[Trajectron++ implementation](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/li_junhao_trajectronpp)

**DSPIGCN.** The registered implementation uses separate displacement and
velocity branches with temporal attention, graph convolution and temporal
convolution. A learned fourth-degree polynomial and its derivative couple
the streams in the training loss. The decoder uses Gaussian parameters and
anchor-based sampling. The registered configuration specifies a 200-epoch
budget with clean validation Energy Score as the primary selection criterion.
A supplemental candidate experiment used an independently packaged workflow
and a different protocol. [Registered DSPIGCN](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/models/dspigcn) · [Candidate workflow](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/models/dspigcn/provisional_variant)

### Paper deviations, implementation provenance, and reproduction

The common runs use the group-owned loader and evaluator. The attribution
records document code reuse and AI assistance. Group 7 reviewed and accepted all six model contributions.

| Model                                                                                                                                                        | Main departure from the reference                                                                                                       | Independent implementation                                                                                                                                                                                                                                                              | Reproduction status                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Individual LSTM](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/individual_lstm/independent_design.md) | Course-designed K=1 displacement encoder–decoder using standard LSTM cells and a squared position loss.                                 | Retained the earlier AI-assisted owner baseline and added the causal adapter. [Attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/individual_lstm/implementation_provenance.md)                                           | All 90 bundles passed the common audit on Apple M5 CPU. [Evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/individual_lstm/README.md)                                                                                                                                                                |
| [Social-LSTM](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_lstm/independent_design.md)         | Deterministic K=1 displacement decoder with closed-loop position loss in place of the paper's Gaussian output.                          | Implemented social-grid pooling and recurrent rollout in the project code. [Attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_lstm/implementation_provenance.md)                                                  | All 90 bundles passed the common audit on macOS. [Evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_lstm/README.md)                                                                                                                                                                           |
| [Social-GAN](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_gan/independent_design.md)           | Capped the visible neighborhood, added a constant-velocity residual, and used a target-level discriminator and shorter training budget. | Implemented the generator, pooling, discriminator, loss, and trainer in the project code. [Attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_gan/implementation_provenance.md)                                    | All 90 bundles passed the common audit. Archived-source ETH replay and pilot retraining matched the saved outputs. [Evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_gan/reproduction_check.json)                                                                                                                                                 |
| [Social-STGCNN](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_stgcnn/independent_design.md)     | Retained the full-scene operators, formed graphs from visible histories, and changed minibatch handling and checkpoint selection.       | Trained and exported with project code. The authors' code was used for numerical reference checks. [Attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/social_stgcnn/implementation_provenance.md)                        | All 90 bundles passed the common audit. A fresh checkout reproduced the ETH seed-0 clean forecast exactly. [Evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_stgcnn/evidence/reproduction_check.json)                                                                                                                                             |
| [Trajectron++](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/li_junhao_trajectronpp/independent_design.md)    | Used one pedestrian edge type, a hard 3 m mask, and motion derivatives computed within each observed window.                            | Trained and exported with the project's pedestrian-only implementation. The earlier upstream run served as a reference. [Attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/li_junhao_trajectronpp/implementation_provenance.md) | All 90 bundles passed the common audit. Relocated release verification passed. [Evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/trajectronpp/provenance.json)                                                                                                                                                                                           |
| [DSPIGCN](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/dspigcn/independent_design.md)                 | Used a Gaussian head for positions relative to the forecast origin and fixed the polynomial's time scaling and numerical conventions.   | Implemented the dual-stream model, physics objective, and anchor assignment in the project code. [Attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/dspigcn/implementation_provenance.md)                                | All 90 bundles passed the common audit. Replay reproduced all stored forecast coordinates exactly. [Audit](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/benchmark_audit.json) · [Replay](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/evidence/extracted_replay.json) |

## Evaluation

Average displacement error (ADE) is the average Euclidean error over the
twelve future positions. Final displacement error (FDE) uses the final point.
For an ensemble, minimum ADE and minimum FDE select their best paths
separately using the observed future, whereas mean ADE and mean FDE average
the errors across all paths. For K=1, minimum and mean coincide. The sample
Energy Score averages the RMS trajectory distance from each draw to the
observed future, then subtracts half the mean RMS distance over pairs of
distinct draw indices [9]. Diversity is that mean pairwise RMS distance,
defined as zero for K=1. A wide ensemble can improve coverage while
increasing typical error. Energy Score and diversity provide evidence about
the ensemble alongside the displacement measures.

We average over windows within each fold and seed, then average the three
seeds within each fold and give the five holdouts equal weight. This keeps
Univ's much larger test population from dominating the aggregate. We pair
robustness changes by fold and seed. Windows can overlap, so seed variation
and fold differences are descriptive. [Group scoring and aggregation](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/GROUP7_COMMON_DATA_EVALUATION_SPEC.md)

## Results

### Clean forecasts

All scores in the table are in meters. The constant-velocity row is
supplementary. The other rows summarize the current common result tables.
The shared benchmark audit verified the interface and scores for all six models.

| Model | K | Minimum ADE | Minimum FDE | Mean ADE | Mean FDE | Energy Score | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Individual LSTM | 1 | 0.574 | 1.191 | 0.574 | 1.191 | 0.685 | 0.000 |
| Social-LSTM | 1 | 0.607 | 1.232 | 0.607 | 1.232 | 0.721 | 0.000 |
| Social-GAN | 20 | 0.322 | 0.603 | 0.839 | 1.748 | 0.558 | 0.892 |
| Social-STGCNN | 20 | 0.583 | 1.006 | 0.971 | 1.760 | 0.796 | 0.622 |
| Trajectron++ | 20 | 0.330 | 0.578 | 0.855 | 1.823 | 0.540 | 0.991 |
| DSPIGCN | 20 | 0.495 | 0.753 | 0.760 | 1.488 | 0.548 | 0.736 |
| Constant velocity (last step) | 1 | 0.534 | 1.148 | 0.534 | 1.148 | 0.651 | 0.000 |

Constant velocity had lower clean ADE and FDE than the two learned K=1
models. The fold means qualify the aggregate difference between those learned
models. Social-LSTM was slightly lower on ETH (0.971 versus 0.998 m) and
Hotel (0.493 versus 0.506 m). Individual LSTM was lower on Univ
(0.558 versus 0.637 m), Zara1 and Zara2. Because their architectures,
context and training settings also differ, the comparison does not
identify the isolated effect of social pooling.
[Individual LSTM results and CV](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/individual_lstm) · [Social-LSTM results](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_lstm/results.csv)

DSPIGCN had the lowest clean mean ADE and mean FDE among the K=20 models.
Social-GAN had the lowest clean minimum ADE, 0.008 m below Trajectron++.
Trajectron++ had lower minimum FDE by 0.025 m and lower Energy Score by
0.018 m. The mean ADE values of both models are more than twice their
minimum ADE values. The table gives Trajectron++ minus Social-GAN on the
same clean test windows, averaged over the three paired seeds in each fold.
Negative differences favor Trajectron++.

| Held-out scene | Difference in minADE (m) | Difference in minFDE (m) | Difference in Energy Score (m) |
| --- | ---: | ---: | ---: |
| ETH | −0.032 | −0.196 | −0.046 |
| Hotel | +0.069 | +0.087 | +0.036 |
| Univ | −0.021 | −0.046 | −0.036 |
| Zara1 | +0.005 | −0.012 | −0.038 |
| Zara2 | +0.017 | +0.039 | −0.005 |

Trajectron++ had lower minimum ADE on two folds, lower minimum FDE on
three, and lower Energy Score on four. Social-GAN had lower values on
all three metrics in Hotel.
[Social-GAN results](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_gan/results.csv) · [Trajectron++ results](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/trajectronpp/results.csv) · [DSPIGCN results](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/results.csv)

For Trajectron++, we scored the first 1, 5, 10 and 20 draws from the same
clean forecast streams at fixed trained weights, giving the five holdouts
equal weight.

| Draws scored (K) | Minimum ADE (m) | Mean ADE (m) |
| ---: | ---: | ---: |
| 1 | 0.857 | 0.857 |
| 5 | 0.484 | 0.855 |
| 10 | 0.394 | 0.856 |
| 20 | 0.330 | 0.855 |

From one to twenty draws, minimum ADE fell by 62% while mean ADE stayed
near 0.855 m. Social-STGCNN showed a similar pattern on its own forecast
streams, with minimum ADE falling from 0.971 to 0.583 m and mean ADE
changing by less than 0.001 m. A larger sample set gives the evaluator
more paths to choose from after observing the future. A system acting before
the future is known needs a rule for using the ensemble.
[Trajectron++ sampling evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/trajectronpp/evidence/sample_budget.csv) · [Social-STGCNN analysis](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_stgcnn)

### Robustness to corrupted histories

The table gives paired macro changes in mean ADE from clean input, in
meters. Positive values indicate larger error. At K=1 this is a change in
the single forecast's error.

| Model | Noise 0.05 m | Noise 0.10 m | Noise 0.20 m | Missing last point | Missing last two |
| --- | ---: | ---: | ---: | ---: | ---: |
| Individual LSTM | +0.218 | +0.554 | +1.298 | +0.076 | +0.161 |
| Social-LSTM | +0.062 | +0.188 | +0.520 | +0.061 | +0.153 |
| Social-GAN | +0.068 | +0.184 | +0.466 | +0.049 | +0.120 |
| Social-STGCNN | +0.027 | +0.105 | +0.341 | +0.036 | +0.090 |
| Trajectron++ | +0.627 | +1.239 | +2.178 | +0.040 | +0.092 |
| DSPIGCN | +0.324 | +0.822 | +1.955 | +0.072 | +0.152 |
| Constant velocity (last step, supplementary) | +0.366 | +0.900 | +2.073 | +0.095 | +0.198 |

The strongest tested noise increased mean ADE more than two missing points
in every common model. For Social-STGCNN the 0.20 m noise penalty exceeded
the two-missing penalty in all five folds, with a larger difference on
Hotel (+0.432 versus +0.046 m) than on ETH (+0.264 versus +0.230 m).
For DSPIGCN, the same noise penalty ranged from +1.719 m on ETH to +2.120 m
on Hotel. Masking two points increased mean ADE by +0.298 m on ETH and
+0.049 m on Hotel. Trajectron++ had a small improvement on Hotel after
masking two points (−0.024 m) and a +0.149 m penalty on Univ. A causal fill can produce a
smoother velocity signal than differencing independently noisy positions.
These runs measure the combined behavior of the input transform and the
forecast model. They leave the individual contribution of each mechanism open.
[Social-STGCNN paired analysis](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_stgcnn) · [Trajectron++ paired evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/trajectronpp/evidence/paired_mean_ade.csv) · [DSPIGCN paired evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/evidence/paired_differences.csv)

The baseline ordering changed under noise. At 0.20 m, the deterministic
Social-LSTM reached 1.128 m ADE, followed by Individual LSTM at 1.871 m and
constant velocity at 2.607 m. With two missing points, Individual LSTM and
constant velocity were close, at 0.735 and 0.732 m. When recent
positions are available but corrupted, repeating the last displacement can
carry the error throughout the forecast. When the suffix is missing, the
benchmark already replaces that suffix with a constant-velocity continuation.
These are distinct observation failures. [Paired LSTM and CV results](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/individual_lstm/evidence/macro_comparison.csv)

### Motion, qualitative examples and runtime

The motion diagnostics use thresholds from the training partition of each
fold. A close approach describes distance between simultaneous forecast
positions. Pairing equal draw indices from independently sampled target
forecasts does not establish a joint forecast distribution. In Social-GAN,
stronger input noise raised displacement errors while the close-approach
rate fell. Interpreting that decrease requires checking where the sampled
paths moved. The Social-STGCNN fixed ETH example shows a separate coverage
failure. The pedestrian doubled back, while all twenty draws continued
forward. The fixed examples and the explicitly selected large-error cases
answer different questions about model behavior. [Social-GAN motion evidence](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_gan/evidence/motion_spatial.csv) · [Social-STGCNN fixed ETH example](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_stgcnn/assets/shared_eth_case.pdf)

The reference timing workload specifies 128 locked targets in batches of
32, with three warmups and ten measured repetitions per fold and seed.
Owners retained preprocessing, model and sampling, and end-to-end
measurements. The recorded experiments use different devices or
preprocessing boundaries. For example, the Social-GAN timing begins with
neutral positions on an Apple M5 Pro CPU, while the Social-LSTM MPS
measurement starts from cached neutral histories. We retain owner timings
with their environment and boundaries and leave cross-model speed rankings
to a matched measurement. [Shared timing workload](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/benchmark/README.md#runtime-and-diagnostics)

### DSPIGCN motion diagnostics and replay

The registered DSPIGCN completed all 15 formal fits at 200 epochs each.
Checkpoint selection used clean validation Energy Score, then mean ADE
and the earlier epoch. The reviewer regenerated all 90 forecast bundles
from the archived weights with a maximum coordinate difference of 0 m.
On clean histories, the 99th percentile of sampled acceleration exceeded
the training threshold in 99.6% of windows after equal-fold aggregation.
This diagnostic qualifies the model's clean mean displacement scores.
[Checkpoint selection](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/evidence/checkpoint_selection.csv) · [Forecast replay](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/evidence/extracted_replay.json) · [Motion diagnostics](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/evidence/diagnostics.csv)

## Limitations and conclusion

The shared evidence spans five scenes and three training seeds. Overlapping
windows, unequal crowd composition and a single dataset family limit the
inferences that can be drawn about other populations. The experiments compare
complete implementations with different architectures, objectives and training
budgets. Claims about individual components require controlled ablations.

The constant-velocity baseline repeats the last observed displacement. It had
the lowest clean ADE among the deterministic predictors. Both recurrent
baselines had lower ADE than this baseline at all three tested noise levels.
Increasing an ensemble's sample budget improves best-path coverage under
oracle selection. Among the K=20 models, Social-GAN led in clean minimum
ADE, Trajectron++ led in minimum FDE and Energy Score, and DSPIGCN led in
mean ADE and mean FDE. Across all six models, 0.20 m noise caused more
mean-ADE degradation than removing and causally filling two observations.

## Contribution

Each member should verify their contribution statement and complete the
individual online form with their private contribution percentage and
written account.

### LU Yichong

LU Yichong implemented the deterministic Individual LSTM and the
constant-velocity sanity check. The work covered converting observed
positions to displacements, restoring metric forecasts, training and
validation selection, and exporting predictions in the shared manifest
order. The contribution includes causal and coordinate checks and the
90-row benchmark table. LU prepared the supplementary constant-velocity
table using the same target histories and noise seeds, analyzed when the
learned baseline improved on extrapolation, and documented changes from
the legacy population. LU also prepared fold and paired result summaries,
reproduction commands, and report and presentation materials. The numerical
audit was recorded on Apple M5 CPU. [Owner evidence](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/individual_lstm)

### WANG Yifan

WANG Yifan implemented the social pooling and recurrent prediction
pipeline for Social-LSTM. The contribution preserves the exploratory
Gaussian variant and records the later deterministic K=1 model under
the group benchmark. WANG ran the ETH pilot and the formal fold and seed
matrix, selected checkpoints with clean validation ADE, and checked
causal features, graph and padding behavior, and output alignment.
The analyses include comparison with constant velocity, noise and
missing-history sensitivity, crowded and turning strata, a fixed Hotel
example, and hardware-specific runtime. WANG supplied the numerical audit,
provenance, an English report subsection, three Beamer frames, speaker
notes and an editable PowerPoint. WANG also documented differences between
the local and reference environments. [Owner evidence](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_lstm)

### Cheung, Alan Pak To

Cheung, Alan Pak To implemented the Social-GAN framework, including the generator, discriminator, observed-neighbor pooling mechanism, residual displacement decoder, and reproducible seeded sampling. He completed the ETH pilot and the full five-fold, three-seed evaluation under the shared benchmark, using clean validation Energy Score for checkpoint selection. He verified that neither target nor neighbor future trajectories leaked into the model inputs and developed the prediction exporter to produce outputs aligned with the shared evaluation protocol. He independently checked the reported metrics and conducted both checkpoint-replay and short-retraining reproducibility tests. For the experimental analysis, he investigated the trade-offs between best-of-K forecast accuracy, typical-sample error, and sample diversity, and evaluated model robustness under noisy and missing trajectory observations. He also prepared qualitative trajectory examples, motion diagnostics, and runtime measurements with documented measurement boundaries. Finally, he contributed the model audit, experimental provenance, Social-GAN report section, and presentation materials. [Owner evidence](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_gan)

### Zongmin Zhang

Zongmin Zhang implemented the full-scene Social-STGCNN and integrated it
with the benchmark. After examining the authors' network, Zhang replaced
an earlier target-only readout with a decoder that retains a feature
vector for every pedestrian. The implementation covers displacement
graphs, spatial and temporal convolutions, Gaussian likelihood, sampling,
training and export. Zhang checked target ordering and future invariance,
ran the ETH pilot and the five-fold, three-seed matrix, independently
rechecked the selected validation likelihoods and replayed forecasts with
the authors' model. Zhang prepared accuracy, robustness, crowd, motion and
timing analyses, published review archives, and delivered figures, the
report section and presentation. The owner statement records AI assistance.
[Owner evidence](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/social_stgcnn)

### LI Junhao

LI Junhao built the shared benchmark used for the six common result
tables, including pinned ETH/UCY data preparation, ordered holdout
manifests, target-history corruption fixtures, scoring, and audits. LI
implemented the independent pedestrian-only Trajectron++ pipeline with a
recurrent graph encoder, probabilistic velocity decoder, and metric
prediction export. A regression test verified that changing future target
and neighbor positions left fixed-seed predictions identical across all six
conditions. LI selected checkpoints using clean validation Energy Score and
delivered 90 audited prediction bundles at K=20 across five holdouts and
three seeds, with release hashes and commands to replay and rescore
predictions. LI analyzed sample budgets and paired robustness to noise and
missing observations, and timed preprocessing and inference separately.
LI also built the shared LaTeX report and Beamer slide masters and their
assembly workflow, and delivered the Trajectron++ report section, figures,
slides, and speaker notes. [Shared benchmark](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/benchmark/README.md) · [Owner evidence](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/trajectronpp)

### YAN Zhengyang

YAN Zhengyang implemented DSPIGCN's displacement and velocity streams,
temporal attention, graph convolution, fourth-order polynomial physics
coupling, training objective and anchor sampler. The registered workflow
includes history-only target and neighbor features, validation-based
checkpoint selection and metric prediction export. YAN documented the
implementation's departures from the paper and checked causal inputs,
coordinate transformations, physical derivatives and likelihood calculations.
The two-epoch ETH seed-0 pilot on the A800 passed the shared audit across
all six conditions. [Registered implementation and pilot evidence](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn)

YAN also completed a separate candidate workflow with five held-out scenes,
three training seeds and twenty forecast samples, producing 15 fitted models
and 90 evaluation rows. The local audit checked source trajectories,
perturbations, checkpoint selection and accuracy and motion metrics. The
analysis covers minimum and mean ADE/FDE, ensemble diversity, Energy Score,
noise and missing-history sensitivity, motion consistency, runtime and
qualitative failures. YAN retained the checkpoints and predictions in a
checksummed archive and verified replay of the first clean window from
each of the fifteen checkpoints. [Candidate audit and replay evidence](https://github.com/Nine-Three-Cattery/comp5331/tree/e956215a75acc2ccebf622c4a89566adc14f8766/models/dspigcn/provisional_variant/results/provisional_main)

To support integration, YAN packaged the candidate in an isolated namespace
and added a group-batch prediction adapter. Eighteen tests passed, including
checks with the actual shared loader for future invariance, padding,
batch ordering and preservation of the caller's random state. YAN prepared
report and presentation materials and recorded external references and
Codex assistance. The candidate's different populations, graph eligibility,
corruptions and scoring conventions remain explicit; its results are kept
outside the common comparison. Group 7 ran the registered formal matrix,
with Zongmin Zhang coordinating training, verification and reporting with
generative AI assistance. [Candidate integration and attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/models/dspigcn/provisional_variant/docs/TEAM_INTEGRATION.md) · [Formal experiment and attribution](https://github.com/Nine-Three-Cattery/comp5331/blob/e956215a75acc2ccebf622c4a89566adc14f8766/submission/contributions/dspigcn/README.md)

## References

1. Pellegrini, S., Ess, A., Schindler, K., and Van Gool, L. 2009. “You'll Never Walk Alone: Modeling Social Behavior for Multi-Target Tracking.” ICCV, 261–268.
2. Lerner, A., Chrysanthou, Y., and Lischinski, D. 2007. “Crowds by Example.” *Computer Graphics Forum* 26(3), 655–664.
3. Hochreiter, S., and Schmidhuber, J. 1997. “Long Short-Term Memory.” *Neural Computation* 9(8), 1735–1780. [Paper](https://doi.org/10.1162/neco.1997.9.8.1735)
4. Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., and Savarese, S. 2016. “Social LSTM: Human Trajectory Prediction in Crowded Spaces.” CVPR. [Paper](https://doi.org/10.1109/CVPR.2016.110)
5. Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., and Alahi, A. 2018. “Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks.” CVPR. [Paper](https://arxiv.org/abs/1803.10892)
6. Mohamed, A., Qian, K., Elhoseiny, M., and Claudel, C. 2020. “Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction.” CVPR, 14424–14432. [Paper](https://arxiv.org/abs/2002.11927)
7. Salzmann, T., Ivanovic, B., Chakravarty, P., and Pavone, M. 2020. “Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data.” ECCV. [Paper](https://arxiv.org/abs/2001.03093)
8. Guo, R., Chen, B., Zhu, Z., Gao, C., Zhao, Y., and Yin, Q. 2026. “DSPIGCN: Dual-stream Physics-informed Graph Convolutional Network for Reliable Pedestrian Trajectory Prediction.” KDD, 383–394. [Paper](https://doi.org/10.1145/3770854.3780329)
9. Gneiting, T., and Raftery, A. E. 2007. “Strictly Proper Scoring Rules, Prediction, and Estimation.” *Journal of the American Statistical Association* 102(477), 359–378. [Paper](https://doi.org/10.1198/016214506000001437)
