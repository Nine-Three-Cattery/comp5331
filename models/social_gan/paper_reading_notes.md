# Social GAN paper reading notes

Gupta, Johnson, Li, Savarese and Alahi, *Social GAN: Socially Acceptable
Trajectories With Generative Adversarial Networks*, CVPR 2018, pp. 2255–2264.
[CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2018/html/Gupta_Social_GAN_Socially_CVPR_2018_paper.html) ·
[Full text](https://arxiv.org/html/1803.10892v1)

## Generator and pooling

Sections 3.3–3.4 and Figures 2–3 separate individual motion encoding from
interaction. A shared LSTM encodes each pedestrian's history. For pedestrian
\(i\), the pooling MLP processes each neighbor's hidden state together with its
position relative to \(i\). An elementwise maximum summarizes this set into a
fixed-size vector, so the ordering and number of neighbors can vary. The
paper pools relative positions directly in place of the occupancy grid used
by Social-LSTM. Section 3.3 describes passing pooled context to the decoder once, even
though Eq. 4 also gives a recurrence with a pooling term. Section 4.1
attributes part of the measured speed advantage to avoiding pooling at every
step. We use pooling at the observation anchor.

The decoder combines the observed motion context with a Gaussian latent draw
and generates the future autoregressively. It predicts positions through a
deterministic recurrence for each latent draw. In our implementation the
recurrence predicts displacements around a constant-velocity reference, and
we accumulate those displacements from the observed anchor to recover metric
positions. Our target-centered pool uses neighbors with a complete observed
history within 6 m of the anchor. This local, observation-only restriction
differs from the paper's scene-wide pool.

## Discriminator and variety loss

The discriminator is a separate recurrent encoder. It sees the observed
prefix followed by either the real or generated future and learns to classify
the complete sequence. The observed prefix matches between real and fake
examples. We apply the discriminator to target trajectories. Coordination
between forecasts is a separate question assessed through spatial diagnostics.

Section 3.5 draws multiple candidate futures and applies a reconstruction
penalty to the candidate closest to the realized future in trajectory
\(L_2\) distance. This gives a training signal when one draw covers the
target and leaves other draws available to represent alternatives. Diversity
and calibration require separate evidence. Training
variety count and test sample count serve different roles. Both are set to
20 in this implementation.

## What the paper's comparisons establish

Section 4 tests five held-out ETH/UCY sets at 0.4 s intervals, with eight
observations and either eight or twelve predictions. In `SGAN-kVP-N`, `k` is
the training variety count, `P` indicates pooling, and `N` is the test sample
count. At twelve predicted steps, Table 1 gives average ADE/FDE of 0.58/1.18 m
for `SGAN-20V-20` without pooling and 0.61/1.21 m for `SGAN-20VP-20` with
pooling. The paper examines possible benefits of pooling in separate
qualitative examples, where its collision observations apply to the scenes
shown.

We use Group 7's locked splits, target-only perturbations and common evaluator.
The paper's values describe its own protocol. In our evaluation, minimum ADE
or FDE rewards finding a good candidate among twenty draws, while mean error
describes the error averaged over the draws. Diversity describes separation
between candidates. Energy Score considers both distance to the observed
future and spread within the forecast. The four views can disagree, and the
experiment evaluates them separately.
