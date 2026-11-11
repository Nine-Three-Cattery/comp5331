# Trajectron++ 5-minute speaker notes

## Slide 1 · 0:00–0:30

I implemented the trajectory-only pedestrian version of Trajectron++ for
Group 7. Our task is to forecast twelve future positions from eight observed
positions on ETH/UCY. Instead of committing to a single path, this model
produces twenty plausible trajectories for each target. I'll explain the
forecasting mechanism, the causal-input audit and the results the group can
compare under the same protocol.

## Slide 2 · 0:30–1:15

The model encodes the target history and the histories of nearby pedestrians
with recurrent networks. It uses a three-meter graph and sums neighbors
within one pedestrian edge type. The history context then defines a
categorical prior over twenty-five latent classes. A recurrent decoder
samples correlated two-dimensional velocities, and a single integrator
turns them into positions in meters. During training, the variational
posterior can use the labeled future. At inference, the prior sees only the
available target and neighbor histories. The implementation uses trajectory
inputs throughout, as required by the common pedestrian benchmark.

## Slide 3 · 1:15–2:05

We hold out each scene in turn and train with three seeds. A checkpoint is
chosen on the validation split by Energy Score, with the same twenty-sample
budget. The causal audit found that centered differences over an entire track
could put the next observed-in-hindsight position into the last history
velocity. We rebuilt the target and neighbor states with backward differences
inside the window and regenerated the caches. For every input condition, the
regression changes future target and neighbor coordinates separately and
checks that features and same-seed forecasts stay identical. That check
matters because a correct-looking eight-position window alone cannot expose
leakage through its derived features.

## Slide 4 · 2:05–2:50

We average each held-out fold over its three seeds, then give the five folds
equal weight. On clean data, minADE is 0.330 meters and minFDE is
0.578 meters. Mean ADE is 0.855 meters, and Energy
Score is 0.540 meters. The best-of-twenty result asks whether
the sampled distribution contains a trajectory near the truth. Mean error
asks how far away a typical sample is. Energy Score combines accuracy with
sample spread. The fold table lets us see scene variation without allowing
the largest test scene to dominate the headline number.

## Slide 5 · 2:50–3:35

We add three levels of Gaussian position noise or hide the last one or two
target observations. A constant-velocity fill uses the last two visible
points. Each perturbation keeps the same test windows, future targets and
neighbor history. Relative to clean, twenty centimeters of noise changes
mean ADE by +2.178 meters and missing two changes it by
+0.092 meters. The chart shows how typical error, best
sample error and Energy Score respond under the same perturbation. These
results describe degradation of the target history while neighbor context is
still available.

## Slide 6 · 3:35–4:25

This is the same ETH window in three conditions, selected by the fixed
manifest rule. Black shows the observed prefix and dashed black shows the
causal fill. The orange path is the labeled future. Blue paths and endpoints
show the twenty predictions. The scale is shared across panels so the
spread is comparable. We also timed a fixed batch workload of one hundred
twenty-eight target windows. The median end-to-end result is
38.7 milliseconds, starting from in-memory neutral rows and
ending with metric predictions on the CPU. Separate times are available for
preprocessing and combined inference with sampling.

## Slide 7 · 4:25–5:00

The deliverable includes fifteen selected checkpoints, ninety test prediction
bundles and an index that links each result to its manifest and configuration.
A group member can reproduce one selected prediction or regenerate the
headline table from stored samples. I would compare models only after matching
their windows, perturbations and evaluator. The proximity diagnostic
describes how often separately sampled paths pass close together under a
training-derived threshold. It does not estimate a calibrated probability of
collision.
