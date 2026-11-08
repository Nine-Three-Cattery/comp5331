# DSPIGCN presentation notes

## Frame 1 — Physical coupling (about 75 seconds)

Explain that both streams receive only the eight visible positions, represented
as displacements and velocities in physical units. Attention summarizes time,
and the graph mixes observed neighbors. The polynomial is a soft training
constraint linking position and velocity, not a deterministic law imposed on
every future sample. Mention the independent implementation and documented
coordinate/context choices. Distinguish the group's shared evaluator from
third-party model reuse. Do not claim exact published-score reproduction.

## Frame 2 — Evaluate the ensemble (about 90 seconds)

Read the finalized clean meanADE and minADE from the slide. Explain that a
best-of-20 score asks whether any sample is close, while meanADE measures a
typical sample. The Energy Score also reflects ensemble spread. Describe the
largest measured paired corruption effect using the exported evidence; do
not invent a causal explanation. Each holdout has equal weight. Error bars
represent the three seed macro averages, not independent-window confidence
intervals. Selection used only clean validation data, with a fixed checkpoint
for the six test conditions.

## Frame 3 — Fixed example and limits (about 75 seconds)

This case was selected by a pre-existing runtime ID, not by an attractive
prediction. Describe the actual plotted difference between clean, noisy and
missing histories after final figure inspection. All twenty samples appear;
the sample mean is not a separately submitted prediction. Mention measured
runtime boundaries and device. Conclude that mean-level physical consistency
cannot establish safe motion for every generated path. Hand over to the
editor's cross-model comparison, which requires matched data and runtime
conditions.
