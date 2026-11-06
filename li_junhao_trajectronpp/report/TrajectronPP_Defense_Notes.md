# Trajectron++ defense notes

**How did you establish a fair comparison?** Use the same ordered target
windows, metric frame, horizon, perturbation masks, sample budget,
evaluator and runtime workload. Exchange the manifest hashes and corruption
fixture with each owner before assembling the six-model table. The personal
report describes the audited Trajectron++ results.

**Could the model see the future?** The old pipeline formed centered
derivatives before splitting observation and forecast windows. At the
anchor that used points after the cutoff. The corrected target and neighbor
states use backward differences within the observed window. The regression
holds history and seed fixed, changes post-anchor positions and compares
features and predictions in all six conditions. The training posterior
uses labeled futures as a supervised training signal. The inference adapter
passes only history, graph masks and the coordinate origin.

**Why perturb only the target?** It isolates target observation degradation
while preserving the neighbor context defined by the common input protocol.
All models can reproduce the target perturbation from the shared fixture.
The result applies to this scenario. A sensor-wide degradation study would
require a separately agreed experiment.

**How was the checkpoint selected?** Every tenth epoch through epoch 100 is
scored on clean validation windows using 20 samples. The selector minimizes
Energy Score, then meanADE, then epoch number. The test script reads the
frozen selection. The candidate table and hashes make the decision auditable.

**How are minADE and Energy Score different?** MinADE measures coverage of
the realized future by the closest sampled trajectory. Mean ADE measures
the typical sampled error. The Energy Score balances distance to the truth
against average pairwise spread. All three use the same 20 samples. The
Energy Score uses RMS distance over time, whereas ADE is a mean of
Euclidean distances.

**What did you independently implement?** The causal loader and
perturbations, graph and recurrent encoders, categorical CVAE, Gaussian
velocity decoder, integration, evaluator and workflow. The paper and
pinned repository document algorithmic details and hyperparameters. The
shipped source and its import boundary are documented in the provenance
files, and the package is checked in a directory without the upstream code.

**Does the motion model ensure physical or social safety?** The integrator
maps sampled velocities to positions. It imposes no obstacle or collision
constraint. Proximity compares equal indices of independently sampled
targets against a training-derived distance threshold. Treat it as a
descriptive diagnostic, alongside speed, acceleration and jerk flags.

**How much can we generalize?** The fold macro gives five scenes equal
weight. Bootstrap intervals use paired scene-level changes because nearby
windows overlap. The result characterizes these scenes and target-history
perturbations. It provides no ranking until the other owners' results pass
the shared benchmark checks.
