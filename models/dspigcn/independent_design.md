# Independent implementation and registered choices

Owner: YAN Zhengyang. Model: DSPIGCN. Protocol: causal_v2. Output budget: K=20.

## Inputs and units

Reuse the group's neutral loader, ordered manifests and target corruption.
Build a target-centered graph separately for each manifest window. Node 0 is
the target; neighbors have all eight visible history points and lie within
3 m at least once during that history. No neighbor future availability is
required; no target is removed. Retained nodes connect by reciprocal metric
distance at each step. Padding has zero degree and cannot affect valid nodes.
The 3 m history filter is a disclosed computational departure from a full
scene graph; all retained nodes have all-to-all edges.

Use observed displacements in meters and backward differences / 0.4 in m/s.
The first unavailable difference is zero. Discard all supplied derivative
columns. Perturb the target first, recompute derivatives and graph weights,
and leave neighboring histories clean as required by the group protocol.
Translation uses the possibly perturbed final target observation; add this
origin back exactly once at export. No fitted normalization, rotation, maps,
future-derived features or test statistics are used.

## Network and departures

Two independent 64-channel streams each use an embedding, four-head temporal
attention with a residual LayerNorm, symmetric GCN, and a causal kernel-three
TCN with residual/dropout 0.1. Attention can use all eight observed points;
these all precede the forecast. The two four-affine-layer MLP heads produce
12 relative **position** Gaussian parameters and 12 velocities respectively.
We interpret paper P(t) as displacement from the forecast origin, rather
than the upstream repository's per-step increment distributions. This makes
P'(t) and velocity dimensionally consistent in seconds. NLL and reconstruction
are computed in the same relative-position space.

Heads use residuals about last-observed constant velocity with zero output
initialization. The polynomial coefficient network receives position means,
velocity predictions and physical time. Three width-64 kernel-three hidden
convolutions and one length-12 output convolution yield a global degree-four
polynomial per target. Coefficients are parameterized on t/4.8 for conditioning,
then converted to seconds before analytic evaluation and differentiation.
This initialization/parameterization is our explicit numerical choice.

Use bounded log standard deviations [-5,3], correlation 0.999*tanh, and
mean reductions over targets/time. Eq. 10 uses squared distance. Physics
residuals use sums over coordinates. These numerical conventions are fixed
before formal training. They must not be mistaken for a bitwise reproduction.

Algorithm 2's mixture weights are unspecified. Use equal temporal weights,
stratified by drawing K anchors from each of 12 component Gaussians. Pool all
240 anchors; repeatedly choose the nearest unused anchor to each mean in
chronological order, removing it globally. Generate each window's random
numbers with its recorded CPU torch seed, independent of batch composition;
the same seeds pair robustness conditions. Never choose samples with labels.

## Training and selection

Formal config: 200 epochs, full training manifest, batch 128, SGD 0.001,
momentum 0.9, learning-rate halving each 50 epochs, gradient norm limit 10.
Reconstruction weight min(5,0.01*1.01^(epoch-1)); physics 0.1. The growth factor
is clarified by the consulted reference; the paper leaves it unspecified.
Validate every 10 epochs and at the last epoch. Select the minimum clean
validation Energy Score, then mean ADE, then earlier epoch, at K=20. All six
test conditions use the same selected checkpoint. Pilot uses two epochs and
validates each epoch, in a separate output root. No test-based tuning.

Last-epoch checkpoints save optimizer, scheduler and device RNG states.
Reproducible epoch shuffles derive from seed and epoch. Changed configs are
rejected for an existing run. Exact reproducibility is bounded by the saved
hardware/software versions; CUDA and CPU training are not assumed identical.

## Resource plan

First verify the shared loader/evaluator and synthetic tests; then run full
ETH seed-0 pilot under all six conditions. Measure pilot training, validation,
peak memory and inference before running 15 independent formal fits on the
available A800. Export 90 bundles, score/audit with the shared CPU evaluator,
measure the locked 128-target runtime workload, generate figures and report,
and package the immutable runs. Optional ablations follow these deliverables.
