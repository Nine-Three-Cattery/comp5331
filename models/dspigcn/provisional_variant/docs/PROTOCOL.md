# Provisional DSPIGCN protocol v1

This is a concrete, reproducible candidate protocol, **not the agreed group
protocol**. Repository inspection confirmed differences from `causal_v2` in
sources, split boundaries, context eligibility, perturbations and runtime.
See [TEAM_INTEGRATION.md](TEAM_INTEGRATION.md). Existing results retain their
original candidate identities and are excluded from the common comparison.

## Population and split

Scene mapping: ETH=`biwi_eth`; Hotel=`biwi_hotel`; Univ=`students001` and
`students003`; Zara1=`crowds_zara01`; Zara2=`crowds_zara02`. The artifact's extra
`crowds_zara03` and `uni_examples` files are excluded from this five-scene
candidate protocol. File SHA-256 values are recorded independently for each fold.

Frame IDs are identifiers, not seconds. One normal consecutive frame increment
represents 0.4 s. Infer this increment as the modal difference of sorted frame
IDs per file. Reject windows spanning larger gaps. Reject duplicate frame/person
rows and malformed/nonfinite values. Never silently interpolate absent interior
positions or filter by linear/nonlinear motion. Eligible people occur at every
one of 20 frames. Include a window even if just one person is eligible.

For a held-out scene, all its complete windows form test data. In each other
scene file, take `floor(0.8 * number_of_frames)` frames as training, discard the
next 19 frames, and use the rest for validation. Generate windows *after*
segmenting. Same-person segments may occur on either side, but frame ranges and
20-frame windows cannot overlap. No learned input normalization is applied.
Train the GMM and compute diagnostic thresholds using training trajectories only.

Each sample is `filename:start_frame:person_id`. Windows are ordered by the
fixed scene mapping, filename, ascending start frame, then ascending person ID.
Manifests include all IDs, split segments, settings, counts and content hashes.
Main stride is 1; pilot stride is 5 and has a different test-set identifier.

## Metrics

Compute Euclidean position errors in meters per time step. ADE is their time
mean; FDE is the final-step error. For K=20, take independent ADE/FDE minima
across **whole trajectory samples**, not minima separately at each timestep.
Mean ADE/FDE average all K samples. Average pedestrians/windows with equal
weight per eligible trajectory, not equal weight per variable-size scene window.
Keep individual seed rows before calculating per-scene sample standard deviation
across seeds. Do not combine differing test-set IDs or sample budgets.

Known check: target `(0,0),(0,0)`; prediction 1 `(0,0),(4,0)`; prediction 2
`(3,0),(0,0)`. Expected minADE=1.5, minFDE=0, meanADE=1.75, meanFDE=2.

Energy score: flatten all 12 future xy coordinates into a 24-vector. For empirical
samples X_i, calculate `mean_i ||X_i-y|| - (1/(2*K*K))*sum_ij ||X_i-X_j||`.
Include diagonal pairs (zero). No division by the sequence length. The known
example above has energy score 2.25. Diversity is the mean per-timestep distance
between different sample pairs. K=1 diversity is zero. Neither score alone
establishes calibration.

## Corruptions and diagnostics

Conditions: clean (sigma=0), Gaussian position noise sigma=0.05/0.10/0.20 m,
or missing final 1/2 observed positions. Conditions are separate; targets are
unchanged. For each sample ID use NumPy PCG64 seeded by the first eight SHA-256
bytes (little endian) of `2026:<sample_id>`. Draw an `[8,2]` normal array per
pedestrian. Sigma conditions reuse identical standard normal draws. Missing
suffixes use the last two remaining observations to extrapolate constant motion;
hidden values never enter the fill. Sampling uses the same random draws across
conditions for a scene/seed, independent of ground truth.

Speed, acceleration and jerk use finite differences divided by dt, dt², dt³,
including the observation/prediction boundary. Exceedance cutoffs are training
99th percentiles. Close-distance cutoff is the training 1st percentile of
simultaneous pair distances (0.2 m only if training has no pairs). Count unordered
pairs from the same scene and time only; denominator is K*T*number_of_pairs.
Export denominator, since no-pair scenes must not imply evidence of zero collision
risk. Sampling indices do not constitute a learned joint interaction distribution.

## Runtime and figures

For fixed-count runtime: 32 pedestrians, 20 samples, 10 warmups, 100 repeats;
synthetic histories from torch seed 0. CPU noise generation plus host-to-device
copy, model forward (including graph construction), and sampling/integration are
timed separately. GPU timers synchronize at boundaries. Store hardware, thread
count and dependency versions; compare only matching settings.

Evaluation additionally records total forward/sampling time at actual scene sizes,
after five warmups, excluding IO, metrics and plotting. These totals must not be
compared to a fixed-count benchmark. The first three eligible windows are fixed
examples; the three highest mean-ADE windows are selected as model-specific
failures. At most eight people are displayed per figure for readability; every
eligible person is retained in metrics and prediction archives.
