# Design and experimental plan

**Owner:** Zongmin Zhang. **Question:** How do the accuracy, sample coverage and runtime of a graph convolutional predictor change across held-out scenes and disturbances to the target's observed trajectory?

The model uses the distance kernel, symmetric graph normalization, graph convolution and time-channel extrapolation in Sections 3–6 of Mohamed et al. (CVPR 2020). One spatial-temporal graph layer encodes the observed frames. Five extrapolation convolutions predict the parameters of twelve correlated bivariate Gaussian displacements in one pass. The likelihood is evaluated on the true increments, including the first step from the last observed position. At prediction time, twenty noise streams produce twenty paths in meters. Integrating a sampled increment sequence allows uncertainty to accumulate over the horizon. The paper describes a position likelihood; the increment parameterization is a documented design choice.

## Input and evaluation

Each ordered group window supplies eight observed metric positions and twelve future positions at 0.4 s intervals. Node zero is the target. Every other pedestrian in the same scene with all eight positions visible is included, ordered by ID. A preliminary data inspection found more than seven neighbors in 23,264 of 26,961 ETH-fold training windows. The registered design therefore retains the full observed graph. Its geometric reduction contains no trainable parameters, so the exact eight target messages can be computed once for clean training and validation. Inference recomputes the graph from the supplied, possibly corrupted target history. We record the graph population in the analysis.

Absolute positions define inter-person distances. The target's last input position is the coordinate origin for the node features and the metric output. Padding contributes no edges or signal. The original full-scene prediction is adapted to a target readout per manifest row. Maps, future context and upstream model weights are outside the input protocol.

The shared loader supplies the five time-disjoint splits. The shared corruption function changes only the target's observed history. Noise standard deviations are 0.05, 0.10 and 0.20 m. For missing suffixes of length one or two, the last two available points determine the constant-velocity fill. The clean history and these five perturbations use the same ordered targets and future labels.

## Training and selection

The prospective settings are in `configs/experiment.json`. Use a two-epoch ETH/seed-0 pilot before the formal matrix. The paper specifies minibatches of 128, SGD for 250 epochs and a learning-rate change from 0.01 to 0.002 after epoch 150. Gradient norms are clipped at 5 for numerical stability. Validation-only selection uses the frozen rule in the config. Every condition uses the same selected checkpoint within a fold and seed. Formal runs use seeds 0, 1 and 2. No held-out scores enter model or checkpoint selection.

Reconstruct positions in meters, then use the common evaluator for min/mean ADE and FDE, trajectory Energy Score, sample diversity and the motion/proximity diagnostics. Average windows within each fold and seed, then the three seeds and five holdouts with equal weight. Match sampling noise between corruption conditions for a fixed fold and seed. Fix example window IDs before inspecting their prediction errors and label any subsequently selected failure as outcome-selected.

Time the 128 registered targets in batches of 32 with three warmups and ten measured repetitions. Within each formal seed-zero fold, also time the first 32 manifest targets in each sufficiently populated neighbor stratum (0, 1–3, 4–7, 8+), using the same warmup and repeat counts. Record the hardware, software, stages and device synchronization. Separate disk access and initial neutral-store preparation from repeated preprocessing and inference. Use the measured pilot throughput when reserving the formal compute budget.
