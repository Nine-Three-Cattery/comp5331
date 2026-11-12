# Implementation and reproduction boundaries

Read sources: COMP5331 proposal; author-linked Zenodo metadata (record 18144467,
concept DOI 10.5281/zenodo.18144387); source files at author artifact commit
`d38fd905e51d401a5e812ae561185ac4a423c2e0`; current DSPIGCN repository commit
`c7d14f7b8dcc0130a0927a9464235a0a15b35927`. The former matches the commit named by
the Zenodo Software Heritage metadata. Direct archive download was interrupted;
the matching Git commit supplied the inspected source. The course paper became
available through the team repository and its equations were reviewed, including
the rendered page 387. [PAPER_CHECK.md](PAPER_CHECK.md) records substantive
differences in the objective and sampler. This variant is not an exact paper
reproduction, and no published numerical reproduction claim is made.

## Components and deliberate choices

| Component | This implementation | Artifact correspondence / departure |
|---|---|---|
| Displacement input | observed backward xy differences (meters/step) | `V_obs` relative differences; our first difference copies the first available difference instead of zero |
| Velocity input | observed differences / 0.4 s | artifact `pi_stgcn_any_order` actually receives absolute observed positions as its second input; here it is explicitly velocity (m/s) |
| Temporal attention | sinusoidal encoding, independent per-pedestrian Transformer encoder | history encoder concept; PyTorch implementation replaces custom multipath attention |
| Spatial graph | inverse relative-motion distance normalized Laplacian | `seq_to_graph`; zero relative-distance off-diagonal weights are zero; padding excluded |
| Forecast decoder | residual temporal convolutions and 8-to-12 temporal projection | ST-GCNN/TXPCNN layout; node-axis convolutions replaced by node-wise temporal convolutions to preserve pedestrian-permutation equivariance |
| Gaussian heads | two bivariate Gaussian heads, each with mean xy, log scales and correlation | five-channel stream outputs; scale/rho clamping gives stable covariance |
| Mean prediction | constant-velocity residual parameterization, then cumulative integration | explicit project addition, reversible to absolute metric coordinates |
| Physics coupling | fourth-order polynomial head; position and analytic velocity constraints plus displacement/dt consistency | polynomial PINN concept; fixes the unused velocity argument in artifact physics loss |
| Reconstruction | mean-trajectory squared error in metric coordinates | artifact `construct_loss` is defined but commented out in the active loss |
| Sampling | train-only, 5-component diagonal GMM over normalized 24-D residual futures; stratified mixture weights and per-component draws | project reconstruction of the described GMM anchor idea; released `test.py` has no GMM sampler |

The default trainable network uses hidden size 64, four attention heads, one
attention layer per stream, one graph block, five prediction convolutions, and
dropout 0.1. `run.json` records exact parameter count. The standalone stratified
Gaussian sampler is retained as a tested API helper; the train/evaluate workflow
uses the saved GMM anchors. Network inference reads no future-coordinate values
and performs no target-based sample choice. However, the candidate data loader
selects graph nodes using complete 20-step eligibility, which depends on future
availability. This differs from the group's history-only neighbor selection and
prevents treating the saved candidate runs as a causal_v2 benchmark.

## Objective and units

Training minimizes `NLL_displacement + NLL_velocity + lambda_rec * L_rec +
lambda_phy * L_phy`, with lambda_rec=1 and lambda_phy=0.1 by default. The
velocity likelihood target is the future position increment divided by dt.
`L_rec` compares integrated displacement means to absolute future positions.

For horizon H=12*dt and coefficients c_j, polynomial position is
`p(t)=sum(j=1..4) c_j (t/H)^j` with p(0)=0. Its exact derivative is
`v(t)=sum(j=1..4) j*c_j/H * (t/H)^(j-1)`. Physics loss is the mean of squared
position-to-polynomial, velocity-to-derivative and increment/dt-to-velocity
residuals. Position terms have units m², velocity terms m²/s²; their numerical
weighting is a documented project choice, not a dimensionless physical law.
Tests verify a known quadratic trajectory and that both streams and the physics
head receive finite nonzero gradients.

The GMM is fitted only on training future increment residuals relative to the
last observed displacement. Its mixture moments are centered and normalized
per future coordinate. A single component is selected for a whole 12-step future;
samples are transformed by each predicted covariance. This yields a distribution
with non-Gaussian modes and temporal structure. **The Gaussian training likelihood
is a surrogate, not the exact likelihood of the GMM sampling distribution.**
Mean/min errors, diversity and energy score evaluate actual exported samples.

## Defects/discrepancies found in the release

- `utils.py` retains only trajectories with `poly_fit(...) == 0` and requires
  `num_peds_considered > min_ped` (default 1). This changes population by excluding
  nonlinear motion and singleton windows. Our loader includes both.
- `poly_physic_loss_new` accepts `Velocity_pred` but never uses it. Our derivative
  and finite-difference constraints couple both streams explicitly.
- Training/validation accumulation skips the batch entering the update branch;
  partial batch normalization also uses the configured rather than actual count.
  Our minibatches optimize every valid pedestrian, including the remainder.
- Several `.view()` calls reinterpret time/channel layouts; `.squeeze()` drops
  singleton node axes. We use explicit permutations and preserve node axes.
- `test.py` computes Latin-hypercube samples into `V_absl`, then evaluates newly
  drawn independent Gaussian samples instead. It also hardcodes a Windows path.
  Our evaluator uses exactly the exported sampler outputs and configurable paths.
- Hardcoded `.cuda()` calls, script execution on import, global variables and
  pretrained pickle arguments are replaced by device-explicit modules and guarded
  commands. Checkpoints use `torch.load(..., weights_only=True)`.

These differences are intentional variant choices after full-text review. The
registered parent implementation follows the paper and shared protocol; the
candidate's checkpoints and results remain separate historical experiments.
