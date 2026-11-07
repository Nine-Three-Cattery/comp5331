# Independent design and known limits

The legacy model was independently written in PyTorch. One LSTMCell is shared across pedestrians. A recipient/grid/hidden tensor sums neighbor states; scene-window and Monte Carlo groups remain isolated. The loader includes observation-complete context and marks future-complete targets. Inference uses only visible history, with displacements recomputed after legacy corruption. Coordinates remain metric with an identity internal transform.

The head outputs two residual means, two log scales and one correlation. Its mean is the previous displacement plus a learned residual. Log scales are clipped to [-5,2], rho=.99*tanh(raw), and initialization has zero residual with .2m scales. Training averages twelve future Gaussian NLL terms using true-history feedback. Adam, weight decay .0001 and norm clipping10 are frozen in `configs/legacy.json`.

The old loader/evaluator intentionally retain legacy behavior, not a shared adapter. `visible_history` is the actual prediction boundary used by the CLI. The owner test changes both target and neighbor future coordinates while holding visible observations fixed. Future positions do not enter inference. Future annotation availability determines scoring eligibility and contributes to a separate training/inference context-population mismatch.

Model quality needs renewed validation. Sampled displacement innovations feed the next mean directly, creating acceleration-noise integration. A frozen zero-residual calculation gives position variance proportional to sum(h^2), and real checkpoints showed persistent impulse responses. Teacher-forced NLL is legitimate but does not guarantee free-running quality. Typical-sample macro error exceeded CV; a Hotel validation-only anchored control worsened ADE and Energy Score. No fix or causal conclusion about social pooling is claimed.

Pending: registered deterministic K=1 design, exact shared manifests, shared target-only corruption/neighbor handling, coherent future-context policy, validation-selected shared pilot and common audit. This source upload contains no new full training matrix.
