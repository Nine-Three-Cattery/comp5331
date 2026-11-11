# DSPIGCN reading and equation audit

Source: Guo et al., *DSPIGCN: Dual-stream Physics-informed Graph Convolutional
Network for Reliable Pedestrian Trajectory Prediction*, KDD 2026, pp. 383–394,
DOI [10.1145/3770854.3780329](https://doi.org/10.1145/3770854.3780329).
Read the course-provided paper, including rendered pages 386–388. Do not
redistribute the course PDF in this contribution.

| Paper component | Independent implementation | Check |
|---|---|---|
| Eq. 1 reciprocal spatial distance | `normalized_graph` | Two-node weights; coincident nodes; padding |
| Eqs. 2–3 temporal self-attention | Separate 4-head attention in each `Stream` | Neighbor permutation and translation |
| Eqs. 4–6 GCN + residual causal TCN | Symmetric degree normalization, one graph layer and left-padded temporal convolution | Padding invariance |
| Eq. 7 fourth-order motion equation | Four convolution layers infer five vector coefficients | Polynomial/autograd agreement |
| Eq. 8 physical residual | Squared position residual plus squared velocity residual | Physical zero case and gradients |
| Eq. 9 bivariate Gaussian NLL | Positive scales, bounded correlation, analytic NLL | PyTorch distribution reference |
| Eq. 10 reconstruction | **Squared** Euclidean residual, verified in rendered PDF | Exact objective check |
| Eq. 11 combined objective | NLL + 0.1 physics + scheduled reconstruction | Tiny training loss decreases |
| Eq. 12 covariance | Correlated Gaussian draw | Seeded pool conservation |
| Algorithm 2 anchors | Pooled time-mixture, nearest unused anchor at each step | Exactly K*T anchors used once |

The paper specifies 8 observations, 12 future steps, K=20, 4 attention heads,
one GCN/TCN layer, four-layer decoders, fourth-order physics, SGD, 200 epochs,
batch 128, initial learning rate 0.001, physics weight 0.1, and reconstruction
weight starting at 0.01 and capped at 5. Parameters not fully specified by the
paper are frozen in the config and documented in `independent_design.md`.

## Reference artifact discrepancies

The paper links Zenodo record 18144387. That endpoint was inaccessible during
this implementation. A public repository whose README identifies itself as
DSPIGCN was consulted at `shangfeng2000/PI-STGCN`, commit
`d38fd905e51d401a5e812ae561185ac4a423c2e0`. Its identity with the Zenodo archive
has **not** been verified. It is a secondary implementation-detail reference,
not a substitute for the paper.

Observed at this commit: `train.py:48–52` comments out reconstruction;
`model.py:335–364` uses an MLP coefficient predictor despite the paper's
convolutional description; `test.py:72–76` creates Latin-hypercube draws, while
`test.py:97–105` uses ordinary Gaussian sampling for its reported paths.
Consequently, our objective and anchor sampler follow the paper, rather than
claim numerical equivalence to that repository. No reference implementation
is imported or copied.

## Interpretation limits

The physical term regularizes decoder means, not every stochastic sample.
It does not constrain maximum speed or guarantee collision avoidance. Anchor
assignment makes samples dependent and changes their marginals; the shared
sample Energy Score is an ensemble diagnostic, not evidence of calibrated
uncertainty. The final selected anchors can be farther from a temporal mean
than early samples, so sample index should not be interpreted as a probability
rank. No reported paper score is used as an achieved result for this code.
