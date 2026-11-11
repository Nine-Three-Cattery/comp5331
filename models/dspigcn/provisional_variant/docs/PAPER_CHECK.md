# Paper equation review of the candidate variant

Reviewed Guo et al., KDD 2026, pp. 383–394, DOI
[10.1145/3770854.3780329](https://doi.org/10.1145/3770854.3780329), including the
rendered equations on p. 387 and Algorithm 2 on p. 388. The course paper was
available in the repository's `main` snapshot `b6bce6e`; it is not redistributed
with this contribution. Review date: 2026-11-08.

This review confirms differences, not implementation equivalence.

| Paper | Candidate implementation | Consequence |
|---|---|---|
| Eq. 7 polynomial coefficients from displacement, velocity and time | Learned fourth-order coefficients with normalized time and zero initial offset | Related physical representation with different parameterization and conditioning |
| Eq. 8 position and analytic derivative residuals | Position and derivative residuals, plus increment/dt-to-velocity consistency | Additional regularizer; numerical weighting is a project choice |
| Eq. 9 displacement bivariate Gaussian NLL | Displacement NLL plus a velocity NLL | Additional likelihood objective absent from Eq. 11 |
| Eq. 10 squared reconstruction error | Squared error of integrated metric position means | Squared form agrees; decoded quantity and reduction differ |
| Eq. 11 NLL + weighted physical and reconstruction losses | Two NLL terms; fixed reconstruction weight 1 and physics weight 0.1 | Different objective and weight schedule |
| Eq. 12 correlated bivariate covariance | Positive scales and bounded correlation | Same covariance parameterization; does not imply sampler equivalence |
| Section 3.6 / Algorithm 2 | Five-component diagonal GMM fitted on train-only 24-D future residuals, with stratified component draws | **Different sampler.** The paper mixes predicted time-step Gaussians in 2-D, then assigns nearest unused anchors to temporal means |
| Section 4.1 SGD, 200 epochs, batch 128; reconstruction weight starts at 0.01 and rises to at most 5 | AdamW, at most 100 epochs, early stopping, batch 32, fixed weights | Different training regime |

The original public artifact did not implement Algorithm 2 in its active test
path. That omission motivated the candidate GMM reconstruction, but cannot
justify calling it the paper's sampler after reading the full text. The saved
results are reported as a variant throughout. Its Gaussian NLL is a surrogate
for the actual non-Gaussian sampling distribution.

The [registered parent implementation](../../paper_reading_notes.md) already
provides the paper-based objective and anchor assignment. It remains the DSPIGCN
entry point for the group's formal comparison. This contribution changes none
of its model, loss, training, configuration or existing evidence files.
