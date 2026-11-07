# Paper reading notes

Primary source: Alahi et al., *Social LSTM: Human Trajectory Prediction in Crowded Spaces*, CVPR2016, DOI [10.1109/CVPR.2016.110](https://doi.org/10.1109/CVPR.2016.110). The social-tensor, position-estimation equations and inference section informed the implementation. No paper PDF is committed.

Each pedestrian has a recurrent state. A recipient-centered grid sums neighbors' previous hidden states, excludes self, and embeds the flattened tensor before the recurrent update. The paper embeds position coordinates, uses hidden128 and embedding64, and describes an 8x8 social pooling grid. A five-parameter correlated bivariate Gaussian predicts next position. Future NLL trains the model; inference feeds predicted positions into recurrence and the social grid.

The paper uses RMSprop .003 and synthetic data for hyperparameter selection. This legacy implementation instead uses metric displacement inputs, a 4m neighborhood, Adam .001, covariance stabilizers and a residual sampled-velocity mean. Particularly the feedback skip changes the sampling dynamics; these are not an exact reproduction of the original equations or published table.

Teacher forcing can compute a valid autoregressive joint likelihood by the chain rule. Good conditional NLL does not certify closed-loop accuracy, uncertainty quality or a deterministic comparison. Best-of20 has a larger budget than K=1. The repository registers this owner as deterministic K=1, so the old probabilistic experiment remains outside the shared comparison.
