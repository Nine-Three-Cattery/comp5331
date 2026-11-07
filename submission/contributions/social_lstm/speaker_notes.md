# Social-LSTM speaker notes

Target: approximately 3 minutes at 120–130 words per minute. Three content slides, no cover. Presenter: WANG Yifan. The spoken script below has approximately 365 words. Figures and results are exploratory, not causal_v2 audited results.

## Slide 1 — 0:00–0:55

My part is Social-LSTM, which adds information about nearby pedestrians to a recurrent trajectory predictor. The key idea comes from Alahi and colleagues: each pedestrian has a hidden state, and a spatial grid pools the hidden states of neighbors. This lets the next prediction depend on both individual motion and surrounding people.

Our model observes eight positions and predicts the next twelve. I implemented the pooling and recurrent prediction in PyTorch and ran five held-out scenes with three seeds, giving fifteen training runs. This implementation uses a Gaussian decoder with an additional residual-velocity connection. That is an adaptation, so I describe the results as an implementation study rather than an exact reproduction.

## Slide 2 — 0:55–1:55

The main result is that best-sample accuracy and typical prediction error tell different stories. On our exploratory split, constant velocity has an ADE of about zero point five three meters. Social-LSTM's average error over twenty sampled paths is about zero point nine two meters. However, selecting the best of those twenty paths reduces the error to about zero point three five meters.

This means the model can generate a path close to the observed future, but many other samples drift away. The best-of-twenty number therefore does not show that our predictor is better than constant velocity for a single forecast. Robustness is also a limitation: with zero point two meters of observation noise, average ADE increases to about two point five five meters.

## Slide 3 — 1:55–3:00

The turning example illustrates this gap. Some predictions follow the future more closely, while others diverge. Our diagnostics identified one possible mechanism: the decoder feeds a sampled displacement into the next velocity estimate, so an early disturbance can influence several later positions. Also, selecting a checkpoint using teacher-forced likelihood does not directly evaluate the quality of a complete rollout.

A small control experiment did not solve the problem, so I am keeping that limitation explicit. The implementation and exploratory analysis are packaged, but the group's formal evaluation is still pending. It requires deterministic prediction, shared trajectory windows, and the same target-only perturbations as the other models. Until those checks pass, these numbers stay outside the common leaderboard. This gives us a reproducible starting point and a clear explanation of what needs validation before the final comparison.

## Sources and delivery notes

- Alahi et al. (2016), Social LSTM: Human Trajectory Prediction in Crowded Spaces, https://doi.org/10.1109/CVPR.2016.110.
- Numeric evidence: evidence/legacy_macro_summary.csv and the separate local diagnostic study, described in report.tex.
- Show the complete trajectory legend. Do not describe the selected failure as representative.
- Transition to the next owner after the last sentence. Rehearse once and trim the first-slide implementation detail if speaking slowly.
