# Social-LSTM: three-minute speaker notes

Presenter: WANG Yifan. Three content slides for the integrated group presentation. Spoken script: 376 words, approximately 3 minutes at 125 words per minute. Values below use the shared deterministic K=1 evaluation.

## Slide 1 — 0:00–1:00

My part is Social-LSTM. Its central idea is to predict each pedestrian using both their own motion and information from nearby people. A recurrent network summarizes each person's history. A spatial grid then pools neighboring hidden states, so the next prediction can respond to the local crowd.

For our group's deterministic comparison, I implemented a variant that observes eight positions and predicts one twelve-step future. It starts from a velocity estimated from the last three observed intervals and learns a recurrent correction. I train the complete predicted trajectory directly, without using the true future as the next input. This differs from the original paper's probabilistic decoder, and that adaptation is documented in the report.

## Slide 2 — 1:00–2:00

The formal experiments use the same test trajectories as the other models: five held-out scenes, three training seeds, and six observation conditions. Each run selects its checkpoint using clean validation error. The test set never decides which checkpoint we keep.

On clean histories, our macro ADE is 0.607 meters, compared with 0.534 meters for constant velocity on exactly the same test windows. Constant velocity therefore remains the stronger aggregate baseline in this experiment. These are single-trajectory errors, so there is no best-of-twenty selection.

The robustness experiments change only the target's history. Neighbor positions stay unchanged. With zero point two meters of observation noise, ADE is 1.128 meters. Removing the final two observations gives 0.761 meters. These conditions help reveal how strongly the model depends on an accurate estimate of recent motion.

## Slide 3 — 2:00–3:00

This figure shows the same turning pedestrian under three conditions, using the same checkpoint. The dark line is the observed input, the green line is the true future, and the orange line is our one predicted trajectory. This is a selected example, so the full tables remain the evidence for overall performance.

I also checked that changing future coordinates cannot affect model inputs, that padding does not change predictions, and that all ninety prediction files match the shared evaluation contract. Runtime measurements use our Mac's MPS device and are reported separately from other hardware.

The main contribution is an independently implemented social-pooling baseline with a complete, traceable evaluation. Its limitations include the deterministic forecast and the declared differences from the original paper. These results can now support the group's common comparison.

## References and rehearsal

Alahi et al. (2016), Social LSTM: Human Trajectory Prediction in Crowded Spaces. https://doi.org/10.1109/CVPR.2016.110.

Evidence: results.csv; evidence/causal_macro_summary.csv; evidence/causal_constant_velocity.json; evidence/causal_qualitative_selection.json; benchmark_audit.json. Interface audit passed locally; editor and reference-environment review are separate.
