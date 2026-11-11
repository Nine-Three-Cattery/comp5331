# Social-GAN proposal and instructor-feedback review

Owner: Cheung Alan Pak To. Branch: `model/social_gan`; review base:
`group7/project`. This checklist covers the owner's contribution. The full
six-model project requires the group editor's integration and peer review.
The source proposal and Raymond's proposal comments remain outside the
repository because they are course originals.

## Owner requirements

| Requirement | Evidence and outcome |
|---|---|
| Read the original paper and independently implement the core | [Paper notes](../../../models/social_gan/paper_reading_notes.md), [paper-to-code map](../../../models/social_gan/independent_design.md), [source inventory](../../../models/social_gan/source_inventory.md) and [reuse register](../../../models/social_gan/reused_fragments.csv). Generator, discriminator, social max pooling, latent sampling, variety loss and training loop are implemented locally. No official model source or pretrained weights were imported. |
| Document departures | The design and [English report](report.tex) state the local neighborhood, constant-velocity residual, target discriminator, dimensions, optimizer, batch size, epoch budget and selection rule. These experiments reproduce the paper's core components under the group protocol; they do not reproduce the paper's exact settings or headline scores. |
| Common data preparation and causal inputs | Calls the group's independently implemented neutral loader and perturbation interface. Eight observations, twelve predictions, 0.4 s spacing and metric coordinates; identical ordered manifests, with no owner filtering. Visible-history derivatives and reversible centering are in `HistoryIndex`. [Regression tests](../../../models/social_gan/own_unit_tests/) check future invariance, coordinates, pooling, visible-neighbor sensitivity and training gradients. |
| Pilot, full folds and seeds | Separate ETH seed 0 two-epoch pilot, followed by all five folds and seeds 0/1/2. Fifteen validation-selected checkpoints yield ninety condition bundles. [Audit receipt](benchmark_audit.json) and [results](results.csv) bind the ordered manifests, configurations, checkpoints and predictions. No fold or seed budget was reduced. |
| Independent metric calculation and known examples | [Independent NumPy evaluator](../../../models/social_gan/check_metrics.py) passes analytic K=1 and K=2 fixtures, including different ADE and FDE winners. [Receipt](evidence/independent_metrics.json) checks all 540 min/mean ADE/FDE, Energy Score and diversity values against the formal CSV. This additional owner check was completed during the proposal review, after the original formal runs. |
| Accuracy, typical-sample error and diversity | [Per-fold clean table](evidence/clean_by_fold.csv), [sample-budget analysis](evidence/sample_budget.csv) and report distinguish best-of-twenty from mean error. The report shows seed standard deviations and equal-fold aggregation. Prefix K analysis reuses existing draws and is not a separate retraining experiment. |
| Common noise and missing-history experiments | All six prescribed conditions use the same selected model and paired latent draws. [Paired changes](evidence/paired_robustness.csv) and [robustness figure](assets/paired_robustness.pdf) retain holdout differences. Shared perturbations affect the target only; neighbors and future labels remain fixed. |
| Motion and proximity | [Per-run diagnostics](evidence/motion_spatial.csv), [macro summary](evidence/motion_spatial_macro.csv) and report cover speed, acceleration, jerk and close approaches using training-only thresholds. A close-approach event is any pair below the threshold at any future step; it is not a collision-avoidance guarantee. |
| Runtime | [Timing table](evidence/runtime_cpu.csv) covers all fifteen runs, 128 locked targets, batch 32, K=20, three warmups and ten repetitions. Review corrected the boundary to rebuild history and neighbor indices inside each repetition, then remeasured all runs. Neutral file and model loading remain outside the interval. Timings are labeled as local CPU results. |
| Fixed examples and failure analysis | [Window IDs and measures](evidence/trajectory_examples.csv) and [figure](assets/trajectory_examples.pdf) link the displayed paths to saved exports. The complete archive contains the first/middle/last manifest examples per source scene and the worst clean minADE example, across conditions. Selection follows the existing Trajectron++ convention; the editor must confirm shared visual examples before cross-model synthesis. |
| Reproducible training and artifacts | [Fresh pilot receipt](evidence/training_reproduction.json) records exact agreement of validation records, all 22 selected model tensors and six export bundles. [Handoff tool](../../../models/social_gan/handoff.py) verifies archived source, selected checkpoints and all formal bundle hashes; its fresh-checkout check tests the archived source, replays the formal ETH checkpoint, retrains the pilot, independently calculates scores and runs the common audits. [Reproduction sidecar](reproduction_check.json) is bound to the archive hash. |
| English owner handoff | [Report](report.tex), [three Beamer frames](slides.tex), [speaker notes](speaker_notes.md), result tables and figures use the shared masters. [Provenance](provenance.json) records source and evidence hashes, runtime environment, archive filename and SHA-256. Build and reproduce commands are in the [model README](../../../models/social_gan/README.md). |

## Response to Raymond's comments

| Comment | Owner response and remaining group responsibility |
|---|---|
| Clearly identify all six models and exact data/metric protocol | The unchanged [registry](../../models.json) lists Individual LSTM, Social-LSTM, Social-GAN, Social-STGCNN, Trajectron++ and DSPIGCN. [The common specification](../../../GROUP7_COMMON_DATA_EVALUATION_SPEC.md) and [executable contract](../../../benchmark/README.md) pin LOSO splits, time-disjoint validation, lengths, sampling and metrics. Social-GAN uses that established pipeline. Other owners remain responsible for their model papers and artifacts. |
| Obtain the original papers and official repositories | Social-GAN's original CVPR paper and authors' repository are identified in the source inventory. The paper drives the implementation; the repository is a reference for official artifact identification. No substantial source was copied. |
| Independently implement all models and the common loader/evaluator | The owner implements model-specific features, architecture, loss and training. The existing Group 7 loader/evaluator supplies agreed interfaces; its implementation provenance is maintained with that shared code. Reuse of these group interfaces is explicit. The owner also supplies an independently written six-metric cross-check. |
| Standardize normalization, pooling, sampling and splits; explain departures | Positions remain in meters, with reversible centering and no fitted normalization. Social context, latent coupling, training/testing draw counts and all paper departures are documented in the design. Model-specific pooling is explicit; it does not change the common evaluated population. |
| Plan resources and establish reproducibility early | The original full-manifest pilot preceded formal training and passed the common gate. It took 10.1 s and 468 MiB peak process RSS, excluding timing and audit. Formal runs use two CPU threads per worker and at most five workers; each has a 20-epoch ceiling and patience five. CPU execution required no shared-GPU booking. Run records preserve actual elapsed time and memory. No broad hyperparameter search was performed. The proposal's eight-week schedule remains the group plan; local completion does not certify other members' milestones. |
| Attribute external material and permitted minimal reuse | The source inventory and empty copied-fragment register identify paper, dataset, library and group-interface use. Assistance is disclosed in implementation provenance and the handoff metadata, as required by the course. |

## Items that remain open for group acceptance

- Reproduce on the SOP's reference Linux x86-64 environment. The completed
  local runs and fresh-checkout checks use macOS arm64; they are not evidence
  that a Linux execution has passed.
- Use the same host and device for any cross-model runtime comparison, and
  confirm that each owner's boundary includes its model-specific preprocessing.
- Transfer the full archive through the group's agreed artifact location.
  The archive is local and ignored by Git; the source PR alone does not
  transfer checkpoints and prediction bundles. Its filename and SHA-256 are
  in provenance. No group storage URL is configured in this contribution.
- Obtain owner and peer review, then let the editor mark Social-GAN ready.
  Its registry entry remains pending. Complete the other model contributions,
  combined comparison, final report/deck and release through the editor's
  workflow; these are not certified by the Social-GAN audit.
