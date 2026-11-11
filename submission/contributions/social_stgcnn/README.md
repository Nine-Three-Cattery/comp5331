# Social-STGCNN · Zongmin Zhang

This independent implementation reproduces the full-scene operators of Mohamed et al.'s public network. It keeps a feature vector for each pedestrian through the temporal decoder and exports twenty paths for each scored target. The implementation has 6,254 active parameters. The authors' module registers 7,563, including 1,309 in layers its forward pass skips. [Architecture and differences from the paper](../../../models/social_stgcnn/paper_reading_notes.md) · [Independent source and run commands](../../../models/social_stgcnn/README.md)

## Formal results

Distances are in meters. Each cell averages target windows within a run, three seeds per holdout, and then the five holdouts equally. The data contain 34,161 distinct test windows. Minimum ADE and minimum FDE select the best of twenty draws separately for each target.

| Target history | minADE | minFDE | meanADE | meanFDE | Energy Score | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.583 | 1.006 | 0.971 | 1.760 | 0.796 | 0.622 |
| Gaussian noise, 0.05 m | 0.599 | 1.029 | 0.998 | 1.801 | 0.819 | 0.634 |
| Gaussian noise, 0.10 m | 0.651 | 1.101 | 1.076 | 1.915 | 0.888 | 0.657 |
| Gaussian noise, 0.20 m | 0.823 | 1.339 | 1.312 | 2.258 | 1.100 | 0.723 |
| Last point missing | 0.614 | 1.051 | 1.007 | 1.808 | 0.832 | 0.622 |
| Last two points missing | 0.662 | 1.110 | 1.061 | 1.872 | 0.884 | 0.623 |

[Formal score table](results.csv) · [Scene, sampling, crowd, timing and motion results](evidence/) · [Matched ETH window across three input conditions](assets/shared_eth_case.pdf)

## Evidence for review

The two-epoch ETH pilot covered 2,395 training scenes, 26,961 scored training targets and the full 364-target test manifest under all six conditions. The training, export and timing command took 38.14 seconds with a peak resident set of 443,060,224 bytes. The formal experiment trained fifteen fresh models for 250 epochs each. The winner among 51 clean-validation candidates ranged from epoch 95 to 250. Re-evaluation of all selected checkpoints with uncached graphs and a separate double-precision Gaussian likelihood reproduced the saved validation NLLs to within 6.0 × 10⁻⁸ nats per step. [Pilot receipt](evidence/pilot_audit.json) · [Selection check](evidence/selection_check.json)

The course model and the authors' code agreed on all-node outputs, active parameter gradients and batch-normalization state in six numerical cases, including a recorded ETH example. We also loaded each trained checkpoint into the authors' network and regenerated twenty forecast paths for one fixed target per holdout, using the same noise as the formal export. The 45 comparisons span three seeds and three conditions. The maximum difference across 21,600 coordinates was 3.82 × 10⁻⁶ m, below the prespecified tolerance of 0.0001 m. [Operator checks](evidence/source_parity.json) · [Trained reference comparison](evidence/trained_reference.json)

The group evaluator independently rescored all 90 forecast bundles. A separate calculation of all four displacement columns had a maximum difference of 8.9 × 10⁻¹⁶ m. A fresh checkout reproduced all 174,720 clean ETH seed-zero forecast coordinates exactly, together with the masks, future labels and ordered target records. Unpacking the finished archive verified 256 listed files, including all 90 formal and six pilot bundles. [Group audit](benchmark_audit.json) · [Owner calculation and heading diagnostics](evidence/review_metrics.csv) · [Fresh-source replay](evidence/reproduction_check.json) · [Unpacked artifact replay](evidence/extracted_reproduction_check.json)

## Report, talk and reproduction package

[English report section](report.tex) · [Three Beamer frames](slides.tex) · [Speaker notes](speaker_notes.md) · [Individual contribution](contribution_statement.tex) · [Archive identity and runtime](provenance.json) · [Reviewer instructions](../../../models/social_stgcnn/REVIEW.md)

`make preview MODEL=social_stgcnn` compiles the contribution in the shared group documents. The editor records the acceptance decision in the group registry. The [full-scene review release](https://github.com/Nine-Three-Cattery/comp5331/releases/tag/social-stgcnn-full-scene-causal-v2-review-v1) contains the standalone PDFs and the full numerical archive. The [earlier target-readout release](https://github.com/Nine-Three-Cattery/comp5331/releases/tag/social-stgcnn-causal-v2-review-v2) preserves the previous architecture and results.
