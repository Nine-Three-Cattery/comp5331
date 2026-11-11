# Social-GAN contribution

Owner: Cheung Alan Pak To. The Social-GAN branch is ready for group
review under the `causal_v2` benchmark. The model registry remains
pending until the editor records an independent review.

- [Report](report.tex), [Beamer frames](slides.tex) and [speaker notes](speaker_notes.md)
- [Audited fold, seed and condition results](results.csv)
- [Benchmark audit](benchmark_audit.json) and [artifact provenance](provenance.json)
- [Clean results by fold](evidence/clean_by_fold.csv), [sampling-budget results](evidence/sample_budget.csv)
  and [paired robustness results](evidence/paired_robustness.csv)
- [Robustness figure](assets/paired_robustness.pdf) and [example window IDs](evidence/trajectory_examples.csv)
- [Runtime by fold and seed](evidence/runtime_cpu.csv) and [motion and proximity diagnostics](evidence/motion_spatial.csv)

Follow the [owner reproduction commands](../../../models/social_gan/README.md)
to replay a selected checkpoint and verify the archive. The formal timing
uses an Apple M5 Pro CPU with two PyTorch threads. Use the same host,
device and timing boundary for any runtime ranking.
