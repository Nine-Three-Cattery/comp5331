# Implementation and verification provenance

Model owner: Cheung Alan Pak To.

The generator, pooling, discriminator, loss and training code were
implemented from the paper's method. The package uses the existing Group 7
neutral loader, perturbation routines, shared evaluator and audit through
their published interfaces. No model source, trained weights or predictions
from the paper's artifact were imported. The [source inventory](source_inventory.md)
records the external references and shared interfaces. The
[reuse ledger](reused_fragments.csv) has no copied source entries.

Generative AI assisted with reading and summarizing the paper, drafting
the Social-GAN implementation and regression tests, executing and
checking the experiments, and drafting the English report, Beamer frames
and speaker notes. Numerical results come from the saved run outputs and
the shared evaluator. The contribution is pending the owner's and group's
review.

The local run used an Apple M5 Pro CPU with two PyTorch threads per process,
CPython 3.14, torch 2.14.0, NumPy 2.5.3 and pandas 3.0.6. The benchmark
library versions match the common pins. The operating system, architecture
and Python version differ from the reference Linux x86-64 environment.
Timing is specific to this host and measurement boundary. The runtime
logs record each repetition and the contribution reports the median of
per-run medians. Reference-host reproduction is an editor review item.

The ETH seed 0 pilot completed both epochs and exported the full six
conditions in 10.1 s with a peak process RSS of 468 MiB, excluding the
separate timing pass and benchmark audit. The pilot receipt and exact
measurements are part of the local reproduction archive.

The proposal review corrected the runtime boundary to rebuild target
histories and the visible-neighbor index from neutral coordinates inside
every timed repetition. All fifteen runs were remeasured serially. Their
prediction bundles and headline accuracy results are unchanged.
An independent NumPy calculation checked 540 headline values across the
90 formal bundles; the largest absolute discrepancy was below 1e-15.
A fresh two-epoch pilot reproduced all 22 selected parameter tensors,
the validation records and all six export arrays and metadata exactly.
The review package repeats these checks using its archived source.
