# Implementation provenance

LI Junhao's final Trajectron++ contribution is the trajectory-only
`causal_v2` model. The active implementation consists of the
`independent_tpp` package and the causal training, adapter, data,
evaluation and analysis scripts enumerated in the reproducibility archive.
The earlier compatibility implementation and predictions remain historical
course-workspace artifacts.

The source inventory records the pinned algorithm reference and data origin.
`reused_fragments.csv` has a header and zero entries. The active independent
core and harness contain no identified copied code fragments from the
official model repository. The review checked imports and searched the active
core and harness against the upstream Python source for contiguous exact
36-token matches. The scope of this declaration is the shipped
independent implementation. The independent audit checks its imports and the
relocation check runs the prediction/evaluation entry points without the
official repository. Upstream text annotations are dataset inputs and are
separately attributed.

The prediction index links every test output to the model identifier, fold,
seed, ordered manifest, configuration and selected checkpoint digest. Stage
records preserve their input signatures and output SHA-256 values. The
archive checksum index uses relative paths.
