# Implementation provenance

**Model owner:** Zongmin Zhang, COMP5331 Group 7.

The model, its full-history graph adapter, training loop and forecast exporter were written for this course contribution. The architecture was derived from Mohamed et al. (CVPR 2020). Two data/evaluation utilities in the authors' public repository were subsequently consulted to clarify the coordinate and graph conventions. The independent network uses the published inverse-distance spatial graph and normalized adjacency. The [source inventory](source_inventory.md) identifies the paper, those two locations, the group interfaces and the dependencies. The [reuse ledger](reused_fragments.csv) contains no copied code fragments. All learned parameters are initialized for the registered run.

Generative AI assistance was used to interpret the paper, plan and implement the model and tests, run and diagnose experiments, analyze outputs, and draft the English report and presentation. The group receives the source, configurations, selected checkpoints, predictions and audit records for review. The model registry keeps the academic acceptance decision with the group editor.

After the formal forecasts were frozen and first published, the owner checked the released model module to document its decoder axes, wrote a separate ADE/FDE calculation, and applied the group's heading diagnostic to the saved draws. These follow-up analyses use the original validation-selected weights and forecasts. [Architecture comparison](paper_reading_notes.md#comparison-with-the-released-network-after-training) · [Review analysis](review_metrics.py)
