# Contribution provenance

Assigned owner: YAN Zhengyang. This package was authored with OpenAI Codex
assistance at the owner's request. AI assistance covered paper/feedback
inspection, implementation, tests, experiment tooling and documentation.
The owner should review and be able to explain the code and report before
course submission; this record does not assert unaided authorship.

The model core, objective, polynomial derivatives and anchor assignment were
written for this project from the paper's equations. The public reference was
read to identify ambiguous details and differences, not copied or wrapped.
The group-owned loader/evaluator are intentionally shared, as requested in
lecturer feedback items 2–4. `reused_fragments.csv` contains no third-party
copied fragments. `source_inventory.md` distinguishes algorithms, data and
shared infrastructure.

Each run saves config (including train/validation identities and environment),
training history, validation candidate table, selected checkpoint and complete
predictions. Shared audit binds scores to configs, weights, predictions and
test manifests. Final release provenance will identify the actual run matrix,
artifact digest and measured runtime; missing experiments must remain pending.

Known implementation differences and the registered selection policy are in
`independent_design.md`. No comparison with published table values is claimed
without matching their preprocessing, coordinate and sampling conventions.
