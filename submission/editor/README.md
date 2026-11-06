# Group editor checklist

## Milestones

Use the course deadline as day T. At T−7 lock the common benchmark, record
the presentation time and arrange GPU slots. At T−5 check each owner's ETH
pilot and estimate the formal workload. At T−3 finish the five-fold runs
and resolve protocol deviations. At T−2 review and merge model branches.
At T−1 rehearse, verify the final claims and export the package. If the
course schedule leaves less time, preserve the order of these gates.

## Review each model

1. Check the model source, tests, paper notes, dependency lock, deviations,
   source inventory and reuse ledger. Confirm the submitted model was
   independently implemented. The common validator checks outputs, while
   the reviewer checks code and attribution.
2. Check the causal and coordinate tests and validation selection log.
   Retrieve the owner's artifact, verify its SHA-256, reproduce a
   selected forecast and rerun `make audit MODEL=… RUNS=…` on the full
   bundles. Copy the resulting `benchmark_audit.json` into the contribution.
3. Verify timings used the shared window IDs, batch, warmups, repeats and
   stage boundaries. Use the same physical host and device for a runtime
   ranking. Otherwise label hardware-specific timings separately and set
   `runtime_policy` to `separate_hardware`.
4. Complete the model entry in `review.json` using the
   [review template](../templates/review.json). Record a reviewer distinct
   from the owner and SHA-256 of both `benchmark_audit.json` and
   `provenance.json`. Set each review field to true only when its evidence
   is checked. Register the model as `ready` and merge the model branch.

## Assemble the submission

- Add `report_synthesis.tex` and `slides_synthesis.tex`. Compare paired
  fold and condition results, distinguish single forecasts from best-of-20,
  and explain measured runtime boundaries and remaining deviations.
- Include the constant-velocity sanity check as supplementary evidence.
  Keep the 6-model table restricted to the six independent models.
- Inspect report, Beamer slides, figures and speaking transitions. The
  title, benchmark, model frames and synthesis should fit the scheduled
  presentation.
- Run `make final-docs`, commit the reviewed source, then run `make release`
  from a clean branch. The release command compiles the documents from the
exported commit, so the PDFs and source archive represent the same tree.
`artifacts/group7/` contains `code.zip`, `report.pdf`,
  `slides.pdf`, `results.csv`, `speaker_notes.md`, `SOURCE_COMMIT` and
  `SHA256SUMS`. Attach the individually archived artifacts through the
  course's agreed storage location. Check the packaged docs against
  Raymond's requirements below.

| Instructor requirement | Group evidence |
|---|---|
| Six concrete implementations and a feasible schedule | Registry, owner plans and pilot measurements |
| Common data loader before formal comparison | Manifest lock and `make benchmark` |
| Common evaluator and comparable experiments | `make score` and current `make audit` receipts |
| Differences and resource planning documented | Owner deviations, validation logs and timing records |
| Independent source with attribution | Owner source inventory, reuse ledger and peer review |
