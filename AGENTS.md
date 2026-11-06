# COMP5331 Group 7 agent handoff

## Start here

1. Follow the [clone-to-merge SOP](CONTRIBUTING.md) and consult the
   [model registry](submission/models.json). Use the registry to identify
   the assigned owner, code directory and contribution ID.
2. Read the [benchmark entry point](benchmark/README.md) and
   [common specification](GROUP7_COMMON_DATA_EVALUATION_SPEC.md) before
   changing data, inference, metrics or experimental settings.
3. Read any AGENTS.md inside the target directory. Keep model work within
   its code directory and matching submission contribution folder.

## Experimental boundaries

- Preserve the five ordered holdout manifests, metric coordinates, target
  corruption fixture, and explicit K=1 or K=20 budgets.
- Derive model inputs from the visible history. Run a future-invariance
  regression for target and neighbor features used by the model.
- Use validation data for checkpoint selection. Record the raw release,
  manifest, configuration, checkpoint and prediction hashes. Register
  deviations and attribution with the owning model.
- Treat shared contracts and result comparability as group decisions.
  Propose shared changes separately and identify which model results need
  to be revalidated. Keep pending contributions pending until audited.

## Report and deck

- Assemble English documents from the shared masters in submission/.
  Write each model's subsection, frames and speaker notes in its
  contribution folder. Leave titles, common methods, layout, bibliography
  style, comparison and conclusions to the group editors.
- Use the shared metric macros and stable figure labels. Publish small
  audited result tables and figures. Build each frame around a claim and its evidence. Report per-fold or
  paired differences where they change the interpretation. Keep
  cross-model claims tied to comparable evidence.
- Run the benchmark gate, owner pilot, formal score and audit, then build
  the owner preview. The editor runs the final review and release checks.

## Repository hygiene

- Commit source, configs, tests, benchmark fixtures and lightweight
  submission evidence. Keep raw data, model weights, full predictions,
  generated PDFs and release archives in ignored local paths or an
  approved artifact store.
- Keep course originals, student numbers, credentials, personal contact
  details, absolute machine paths and upstream checkouts out of commits.
- Keep the audited Trajectron++ release and existing code paths usable.
  Run a model-specific smoke test when changing executable code and
  record remaining validation in the handoff. Stage only the intended
  changes and inspect the staged file list before any authorized push.
