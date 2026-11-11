# Trajectron++ agent handoff

The accepted experiment is the independent `causal_v2` run. For group
writing, edit the [Trajectron++ contribution](../submission/contributions/trajectronpp/README.md).
Use this directory for model and reproduction work.

- Keep the code path stable for the existing archive and commands.
- Read [README.md](README.md) and [independent_design.md](independent_design.md).
  The independent runtime entry points are in `independent_tpp/` and
  the causal scripts listed in the guide.
- Keep data, caches and full outputs local. Treat earlier upstream and
  pre-causal materials as historical reference. Use a new output root for
  any experiment that changes code, protocol, seeds or configuration.
- Run the causal-history regression and tiny metric fixture for input
  or evaluator changes. If an update changes audited numbers, refresh the
  group snapshot, figures and provenance together after a fresh audit.
