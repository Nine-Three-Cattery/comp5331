# Contract mapping and remaining acceptance

All 15 local formal runs (five folds, seeds 0/1/2) and 90 K=1 prediction bundles passed the common causal_v2 audit. See submission/contributions/social_lstm/benchmark_audit.json and results.csv.

- Common manifests, ordered IDs, metric coordinates, target-only corruption and six conditions were checked by the shared scorer/auditor.
- Observed-only context, derivative recomputation, pool isolation, padding and complete training bucket coverage passed owner tests and independent checks.
- Every run saved validation candidates and selected one checkpoint by clean validation ADE, with earlier-epoch tie breaking; that checkpoint served all conditions.
- All 15 runs have synchronized runtime measurements: 128 ordered targets, batch 32, three warmups and ten repetitions. Timing begins with cached neutral histories and excludes initial disk/window caching and scoring.
- The frozen deterministic social-pooling adaptation uses a velocity anchor and closed-loop MSE. It is not the paper's Gaussian model. Legacy K20 experiments remain separate.

Remaining acceptance: independent group editor review, Linux/Python 3.12/PyTorch 2.14 reference reproduction, the group's hardware timing policy, and transfer of the large release artifact to agreed storage. Local execution used macOS MPS/Python 3.13/PyTorch 2.11. The shared registry remains pending until editor acceptance; local experimental work is complete.

Clean ADE is 0.607 m versus the identical-window constant-velocity baseline's 0.534 m. Audit success establishes interface compliance, not superior forecasting quality.
