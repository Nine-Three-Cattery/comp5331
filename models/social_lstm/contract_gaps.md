# Legacy versus causal_v2: not adapted

The shared contract and registry govern formal work. This directory preserves exploratory code; it does not change that agreement.

| Item | Legacy | Required shared work |
|---|---|---|
| Model/budget | Gaussian K20 with sampled-velocity residual mean | Registered deterministic K1 definition and training loss; one random sample is not deterministic. |
| Seeds | 11/22/33 | 0/1/2, explicit sampling seed metadata. |
| Raw input | Six pinned coordinate mirror files | Exact shared loader/release identity; filenames alone are insufficient. |
| Split/population | 80% temporal boundary, validation +20 steps, crossing IDs purged; train stride4 and val/test stride1 | Shared ordered causal_v2 manifests, target IDs and boundary exclusions on both sides; no silent filtering. |
| Context | Observation-complete packed pedestrians, future-complete scoring targets | Shared target-indexed records and causal observed-neighbor features. |
| Future graph | Train isolates observation-only context after observations; inference rolls it forward | Reconcile the graph-distribution mismatch without future-coordinate features or changing test IDs. |
| Corruption population | All target/context observed histories | Target history only; shared masks and derivatives recalculated after corruption. |
| Corruption RNG | PCG64/SHA of legacy ID+condition | RandomState seed100003*training_seed+1009*manifest_index and exact shared draw order. |
| Missing suffix | Causal CV fill, no exported bool mask | Shared fixture plus observed_mask and neutral records. |
| Conditions | clean/noise_0.05/noise_0.10/noise_0.20/missing_1/missing_2 | Shared clean_0/noise_005/noise_010/noise_020 path tags and record condition/level fields. |
| Bundle | pred[K,N,12,2], legacy IDs and offsets | samples[N,K,12,2], ground_truth, observed_mask, ordered records, identities/config/checkpoint metadata. A transpose alone is insufficient. |
| Evaluation | Old independent metrics and pair-time diagnostics | Shared evaluator, training thresholds, spatial summaries and audit. |
| Selection | Lowest clean-val teacher-forced NLL, earlier epoch tie | Predeclare K1 criterion/tie-break and export shared selection identities. |
| Runtime | Local CPU/MPS 8/16/32-person tensor latency | Reference host, ordered128 targets/batch32, shared stage boundaries. |
| Acceptance | Legacy smoke/owner tests | Shared benchmark, full ETH seed0 pilot, fivefold formal matrix, score/audit and editor review. |

The future-invariance tests vary target and neighbor future coordinates while keeping observed population fixed. They certify only the legacy boundary, not an unimplemented adapter. Future annotation availability marks scoring targets; production inference does not read future truth. The future-context discrepancy is a training/inference graph-distribution issue, not demonstrated inference leakage.

The residual mean is a substantive paper departure. Frozen-head innovation variance grows with sum(h^2); real checkpoint impulse responses persisted. A validation-only anchored control did not improve. Neither replacing this skip nor changing K is a validated repair. Preserve pending status until the model definition and shared experiments are reviewed.
