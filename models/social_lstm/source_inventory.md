# Sources and reuse inventory

No upstream neural model, pretrained weight or checkout is imported. The new causal branch uses the group-owned common loader/protocol and shared evaluator as required; no official neural model core is reused. **No copied code** from upstream model repositories; `reused_fragments.csv` is intentionally header-only. References and assistance were still used and are disclosed here.

| Source | Version / identifier | Use |
|---|---|---|
| Alahi et al., Social LSTM | CVPR2016, DOI10.1109/CVPR.2016.110 | Architecture, social tensor and Gaussian loss equations. |
| quancore/social-lstm, GitHub | model.py and grid.py inspected during local development; no commit pin captured | Read tensor layouts and embedding interfaces; no source committed or executed. No license-dependent fragment copied. |
| abduallahmohamed/Social-STGCNN | commit333d3a57b4d2705e129b21aefefa09c79b2b9ae1 | Coordinate mirror only; exact raw URLs/hashes in configs/raw_release.json. No neural code reused. |
| ETH / UCY | Pellegrini et al. ICCV2009; Lerner et al. CGF2007 | Original trajectory datasets; provider terms apply. Raw files stay local. |
| Gneiting and Raftery | JASA2007, DOI10.1198/016214506000001437 | Energy-score background. |
| Owner-authored exploratory code | Local source snapshot2026-11-05; per-file SHA256 in configs/local_source_snapshot.json | This port preserves the owner's own core with portable paths and documentation. |
| Group-owned common input/protocol/evaluator | li_junhao_trajectronpp/independent_tpp/data.py and protocol.py; benchmark/ | Imported shared neutral data/corruption identities and common scoring; not official T++ neural code. |
| New owner causal implementation | causal/*.py; source_hashes.json | Independent deterministic model, closed-loop trainer, neutral exporter and runtime CLI. |
| PyTorch / NumPy | shared benchmark environment; legacy versions separate | General numerical libraries, not a third-party model core. |

URLs: [paper](https://doi.org/10.1109/CVPR.2016.110), [inspected reference](https://github.com/quancore/social-lstm), [data mirror](https://github.com/abduallahmohamed/Social-STGCNN), [energy score](https://doi.org/10.1198/016214506000001437).

The unpinned inspected reference is a provenance limitation, not an exact upstream reproduction claim. Primary-paper equations and explicit tests define this implementation. Local snapshot hashes identify pre-port owner source; `source_hashes.json` identifies portable source. Neither contains machine paths or historical model artifacts.
