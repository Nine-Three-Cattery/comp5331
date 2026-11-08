# Sources and assistance

- Runkang Guo et al., **DSPIGCN: Dual-stream Physics-informed Graph Convolutional
  Network for Reliable Pedestrian Trajectory Prediction**, KDD 2026, pp. 383–394.
  https://doi.org/10.1145/3770854.3780329
- Author artifact: https://doi.org/10.5281/zenodo.18144387 (version 18144467).
  Source inspected at https://github.com/shangfeng2000/PI-STGCN , commit
  d38fd905e51d401a5e812ae561185ac4a423c2e0; and
  https://github.com/shangfeng2000/DSPIGCN , commit
  c7d14f7b8dcc0130a0927a9464235a0a15b35927. Both provide reference architecture,
  losses, data format and evaluation code. The included license is MIT,
  copyright (c) 2020 Abduallah Mohamed. Its text is retained in
  `AUTHOR_ARTIFACT_LICENSE.txt` for attribution. No author's pretrained weights
  or claimed results are presented as the student's trained model/results.
- Abduallah Mohamed et al., **Social-STGCNN**, CVPR 2020.
  https://arxiv.org/abs/2002.11927 ; https://github.com/abduallahmohamed/Social-STGCNN .
  The DSPIGCN release derives graph/convolution components from this lineage.
- ETH/UCY datasets: Stefano Pellegrini et al. (ICCV 2009), Alon Lerner et al.
  (Computer Graphics Forum 2007), as cited in the project proposal. The model
  code's MIT license does not independently license the underlying data. Data
  are obtained from the cited release and not committed as newly authored work.
- Tilmann Gneiting and Adrian E. Raftery (2007), **Strictly Proper Scoring Rules,
  Prediction, and Estimation**, https://doi.org/10.1198/016214506000001437 .
  Sample-based energy score; exact empirical convention in `PROTOCOL.md`.
- PyTorch, NumPy, scikit-learn and Matplotlib supply numerical primitives,
  attention, GMM fitting and figures. Actual versions are exported per run.

The package is newly written code guided by the cited sources, with deliberate
deviations documented separately. It was prepared with OpenAI Codex AI coding
assistance for implementation, testing and documentation. Zhengyang Yan should
review the implementation, final report and course disclosure requirements.
