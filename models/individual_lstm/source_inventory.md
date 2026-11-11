# Source inventory

- Original project sources: COMP5331 group proposal and Raymond's comments; private course PDFs intentionally omitted.
- Prior owner deliverable: module1-v1/trajpred/models.py (October 2026), preserved unchanged as model.py. Original AI-assisted course implementation, not a third-party repository.
- Shared group code: benchmark/setup.py, benchmark/validate.py, li_junhao_trajectronpp/independent_tpp/{data,protocol}.py, and scripts/{evaluate,analyze_sample_budget}.py at df016def5890898b70d372fb0e8e2bc142121330. Used as the required group interface. No Trajectron++ model imported.
- Data: six ETH/UCY metric text files, actual SHA-256 checked against li_junhao_trajectronpp/configs/causal_v2_raw_sha256.json. The legacy Social-STGCNN and locked Trajectron++ distributions have byte-identical versions of these six files.
- Literature: https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf ; https://arxiv.org/abs/1803.10892 .
- Dependencies: PyTorch LSTM, LSTMCell, Adam and autograd; shared benchmark's NumPy and pandas.
