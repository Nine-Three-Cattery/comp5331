# Source and license inventory

| Source | Use | Attribution and terms |
|---|---|---|
| Gupta et al., CVPR 2018 | Generator, pooling, discriminator and variety-loss design. | [Proceedings paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Gupta_Social_GAN_Socially_CVPR_2018_paper.html). The model code and weights were developed for this contribution from the paper. |
| Pellegrini et al., ICCV 2009, and Lerner et al., CGF 2007 | ETH and UCY pedestrian trajectory annotations. | Original dataset terms apply. Both papers are cited in the [group bibliography](../../submission/references.bib). |
| Stanford Autonomous Systems Lab, Trajectron++ data release | Source of the six coordinate files selected by Group 7. The shared setup checks the expected SHA-256 for each file. | [Pinned raw files](https://github.com/StanfordASL/Trajectron-plus-plus/tree/ef0165a93ee5ba8cdc14f9b999b3e00070cd8588/experiments/pedestrians/raw/raw/all_data). The data retain their original terms. |
| Group 7 `causal_v2` benchmark | Neutral position loader, corruption rules, manifests, scoring and audit. | [Shared benchmark contract](../../benchmark/README.md). The owner package calls the shared interfaces and implements its own model, feature adapter and training loop. |
| Gneiting and Raftery, JASA 2007 | Definition and interpretation of Energy Score. | [Journal article](https://doi.org/10.1198/016214506000001437). The common evaluator computes the project version. |
| PyTorch, NumPy, pandas and Matplotlib | Training, data handling, shared scoring and figures. | [PyTorch BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [NumPy BSD license](https://numpy.org/doc/stable/license.html), [pandas BSD license](https://pandas.pydata.org/docs/getting_started/overview.html#license) and [Matplotlib license](https://matplotlib.org/stable/project/license.html). |

AI-assisted research, implementation and editing were used in preparing this
contribution. No Social GAN source fragments or pretrained weights were used.
See the [fragment register](reused_fragments.csv) for the code reuse record.
