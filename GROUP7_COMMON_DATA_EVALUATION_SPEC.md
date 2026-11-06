# Group 7 统一数据、评估与独立实现规范

- **版本：** 2026-10-05
- **对应反馈：** Raymond 对 COMP5331 Group 7 Implementation Project 的反馈 2–4
- **项目：** Implementation and Comparison of Six Pedestrian Trajectory Prediction Models

这份文件是所有模型 owner 在开始正式模型实现前必须共同确认的接口规范。它的目的不是规定每个模型内部必须使用什么网络，而是固定：

1. 所有模型看见的样本是否完全相同；
2. 所有模型的预测输出是否可以由同一个 evaluator 读取；
3. 所有模型的误差、概率、鲁棒性和运行时间是否可以公平比较；
4. 参考论文和公开代码如何使用而不演变为直接复制。

当前可执行的 `causal_v2` 接口、共享 fixtures 和命令见
[benchmark 入口](benchmark/README.md)。模型 ready 状态由
[交付 registry](submission/models.json) 管理。

---

## 1. 六个固定模型

正式比较的六个模型是：

| 编号 | 模型 | 类型 | 是否概率输出 | owner 交付 |
|---:|---|---|---|---|
| 1 | Individual LSTM | 只使用目标行人历史的确定性 encoder-decoder | 否 | 独立实现；Constant Velocity 只作为 sanity check |
| 2 | Social-LSTM | LSTM + 邻居 social pooling | 否 | 独立实现 social pooling |
| 3 | Social-GAN | recurrent generator + pooling + discriminator + variety loss | 是 | 独立实现生成器、pooling、判别器和采样 |
| 4 | Social-STGCNN | spatial-temporal graph convolution | 以论文复现设置为准 | 独立实现图构建和时空卷积 |
| 5 | Trajectron++ | graph-structured recurrent CVAE + dynamics integration | 是 | 独立实现 pedestrian core |
| 6 | DSPIGCN | dual-stream physics-informed graph convolution | 是/以论文定义为准 | 独立实现 displacement/velocity streams、physics coupling、sampling |

Constant Velocity predictor 是公共 sanity check，不计入“六个模型”，也不占用一个成员 owner。

参考论文和官方/作者 artifact：

- [Social LSTM CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)
- [Social GAN arXiv](https://arxiv.org/abs/1803.10892)，[作者代码仓库](https://github.com/agrimgupta92/sgan)
- [Social-STGCNN arXiv](https://arxiv.org/abs/2002.11927)，[作者代码仓库](https://github.com/abduallahmohamed/Social-STGCNN)
- [Trajectron++ arXiv](https://arxiv.org/abs/2001.03093)，[官方仓库](https://github.com/StanfordASL/Trajectron-plus-plus)
- [DSPIGCN DOI](https://doi.org/10.1145/3770854.3780329)，[proposal 中记录的作者 artifact](https://doi.org/10.5281/zenodo.18144387)

公开仓库只能用于阅读算法细节、确认超参数和进行 sanity comparison。最终提交的六个模型必须是 group-owned implementations，不能直接 import、复制或包装官方模型核心代码。

---

## 2. 统一输入定义

### 2.1 原始记录

所有模型共享同一份 raw release 和 raw file hash。原始记录至少包含：

```text
scene_id, source_file, frame_id, timestamp, pedestrian_id, x, y
```

其中 `x,y` 是 metric/world coordinates。不得把不同模型的坐标单位、采样率或 frame convention 混在一起。

### 2.2 时间设置

```text
resampling interval: dt = 0.4 s
observation length:  8 points
prediction horizon:   12 points
observation interval: 7
```

每个窗口的记录必须明确：

```text
fold, split, scene_id, source_file, target_id, anchor_t,
obs_start, obs_end, fut_start, fut_end, window_id,
num_observed, num_future, dt, coordinate_unit
```

### 2.3 五折 leave-one-scene-out

| Fold | Held-out test | Development scenes |
|---|---|---|
| ETH | ETH | Hotel, Univ, Zara1, Zara2 |
| Hotel | Hotel | ETH, Univ, Zara1, Zara2 |
| Univ | Univ | ETH, Hotel, Zara1, Zara2 |
| Zara1 | Zara1 | ETH, Hotel, Univ, Zara2 |
| Zara2 | Zara2 | ETH, Hotel, Univ, Zara1 |

Development scenes 必须在生成窗口前按时间切分 train/validation；边界两侧固定排除 20 个 resampled steps，避免窗口跨越 train/validation 边界。

所有模型必须读取相同 manifest、相同 `window_id` 和相同 target population。若某个模型无法处理某个窗口，必须输出 explicit exclusion table，不能静默删除后继续比较。

### 2.4 派生状态和坐标变换

速度和加速度必须由重采样后的 metric coordinates、固定 `dt=0.4` 计算。
模型输入的派生状态只使用预测锚点及之前、该输入窗口中可见或因果补齐的位置。
目标和邻居的派生状态都应通过未来位置不变性测试。若模型内部使用 scene-centered 或 agent-centered coordinates：

1. 保存变换参数；
2. 预测后 inverse-transform 回原始 metric coordinates；
3. evaluator 只读取 inverse-transformed predictions；
4. 对每个模型执行 round-trip unit test。

### 2.5 输入限制

主比较只允许使用：

- target pedestrian trajectory history；
- nearby pedestrian trajectory history；
- trajectory-derived velocity/acceleration；
- 模型论文中明确要求的 trajectory graph representation。

主比较不允许使用：

- map image 或 semantic map；
- RGB/camera/LiDAR；
- 机器人未来 motion plan；
- held-out test scene 的统计量；
- 另一个模型生成的 pseudo-label 或 prediction。

这也是 Trajectron++ 当前 `use_map_encoding=false`、`incl_robot_node=false` 的 protocol 原因。

---

## 3. 统一扰动定义

扰动只改变 observed history，future ground truth 永远不变：

| Condition | 定义 |
|---|---|
| clean | 不改变 observed positions |
| noise-005 | observed positions 加二维 Gaussian noise，σ=0.05m |
| noise-010 | observed positions 加二维 Gaussian noise，σ=0.10m |
| noise-020 | observed positions 加二维 Gaussian noise，σ=0.20m |
| missing-1 | 隐藏最后 1 个 observed step |
| missing-2 | 隐藏最后 2 个 observed steps |

各模型必须使用相同 corruption seed、相同 observation mask 和相同 missing suffix fill rule。扰动后重新计算 velocity/acceleration；不能继续使用未扰动位置派生的状态。

---

## 4. 共享模型适配器接口

每个模型 owner 可以拥有自己的内部网络，但必须提供同一组外部接口：

```python
class ModelAdapter:
    def fit(self, train_manifest, val_manifest, config, seed): ...
    def select_checkpoint(self, val_manifest): ...
    def predict(self, observation_batch, num_samples, seed): ...
    def save_provenance(self, output_dir): ...
```

统一预测输出必须满足：

```text
samples:      [N, K, T, 2]
ground_truth: [N, T, 2]
observed:     [N, H, 2]
records:      N rows in exact manifest order, keyed by window_id
```

其中：

```text
N = window count
K = 1 for deterministic models, 20 for probabilistic models
T = 12
H = 8
```

每个 prediction bundle 还必须保存：

```text
model_name, fold, split, seed, checkpoint,
condition, sample_count, sampling_seed,
window_id, scene_id, target_id, anchor_t,
coordinate_unit, dt, config_hash, manifest_hash
```

评估程序不能 import 任何模型内部类；它只能读取这个中立格式。

---

## 5. 共享评估规范

### 5.1 位置误差

对每条样本 `k`：

\[
ADE_k=\frac{1}{T}\sum_{t=1}^{T}\|\hat p_{k,t}-p_t\|_2,
\qquad
FDE_k=\|\hat p_{k,T}-p_T\|_2.
\]

概率模型报告：

```text
minADE@20, minFDE@20
meanADE@20, meanFDE@20
sample diversity
sample-based energy score
```

确定性模型报告单条预测结果，不能把它伪装成 best-of-20。

### 5.2 聚合层级

固定采用：

```text
window → scene/fold mean → macro average over held-out scenes
```

最终表同时保留：

- 每个 window 结果；
- 每个 scene/fold 的均值；
- seed mean/std；
- overall macro mean/std；
- window count。

### 5.3 运动和空间诊断

在 metric coordinates 上统一计算：

- speed；
- acceleration；
- jerk / abrupt change；
- heading change；
- minimum pairwise distance / close approach。

阈值只由当前 fold 的 training data 确定。close approach 只能叫 spatial-consistency diagnostic，不能叫 collision-avoidance guarantee。

### 5.4 运行时间

每个模型必须记录：

```text
hardware, software environment, batch/agent count,
warm-up count, CUDA synchronization boundary,
preprocessing time, prediction time, sampling time, total time
```

如果某模型无法可靠拆分 prediction 和 sampling，应报告明确的 coupled timing，不得伪造子阶段数字。

---

## 6. Tiny fixture 和验收门槛

在任何 full training 前，所有模型和 evaluator 都必须通过同一组小测试：

1. `8 observed → 12 future` shape test；
2. coordinate round-trip test；
3. derivative test；
4. manifest/window ID equality test；
5. clean/noise/missing deterministic seed test；
6. tiny ADE/FDE reference test；
7. prediction output shape and finite-value test；
8. deterministic model `K=1` / probabilistic model `K=20` contract test。

### Group-level acceptance gates

```text
PROTOCOL_LOCKED
COMMON_LOADER_OK
COMMON_EVALUATOR_OK
MODEL_SMOKE_OK model=<name>
PILOT_OK model=<name> fold=<name>
FORMAL_OUTPUTS_OK model=<name> folds=5 seeds=3
```

没有通过 `COMMON_LOADER_OK` 和 `COMMON_EVALUATOR_OK`，不开始六个模型的正式训练。

---

## 7. 代码独立实现与引用规则

### 允许

- 阅读原论文中的数学公式、伪代码和模型图；
- 阅读官方仓库来确认 tensor shape、超参数、数据预处理细节；
- 用小例子与官方实现做 numerical sanity check；
- 在文档中引用论文和 repository；
- 复用极少量、无法合理重写的通用代码，并逐行标注来源和许可证。

### 不允许

- 直接 import 官方模型类作为最终模型；
- 复制官方 repository 的核心网络、loss、trainer 或 evaluator；
- 对官方代码只改路径/变量名后声称独立实现；
- 用官方 pretrained weights 产生最终比较结果；
- 把公开 repository 的大段代码放进 owner package 而不作 attribution。

### 每个 owner 必须提交

```text
paper_reading_notes.md
source_inventory.md
independent_design.md
implementation_provenance.md
reused_fragments.csv
own_unit_tests/
```

`reused_fragments.csv` 至少包含：source URL、source commit/version、file/line、复制或改写内容、许可证、最终代码位置和理由。若没有复用代码，也提交空表并写明“no copied code”。

### Trajectron++ 当前状态

Trajectron++ 的正式交付使用独立 `causal_v2` 实现，并已通过五折三种子
实验审计及解包复现。公开交接只包含可验证的小型结果快照与图。
上游 checkout 和早期兼容性运行保存在本地参考目录。其他模型通过同一
manifest、扰动和 evaluator 口径核对后再进入组内比较。

[Trajectron++ 组稿贡献](submission/contributions/trajectronpp/README.md) ·
[复现入口](li_junhao_trajectronpp/README.md)
