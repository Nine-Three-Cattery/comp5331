# COMP5331 第 7 组 · 行人轨迹预测

本仓库用于协作完成六个 ETH/UCY 行人轨迹预测模型的独立实现、统一实验，以及小组英文报告和 Beamer 幻灯片。目前 Trajectron++ 已通过验收，其余五个模型待合入。

## 加入小组项目

`group7/project` 是小组协作和最终汇总的分支。运行下面的命令完成首次克隆。

```bash
git clone --single-branch --branch group7/project https://github.com/Nine-Three-Cattery/comp5331.git comp5331-group7
cd comp5331-group7
make check-docs
```

1. 在[模型登记表](submission/models.json)里按 `owner` 找到自己，记下模型 `id` 和代码目录 `code`。
2. 从 `group7/project` 创建 `model/<model_id>` 分支。已有 checkout 的组员可按[协作 SOP](CONTRIBUTING.md)切换到小组分支。
3. 按[协作 SOP](CONTRIBUTING.md)完成实现、实验验收和交付；如使用 coding agent，先让它阅读 [AGENTS.md](AGENTS.md)。
4. 推送自己的模型分支，开 PR 时将 **base 设为 `group7/project`**，compare 设为 `model/<model_id>`。组内审核后统一合入。

## 跑统一实验

1. 按 [SOP 的环境步骤](CONTRIBUTING.md#1-create-your-model-branch-and-set-up-the-benchmark)安装依赖，运行 `make benchmark PYTHON=python`，确认输出 `BENCHMARK_OK`。这一步准备固定版本的 ETH/UCY 数据，校验五折划分、扰动和指标。
2. 先完成 ETH、seed 0 的 pilot。用自己模型的训练与预测入口生成六个条件的预测，再用 `make pilot` 验收。提交代码前验证目标和邻居的输入都只依赖可见历史，并完成坐标往返测试。
3. pilot 通过后跑五折 × 三个种子（0、1、2）× 六个条件。每个 fold/seed 只在验证集选 checkpoint，同一个 checkpoint 用于六个测试条件。统一扰动目标行人的历史，邻居历史保持可见。
4. 正式预测导出后，依次运行 `make score` 和 `make audit`。完整命令及目录参数见[协作 SOP](CONTRIBUTING.md)，数据格式、指标和计时口径见[统一实验接口](benchmark/README.md)和[实验规范](GROUP7_COMMON_DATA_EVALUATION_SPEC.md)。

## 提交英文报告和 Beamer 幻灯片

在 `submission/contributions/<model_id>/` 下提交英文 `report.tex` 小节、`slides.tex` Beamer 页面和 `speaker_notes.md` 逐页讲稿，以及 `results.csv`、`provenance.json`、`benchmark_audit.json` 和必要的小型图表。可从[模板](submission/templates/)开始，也可参考已验收的 [Trajectron++](submission/contributions/trajectronpp/README.md)。沿用小组主题和指标宏，详细规则见[文档提交规范](submission/README.md)。

按 SOP 设置 `MODEL` 后，运行 `make preview MODEL="$MODEL"` 检查个人部分。小组编辑按[汇总清单](submission/editor/README.md)把各模型小节和页面编入同一份报告和幻灯片。

## 仓库目录

| 路径 | 用途 |
|---|---|
| `benchmark/` | 统一数据准备、评估入口，以及窗口、扰动和计时核对样例 |
| `models/<model_id>/` | 各模型的代码、配置、环境、测试和来源登记 |
| `li_junhao_trajectronpp/` | Trajectron++ 的实现和复现入口 |
| `submission/contributions/<model_id>/` | 各模型的英文报告小节、Beamer 页面、讲稿及验收证据 |
| `submission/editor/` | 小组比较、结论和最终交付清单 |
| `submission/build/` | 本地生成的报告和幻灯片 PDF |

## 编译小组草稿

在仓库根目录运行 `make check-docs` 检查贡献格式，再运行 `make docs`。PDF 编译需要 Python 3、pdfLaTeX、BibTeX 和 Beamer Metropolis。生成文件为 `submission/build/report.pdf` 和 `submission/build/slides.pdf`。`make final-docs` 要求六个模型都通过验收，并完成小组对比与总结。

## 提交前检查

每次提交前运行 `git diff --cached --name-status` 检查暂存文件。原始数据、课程原件、Canvas 导出、模型权重、完整预测、编译产物和完整复现包放在 Git 忽略的目录中。课程资料放 `local/`，复现包放 `artifacts/`。小组仓库提交报告和幻灯片所需的已验收小表和源图。完整复现包通过小组约定的位置共享，并在 `provenance.json` 记录文件名和 SHA-256。
