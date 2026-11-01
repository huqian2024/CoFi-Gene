# 项目状态（每次合并 PR 后更新）

最后更新：2026-11-01（Week 1 收尾 + Week 2 开发完成：数据管线 / 划分 / 防泄漏 / B0–B3 端到端跑通）
阶段：**M2 代码完成（合成数据）→ 等真实数据与服务器验证**

## 已确认的事实

- 研究设计在根目录 `AGENTS.md`，范围固定为 B0–B7；顶部含 Agent 工程维护入口。
- 团队两人：coding/模型实现一人为主；另一人负责上平台训练并共同确认数据与生物学设置。
- 远程仓库 https://github.com/huqian2024/CoFi-Gene 公开，默认分支 `main`；开发在 `feat/week2-data-pipeline` 分支进行。
- 本机（开发机）：Python 3.12.2、numpy 2.5.4、pyarrow 26.0.0、**torch 2.15.0+cpu（仅 CPU）**；GPU 为 RTX 4060 Laptop，**不是 5090 训练机**。
- 开发机访问 GitHub / PyPI 需经 `127.0.0.1:7890` 代理（仓库 `.git/config` 已设 http(s).proxy；pip 需 `export HTTPS_PROXY=...`）。
- CI（`.github/workflows/ci.yml`）有两个 job：`data-and-splits`（快，numpy/pyarrow）与 `torch-cpu`（装 CPU 版 torch 并跑训练命令）。

## 当前验收

| 工作项 | 状态 | 验收证据 |
| --- | --- | --- |
| 原始研究范围 B0–B7 | 已定义 | `AGENTS.md` |
| 包结构 / 依赖分组 / CI | 已完成 | `pyproject.toml`（core=numpy+pyyaml；extras: data/train/dev）、两个 CI job |
| 单测总体 | **99 passed**（4 s，CPU） | `.venv/bin/python -m pytest -q` |
| STRING 双通道解析（分列、阈值、去重、自环、未知蛋白、combined-score 不当证据） | 已实现，合成数据验证 | `cofi_gene/string_io.py`、`graph_build.py`、`tests/test_string_io.py` |
| 基因 ID 对齐（符号归一化、多蛋白冲突计数） | 已实现，单测覆盖 | `cofi_gene/gene_ids.py`、`tests/test_gene_ids.py` |
| 边划分 / 分组划分 / 负样本 | 已实现，含确定性 | `cofi_gene/splits.py`、`negative_sampling.py`、`tests/test_splits.py` |
| 防泄漏防线 | 已实现并在构建期强制 | `CoFiDataset.validate()`、`tests/test_leakage.py`（7 个对抗用例） |
| 特征 3072→128 | 已实现（固定符号 SVD/PCA） | `cofi_gene/features.py`、`tests/test_features.py` |
| PIE DE 读取与标签规则 | 已实现（pyarrow） | `cofi_gene/pie_io.py`、`tests/test_pie_io.py` |
| 数据构建 CLI + manifest | 已实现（sha256/git/版本/命令/报告） | `scripts/build_dataset.py`、`tests/test_scripts.py` |
| B0–B3 训练/评估 | 已实现，合成数据端到端跑通 | `cofi_gene/train.py`、`models/`、`tests/test_train.py`（4 模型 × 真实前向反向） |
| 训练确定性 | 已确认（同 seed 两次 loss 与指标完全一致） | `tests/test_train.py::test_training_is_deterministic_for_a_fixed_seed` |
| B4–B7 | **未实现**（registry 主动抛 `NotImplementedError`） | `cofi_gene/models/__init__.py` |
| 下游 DE MLP | **未实现**（M6） | — |
| 真实 STRING / PIE 数据 | **未下载、未核验** | `docs/DATA_CONTRACT.md` 第 3 节仍为空 |
| 服务器 torch/CUDA（5090） | **未验证** | 需队友跑 `python scripts/check_environment.py --strict` |

## 正在执行（最多 1–2 项）

1. **M1 收尾（数据）**：确认 STRING 版本/物种/阈值（D-005）、gene_text 来源与压缩方式（D-006/D-017）、DE 标签规则（D-014），然后跑真实数据构建。
2. **M3 准备**：确定训练预算（D-018），实现 B4（Gate）/B5（source-aware Attention）。

## 阻塞项/需两人敲定

详见 `docs/DECISIONS.md` 的 D-005、D-009～D-018。最关键的三个：

- STRING 版本/species/通道阈值（D-005）与边权是否归一化（D-011）。
- `gene_text` 的压缩方式与拟合基因集（D-012/D-017）——一变就要重跑全部模型。
- DE 标签规则（D-014）与负样本比例（D-015）——决定指标基线。

## 最近完成（本次提交）

- 新增数据管线：`string_io`、`gene_ids`、`graph_build`、`features`、`pie_io`、`splits`（扩展）、`negative_sampling`、`dataset`、`pipeline`、`manifest`。
- 新增上游训练：`config`（严格校验 + `base:` 继承）、`metrics`（自实现 AUPRC/AUROC，含 tie 处理）、`models/{gcn,fusion,heads}`、`train`（CLI + 运行记录 + checkpoint）。
- 新增脚本：`scripts/build_dataset.py`、`scripts/make_synthetic_dataset.py`；新增 `configs/`（`_base.yaml`、`b0–b3.yaml`、`data.yaml`、`data.synthetic.yaml`）。
- 新增测试 14 个模块（含 conftest）、99 个用例；CI 拆成 `data-and-splits` 与 `torch-cpu` 两个 job。
- 验证命令与输出：
  - `.venv/bin/python -m pytest -q` → 99 passed
  - `python scripts/make_synthetic_dataset.py --raw data/raw/ci_synthetic --processed data/processed/ci_synthetic` → 60 节点 / 16 维特征
  - `python -m cofi_gene.train --config configs/b3.yaml --dataset data/processed/ci_synthetic --out-dir runs/ci --epochs 4 --run-name ci_b3` → 生成 `metrics.json`

## 明确的限制（不得当成已完成）

1. 所有"跑通"均基于**合成数据**（`cofi_gene.synthetic`），不是真实 STRING/PIE，也不是任何生物学结论。
2. 训练仅在 **CPU** 上验证过；RTX 5090 的 torch/CUDA（sm_120）与显存行为**未验证**。
3. `configs/_base.yaml` 的 lr/epochs/early-stop 是**占位值**；合成小图上 loss 曾出现尖峰（2.4→128→收敛），说明超参仍需在 M3 固定。
4. 未做：B4–B7、缺边鲁棒性（10/30/50%）、Gene ID 错位对照、下游 DE MLP、跨种子汇总、三类图。

## 维护规则

- 状态只写实际验证过的事情；"计划""实施中""已完成"严格区分，合成/真实数据必须标注。
- 完成代码任务必须记录：commit、测试命令与结果、数据版本、实验 run ID、下一步。
- 在新对话里先以此文件为准，不从上一次对话推断进度。
