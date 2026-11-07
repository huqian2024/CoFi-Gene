# 项目状态（每次合并 PR 后更新）

最后更新：2026-11-07（**Week 3 / M3 代码完成**：B4 Gated Fusion、B5 Cross-Graph Attention 已实现并通过测试与冒烟训练；仍未接触真实数据与 GPU）
阶段：**M3 代码完成（合成数据验证）→ B6/B7（Feedback）尚未开始**
详细报告：`docs/WEEK3_IMPLEMENTATION_REPORT.md`

## Week 3 交付（2026-11-07）

| 项 | 值 |
| --- | --- |
| 分支 | `feat/week2-data-pipeline`（沿用，未合入 `main`，未重命名，未推送） |
| 新增模型 | **B4 Feature-wise Gated Fusion**（单 `Linear`+Sigmoid，单融合向量）、**B5 对称双向 Cross-Graph Attention**（4 头 × 32 维） |
| 新增文件 | `models/attention.py`、`configs/b4.yaml`、`configs/b5.yaml`、`tests/test_week3_models.py`、`docs/WEEK3_IMPLEMENTATION_REPORT.md` |
| 修改文件 | `models/__init__.py`、`models/fusion.py`（+`GatedFusion`）、`train.py`（`--skip-test`）、`config.py`（`attention_heads`/`attention_dropout`/`attention_residual`）、`tests/test_{models,config,train}.py`、`.github/workflows/ci.yml` |
| 测试 | `python -m pytest -q` → **147 passed**（Week 2 为 99，本次 +48） |
| 冒烟训练 | B3/B4/B5 在 `data/processed/synthetic_v1`（60 基因/16 维）上跑通，**用 `--skip-test` 不读 test split**；产物在 `runs/week3_smoke/` |
| 可复现 | 同 seed 下 B3/B4/B5 的 loss 与 val 指标完全一致（参数化确定性用例） |
| 参数统计（feature_dim=16, hidden=128） | B0 2,176 ｜ B1/B2 18,688 ｜ **B3 = B4 = 70,272（参数对齐）** ｜ B5 169,344 |
| 尚未做 | B6/B7、缺边鲁棒性、错位对照、下游 DE MLP、3 seed、真实数据、GPU |

### Week 3 明确未做

- **未实现任何 Feedback**（B6/B7 仍抛错），遵守"Feedback 不是本周目标"。
- 未改数据管线 / 划分 / 负采样 / 泄漏防线 / `metrics.py` / B0–B3 数学定义。
- 未改 `docs/DECISIONS.md` 既有条目（只在末尾**新增** D-019～D-021 待确认项）。
- 未用真实 STRING/PIE 数据，未在任何 GPU 上训练，未 commit、未 push、未合并。

## Week 2 最终交付（2026-11-01）

| 项 | 值 |
| --- | --- |
| 分支 | `feat/week2-data-pipeline`（**未合入 main**；main 停留在 M0 的 `dfa496e`） |
| 功能提交 | `59f0dd9` feat(week2): 数据管线 + 防泄漏防线 + B0–B3 上游训练 |
| 文档提交 | `21d389f` docs(status): 记录 CI run 41283635590 两个 job 均通过 |
| CI | run 41283635590（`data-and-splits` ✅、`torch-cpu` ✅，后者含完整训练命令端到端检查） |
| 测试 | `python -m pytest -q` → **99 passed**（CPU，torch 2.15.0+cpu，pyarrow 26.0.0） |
| 确定位 | 同 seed 两次训练的 loss 与 val/test 指标完全一致 |
| 合成数据规模（示例） | 60 基因 / 16 维特征；A=117 边、B=106 边、重合 52（jaccard 0.30） |
| 队友交接 | 见 `docs/TEAM_HANDOFF.md` |

### 四层状态区分（重要，勿混淆）

| 层次 | 内容 |
| --- | --- |
| ① **代码已实现** | STRING 双通道读取、ID 对齐、图构建、边/目标划分、负采样、泄漏防线、特征压缩、PIE 标签、数据集容器与 manifest、B0–B3 训练/评估、两个数据 CLI、配置体系、99 个测试、两个 CI job |
| ② **合成数据已验证** | 上述全部在 `cofi_gene.synthetic` 生成的假 STRING/PIE 上跑通；B0–B3 各有 val/test 指标，B1 val AUPRC 明显高于 1:1 负样本的 0.5 基线；训练可复现 |
| ③ **真实数据未验证** | 未下载、未核验任何 STRING / PIE 真实文件；`docs/DATA_CONTRACT.md` 第 3 节表格仍为空；基因交集、边数、覆盖率全部未知 |
| ④ **GPU 未验证** | 本机只有 RTX 4060 Laptop 且 torch 为 CPU 版；RTX 5090 上的 torch/CUDA（sm_120）、显存与真实规模训练**均未验证** |

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
| 单测总体 | **147 passed**（3.3 s，CPU） | `.venv/bin/python -m pytest -q` |
| STRING 双通道解析（分列、阈值、去重、自环、未知蛋白、combined-score 不当证据） | 已实现，合成数据验证 | `cofi_gene/string_io.py`、`graph_build.py`、`tests/test_string_io.py` |
| 基因 ID 对齐（符号归一化、多蛋白冲突计数） | 已实现，单测覆盖 | `cofi_gene/gene_ids.py`、`tests/test_gene_ids.py` |
| 边划分 / 分组划分 / 负样本 | 已实现，含确定性 | `cofi_gene/splits.py`、`negative_sampling.py`、`tests/test_splits.py` |
| 防泄漏防线 | 已实现并在构建期强制 | `CoFiDataset.validate()`、`tests/test_leakage.py`（7 个对抗用例） |
| 特征 3072→128 | 已实现（固定符号 SVD/PCA） | `cofi_gene/features.py`、`tests/test_features.py` |
| PIE DE 读取与标签规则 | 已实现（pyarrow） | `cofi_gene/pie_io.py`、`tests/test_pie_io.py` |
| 数据构建 CLI + manifest | 已实现（sha256/git/版本/命令/报告） | `scripts/build_dataset.py`、`tests/test_scripts.py` |
| B0–B3 训练/评估 | 已实现，合成数据端到端跑通 | `cofi_gene/train.py`、`models/`、`tests/test_train.py`（4 模型 × 真实前向反向） |
| 训练确定性 | 已确认（同 seed 两次 loss 与指标完全一致） | `tests/test_train.py::test_training_is_deterministic_for_a_fixed_seed` |
| B4 Gated Fusion | ✅ 已实现，合成数据跑通 | `models/fusion.py::GatedFusion`、`configs/b4.yaml`、`tests/test_week3_models.py` |
| B5 Cross-Graph Attention | ✅ 已实现，合成数据跑通 | `models/attention.py`、`configs/b5.yaml`、`tests/test_week3_models.py` |
| B6–B7（Feedback） | **未实现**（registry 主动抛 `NotImplementedError`） | `cofi_gene/models/__init__.py` |
| 下游 DE MLP | **未实现**（M6） | — |
| 真实 STRING / PIE 数据 | **未下载、未核验** | `docs/DATA_CONTRACT.md` 第 3 节仍为空 |
| 服务器 torch/CUDA（5090） | **未验证** | 需队友跑 `python scripts/check_environment.py --strict` |

## 正在执行（Week 3 已交付，下一步起点）

1. **B4/B5 已实现**：`configs/b4.yaml`、`configs/b5.yaml` 可用；B6/B7 仍拒绝。
2. **M1 收尾（数据，需队友）**：确认 STRING 版本/物种/阈值（D-005）、gene_text 来源与压缩方式（D-012/D-017）、DE 标签规则（D-014），然后跑真实数据构建。
3. **M0 收尾（服务器，需队友）**：5090 上 `python scripts/check_environment.py --strict` 通过并留档。
4. **M3 收尾（需两人）**：锁定 D-018（训练预算）以及新增的 D-019/D-020/D-021（门控/注意力/监督形式的工程默认），
   然后在真实数据上跑 B0–B5 单 seed 冒烟 → 才允许跑多 seed 消融。
5. **M4（未启动）**：B6/B7 单轮 Graph Feedback（`AGENTS.md` 4.3、D-009：只调既有边权重，不补边）。

## Week 2 明确未做 / 不做的部分

- 未实现 **B4–B7**（注册表主动抛 `NotImplementedError`）；未实现下游 DE MLP。
- 未做缺边鲁棒性（10/30/50%）、Gene ID 错位对照、多种子汇总、三类图。
- 未下载/核验任何真实数据，未在任何 GPU 上训练过。
- 未合并到 `main`，未改动 `main`，未 force push。
- 按交付要求，**今日不进入 Week 3 / M3**。

## 下一阶段启动条件（缺一不可）

1. `docs/DECISIONS.md` 的 **P0 三项**（D-005、D-009、D-014）已锁定。
2. 队友在 5090 上跑通 `scripts/check_environment.py --strict` 并记录版本信息。
3. `configs/data.yaml` 填入真实路径后 `scripts/build_dataset.py` 成功产出 `manifest.json`（含基因数/边数/重合率/覆盖率）。
4. **P1 五项**（D-012/D-017/D-011/D-015/D-018）在开跑 B4/B5 之前锁定，否则消融不可比。

## 最近完成（Week 3 提交，2026-11-07）

- 新增 `models/attention.py`（`CrossGraphAttention`：**4 头 × 32 维**、每基因 2 个候选 token、来源特异 Q/K/V、
  双向对称、残差可配置且默认关闭）。
- `models/fusion.py` 新增 `GatedFusion`（**单层** `Linear(2d→d)` + Sigmoid，`H = g⊙H_A + (1-g)⊙H_B`，单融合向量）。
- `models/__init__.py`：`IMPLEMENTED_MODELS=(b0..b5)`、`PLANNED_MODELS=(b6,b7)`；B4/B5 使用 per-channel edge head；
  新增 `encode()`（唯一 `[N,128]` 下游嵌入）与 `fusion_representations()`（B5 的 `Z_A/Z_B`，仅分析用）。
- `train.py`：打分传 `channel=`；B5 超参接线（`attention_heads`/`attention_dropout`/`attention_residual`）；
  `metrics.json` 的 `model` 段新增 `fusion_kind` / `edge_heads`；新增 `evaluate_test` 参数与 `--skip-test` 开关。
- `config.py`：允许并严格校验可选键 `attention_heads` / `attention_dropout` / `attention_residual`（**不写入 `DEFAULTS`**，
  因此 B0–B3 的 resolved config 与 run 名 hash 完全不变，实测 `configs/b3.yaml` 仍是 `f2ebde49513f`）。
- 新增 `configs/b4.yaml`、`configs/b5.yaml`；CI 的 `torch-cpu` job 增加 B4/B5 训练命令检查。
- 验证命令与输出：
  - `.venv/bin/python -m pytest -q` → **147 passed**（新增 48 个用例）
  - `python -m cofi_gene.train --config configs/b4.yaml --dataset data/processed/synthetic_v1 --out-dir runs/week3_smoke --epochs 40 --run-name week3_smoke_b4 --seed 7 --skip-test` → `metrics.json`（`fusion_kind=gate`, `edge_heads=per_channel`, `test=null`）
  - 同上把 `b4` 换成 `b5` → `fusion_kind=attention`；换成 `b3` 作回归对照
  - 参数统计（feature_dim=16）：B0 2,176 ｜ B1/B2 18,688 ｜ B3 = B4 = 70,272 ｜ B5 169,344
- 详细说明、公式、张量形状、待确认事项见 **`docs/WEEK3_IMPLEMENTATION_REPORT.md`**。

## 最近完成（Week 2 提交）

- 新增数据管线：`string_io`、`gene_ids`、`graph_build`、`features`、`pie_io`、`splits`（扩展）、`negative_sampling`、`dataset`、`pipeline`、`manifest`。
- 新增上游训练：`config`（严格校验 + `base:` 继承）、`metrics`（自实现 AUPRC/AUROC，含 tie 处理）、`models/{gcn,fusion,heads}`、`train`（CLI + 运行记录 + checkpoint）。
- 新增脚本：`scripts/build_dataset.py`、`scripts/make_synthetic_dataset.py`；新增 `configs/`（`_base.yaml`、`b0–b3.yaml`、`data.yaml`、`data.synthetic.yaml`）。
- 新增测试 14 个模块（含 conftest）、99 个用例；CI 拆成 `data-and-splits` 与 `torch-cpu` 两个 job。
- 验证命令与输出：
  - `.venv/bin/python -m pytest -q` → 99 passed
  - `python scripts/make_synthetic_dataset.py --raw data/raw/ci_synthetic --processed data/processed/ci_synthetic` → 60 节点 / 16 维特征
  - `python -m cofi_gene.train --config configs/b3.yaml --dataset data/processed/ci_synthetic --out-dir runs/ci --epochs 4 --run-name ci_b3` → 生成 `metrics.json`
- CI（run 41283635590，commit `59f0dd9`，分支 `feat/week2-data-pipeline`）：`data-and-splits` ✅、`torch-cpu` ✅（后者包含完整训练命令的端到端检查）。

## 明确的限制（不得当成已完成）

1. 所有"跑通"均基于**合成数据**（`cofi_gene.synthetic`），不是真实 STRING/PIE，也不是任何生物学结论。
2. 训练仅在 **CPU** 上验证过；RTX 5090 的 torch/CUDA（sm_120）与显存行为**未验证**。
3. `configs/_base.yaml` 的 lr/epochs/early-stop 是**占位值**；合成小图上 loss 曾出现尖峰（2.4→128→收敛），说明超参仍需在 M3 固定。
4. 未做：**B6–B7（Feedback）**、缺边鲁棒性（10/30/50%）、Gene ID 错位对照、下游 DE MLP、跨种子汇总、三类图。
   （B4/B5 已在 Week 3 补齐；B0–B5 仍未在真实数据或 GPU 上跑过。）

## 维护规则

- 状态只写实际验证过的事情；"计划""实施中""已完成"严格区分，合成/真实数据必须标注。
- 完成代码任务必须记录：commit、测试命令与结果、数据版本、实验 run ID、下一步。
- 在新对话里先以此文件为准，不从上一次对话推断进度。
