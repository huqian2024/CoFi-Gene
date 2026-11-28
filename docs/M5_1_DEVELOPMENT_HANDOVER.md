# M5.1 开发交接（给项目负责人本人）

分支：`codex/m5.1-evaluation-audit` → **已快进合并入 `main` 并推送（`c58b69c`）**；下文保留开发过程记录（当时基于 `4cabf59`）。
日期：2026-11-28 ｜ 环境：Python 3.12 / torch 2.15.0+cpu（本地只做 CPU 验证）

## 1. 修改文件清单与原因

| 文件 | 类型 | 原因 |
| --- | --- | --- |
| `src/cofi_gene/negative_sampling.py` | 重写 | B-01 跨 split 负样本重复；新增 `degree_matched`/`two_hop`、fallback 与统计 |
| `src/cofi_gene/dataset.py` | 扩展 | B-02 校验不足：`inspect()`/`report()`/`edge_pairs_checked()`，新增 C01–C10 |
| `src/cofi_gene/protocols.py` | 新增 | B-04 统一可见性协议（`cross_view_transfer` / `dual_view_strict_holdout`）+ 可见性报告 |
| `src/cofi_gene/perturbation.py` | 扩展 | B-03 新增 `supervision_aware` 删边、按通道删边；遮蔽委托 protocols |
| `src/cofi_gene/audit.py` | 新增 | B-05 结构化审计报告、`require_pass()`、A01–A07 检查、Markdown 渲染 |
| `src/cofi_gene/baselines.py` | 新增 | B-06 非神经网络基线（PA/CN/AA/RA/余弦），只用训练图与同一候选集 |
| `src/cofi_gene/evaluate.py` | 新增 | Stage 3 冻结 checkpoint 评估（按 checkpoint 自带 config 重建模型） |
| `src/cofi_gene/pipeline.py` | 修改 | 接入新负采样器；`PipelineConfig` 新增 3 个负采样字段并校验；meta 记录协议 |
| `src/cofi_gene/config.py` | 修改 | 允许并校验 `data.visibility_protocol` / `edge_removal_mode` / `drop_channels` |
| `src/cofi_gene/train.py` | 修改 | 训练前审计（strict/warn/off）、`audit.json`、`protocol` 块、`status=complete`、6 个新 CLI 开关 |
| `src/cofi_gene/graph_build.py` | 扩展 | `subset_edge_arrays()`：遮蔽与删边共用同一集合运算 |
| `scripts/audit_dataset.py` | 新增 | 数据审计 CLI（退出码 0/1/2） |
| `scripts/run_baselines.py` | 新增 | 基线 CLI（可带协议/删边），输出 JSON+MD |
| `scripts/run_experiments.py` | 新增 | Stage 0–5 矩阵：dry-run / resume / only-* / fail-fast / 冻结评估 |
| `scripts/build_dataset.py` | 修改 | 新增 `--negatives-strategy/--negatives-fallback/--negative-degree-buckets` |
| `configs/data.yaml` | 修改 | 写入新负采样键（默认 `uniform_random`，保持历史口径） |
| `.github/workflows/ci.yml` | 修改 | 新增 M5.1 审计/基线/Stage 4/Stage 3 冒烟 |
| `tests/test_leakage.py` | 修改 | 负样本改为互斥夹具；新增 6 个反例用例 |
| `tests/test_pipeline_synthetic.py` | 修改 | 6 节点陷阱图改用 `negatives_ratio=0.5`（互斥后 1:1 不可行） |
| `tests/test_m5_1_*.py`（7 个新文件） | 新增 | 负采样/审计/协议/删边/基线/runner/训练网关/脚本 CLI |
| `docs/M5_1_EVALUATION_AUDIT.md`、`docs/M5_1_SERVER_TRAINING_GUIDE.md`、本文件 | 新增 | 交付文档 |
| `docs/{STATUS,ROADMAP,DECISIONS,DATA_CONTRACT}.md`、`README.md` | 更新 | 同步状态、决策、数据合同与入口 |

**未改动**：B0–B7 的数学定义与注册表（`CHANNEL_SETS`、`fusion`、`attention`、`feedback`、`heads`、`gcn`）、
`metrics.py`、`splits.py` 的划分语义、`features.py`、历史结果文档的数值。

## 2. 已解决的问题

见 `docs/M5_1_EVALUATION_AUDIT.md` 的 B-01…B-06，每条都有位置、旧代码证据、修复方式与回归测试。

## 3. 新增能力

1. **负采样协议**：`uniform_random`（默认，兼容历史）、`degree_matched`、`two_hop`；跨 split 互斥；fallback 可配置且被统计。
2. **可见性协议**：Protocol A（跨证据迁移，报告可见比例）、Protocol B（严格遮蔽，残留可见度必须为 0）。
3. **删边模式**：`adjacency`（M5 旧口径）与 `supervision_aware`（同步删训练正例），支持按通道选择。
4. **审计工具**：`audit_dataset()` + `scripts/audit_dataset.py`；训练入口默认 strict 闸门。
5. **基线**：5 个非神经网络评分器 + `scripts/run_baselines.py`，与模型共用候选集。
6. **实验矩阵**：`scripts/run_experiments.py`（Stage 0–5，共 57 格；支持 resume/过滤/冻结评估）。
7. **可追溯性**：`metrics.json` 新增 `status`/`finished_at`/`protocol`/`audit`；run 名编码协议与删边模式。

## 4. 测试执行情况（本地真实结果）

| 命令 | 结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **300 passed**（含 torch 用例，无 skip） |
| `python scripts/audit_dataset.py --dataset /tmp/m51/processed --out ...` | `status=PASS`，44 项检查，退出码 0 |
| `python scripts/run_baselines.py --dataset /tmp/m51/processed` | 5 基线 × A/B 两通道输出 AP/AUROC |
| `python scripts/run_experiments.py --stage 0/1/3/4 --dataset /tmp/m51/processed ...` | Stage 0/1/3/4 各跑通；`--resume` 正确跳过；Stage 4 删除了 40 条编码边 + 40 条训练正例 |
| `python scripts/run_experiments.py --stage 0..5 --dry-run` | 1 / 8 / 4 / 8 / 12 / 24 格 |

## 5. 兼容性

- **数据**：旧 `real_v1` 仍可被 `CoFiDataset.load()` 读取，但 `validate()` 会因负样本跨 split 重叠而拒绝
  （这是修复的目的）；训练前审计会给出 `A04` WARN/`C09` FAIL。新数据请建 `real_v1_1`。
- **配置**：新增键都是可选的、**没有**进入 `DEFAULTS`，因此 B0–B7 的 resolved config / run 名 hash 不变
  （`tests/test_config.py` 全绿）。
- **checkpoint**：模型结构未改，旧 checkpoint 仍能加载；`state_dict` 键不变。
- **metrics.json**：新增 `status`/`finished_at`/`protocol`/`audit` 字段（旧的读取代码不受影响）；
  历史 `metrics.json` 没有这些字段，`--resume` 会视为未完成（符合预期）。
- **M6**：`encode()` 输出维度（128）与基因顺序未变，下游冻结 embedding 接口不受影响。

## 6. 已知风险 / 未解决

1. **服务器实现未入库**：`codex/m5-source-specific`、`codex/m6-downstream` 不在 GitHub；B4–B7 语义差异未消除，
   M5/M6 数值仍不可与本仓库代码互换引用。
2. **真实数据审计未做**：本仓库没有 `real_v1`，12/354 的重复计数来自训练方记录，未本地复现。
3. **特征溯源**：`A05` 只能确认 `meta` 声明，无法核验外部 `gene_text` 文件哈希。
4. **预算**：D-018 仍未锁定，M5 有 26/40 单元最优 epoch 在 200 上限。
5. **测试集语义**：历史 test split 已用于研究判断（B-07），Stage 1–4 只能算开发诊断。

## 7. 需要服务器负责人完成的事

按 `docs/M5_1_SERVER_TRAINING_GUIDE.md` 执行：同步分支 → 建 `real_v1_1` → 数据审计 PASS →
Stage 0/1/2/3/4/5 → 回传 §8 清单。**不要**用 `--audit warn` 绕过失败。

## 8. 对 M6 的影响

- 冻结 embedding 导出接口（`model.encode()` → `[N, 128]`）与基因顺序未变；
- 下游数据读取/PIE 目标分组代码未改；
- 但 **M6 的旧结果建立在有重叠负样本 + 旧 B4–B7 语义之上**，正式下游结论必须用 `real_v1_1` 重训的上游表示重跑。

## 9. 下一阶段建议

1. 先把服务器两个 `codex/*` 分支推回 GitHub（或补丁），使 B4–B7 语义一致；
2. 用 `real_v1_1` 跑 Stage 0–4，先回答"修复后双图是否仍不如单图"；
3. 若 Stage 2/3 显示双图在严格协议下仍无优势，如实报告，并把重点转向 M6 下游诊断（macro/micro 分歧、低阳性率目标）；
4. 锁定 D-018 预算后再跑 Stage 5 三种子。
