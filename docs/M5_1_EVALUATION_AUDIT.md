# M5.1 评估审计（Evaluation Audit）

最后更新：2026-11-28 ｜ 基线：`main` = `4cabf59` → 开发分支 `codex/m5.1-evaluation-audit`
范围：只做 **M5.1 = Evaluation Audit & Protocol Enhancement**，不新增模型、不改 B0–B7 的研究定义、
不覆盖任何历史数据/权重/结果。所有结论都可由"证据"列定位到代码或测试。

> 术语：**Confirmed Bug** = 代码+测试可证明的缺陷；**Potential Risk** = 有迹象但本仓库无法最终确认；
> **Protocol Limitation** = 设计层面的使用限制，不是代码错误；**Research Hypothesis** = 待实验回答。

## 1. 版本与权责审计

| 项 | 结论 | 证据 |
| --- | --- | --- |
| 本地入口版本 | `main` `4cabf59`（含 M5/M6 结果文档），工作区干净 | `git log --oneline`、`git status -sb` |
| 服务器 M5 实现 | 报告引用 `5ec6cbf…`（`codex/m5-source-specific`），本仓库 `git cat-file` → **not-in-this-repo** | `docs/M5_ROBUSTNESS_RESULTS_2026-11-25.md` 第 3、130 行 |
| 服务器 M6 实现 | 报告引用 `8ff31f8…`（`codex/m6-downstream`），同样不在本仓库；`scripts/run_downstream_de.py` 不存在 | `docs/M6_DOWNSTREAM_RESULTS_2026-11-26.md` 第 8、98 行；`ls scripts/` |
| 模型语义差异 | 服务器 M5/M6 使用**来源特异 + 残差**版 B4–B7；本仓库 main 让两个通道共享同一融合向量（B4 门控输出 / B5 拼接投影） | `models/__init__.py::_fuse`/`encode`；M5 报告第 122 行 |
| 结论 | 两边 B4–B7 指标**不可互换引用**；M5/M6 数值只能作为"当前表示"的初评，复现待实现入库 | §7 |
| 本次改动范围 | 数据/协议/审计/基线/工具链/文档；**未修改** B0–B7 数学定义、`CHANNEL_SETS`、损失、融合公式、`encode()` 输出宽度 | `git diff --stat`；`tests/test_models.py`、`tests/test_week3_models.py` 全绿 |

## 2. 问题清单

| 编号 | 类别 | 严重度 | 一句话 | 状态 |
| --- | --- | --- | --- | --- |
| B-01 | Confirmed Bug | P0 | 三个 split 的负样本各自独立采样，只排除已知正边 → 跨 split 重复 | Fixed |
| B-02 | Confirmed Bug | P0 | `validate()` 不检查负样本跨 split 互斥、self-loop、重复正边、`encode ⊆ train`、权重长度 | Fixed |
| B-03 | Confirmed Bug | P0 | 缺边只删 `encode_edges`，训练正例监督不变，却被当作"训练关系缺失"报告 | Fixed |
| B-04 | Confirmed Bug | P0 | 跨视图遮蔽在 `pipeline.py` 与 `perturbation.py` 各有一份实现，无一致性保证 | Fixed |
| B-05 | Confirmed Bug | P0 | 训练入口无数据审计、无 `audit.json`、无完成标志，无法安全 resume | Fixed |
| B-06 | Protocol Limitation | P1 | 只有均匀随机负样本；无困难负样本、无非神经网络基线 | 已补策略与基线 |
| B-07 | Protocol Limitation | P1 | 历史 M5/M6 已多次查看 test split，不能再宣称"从未查看过的独立测试集" | 文档口径（§6） |
| B-08 | Potential Risk | P2 | `optim.negatives_ratio` 只被校验、训练期不生效（负样本构建期固定） | 文档化（未改语义） |
| B-09 | Potential Risk | P2 | `gene_text` 特征来源无法从仓库独立核验（真实文件不在库中） | A05 + 待服务器确认 |
| B-10 | Protocol Limitation | P2 | M5 主曲线 26/40 单元最优 epoch 落在 200 上限，预算未按验证集锁定 | 待服务器按 D-018 统一预算 |
| B-11 | Pending Server Verification | P2 | 服务器 checkpoint、`data/processed/real_v1`、`runs/*` 均不入库 | 待服务器同步 |

### B-01 负样本跨 split 重复（P0，Fixed）

- 位置：`src/cofi_gene/pipeline.py::_build_channel_splits`（旧代码 173–177 行）。
- 旧实现：每个 split 只把"已知正边"放进 forbidden，**没有排除其他 split 已抽到的负边**：

```python
known = set(split.train) | set(split.val) | set(split.test)
for split_name in ("train", "val", "test"):
    seed = config.negative_seed + 1000 * channel_index + split_index
    negatives[split_name] = sample_negative_pairs(num_nodes, count, known, seed=seed)
```

  于是同一基因对可以是 train 负样本同时是 test 负样本。训练方检查记录：通道 A 12 对、通道 B 354 对
  test 负样本出现在 train 负样本中（该数字来自训练方检查，不是本仓库复现结果，见 M6 报告第 83 行）。
- 影响：test 负样本与训练负样本重叠会**低估**泛化误差，并让"缺边鲁棒性"曲线混入负样本记忆效应。
- 修复：新增 `negative_sampling.sample_split_negatives()`，按 `train → val → test` 顺序采样并**累积** `used`，
  后一个 split 的 forbidden = 已知正边 ∪ 之前所有 split 的负边；返回 `NegativePartition`
  （每 split 的 requested/sampled/attempts/matched/fallback/degree 统计 + `cross_split_overlap`）。
- 迁移兼容：保留原训练负样本的种子派生方式与 forbidden 语义，因此 train 负样本不变；只让 val/test 避开已用负边。
  历史 checkpoint 仍可用于诊断，但正式结论需重训（训练指南 §6）。
- 回归测试：`tests/test_m5_1_negative_splits.py`（13 例）＋ `tests/test_leakage.py::test_processed_pipeline_output_is_leakage_free`。

### B-02 `validate()` 覆盖不足（P0，Fixed）

- 位置：`src/cofi_gene/dataset.py`（旧 `validate()` 逐个 raise）。
- 旧实现只覆盖 4 项：正边互斥、`encode ∩ (val|test)`、负边 vs 已知正边、负边 split 内重复。
  缺失：**负边跨 split 互斥**、self-loop、反向重复正边、`encode ⊆ train`、权重长度、特征有限性。
- 修复：拆成 `inspect()`（永不抛错，返回 `(checks, report, failures)`）+ `validate()`（首个 FAIL 抛错，保留旧报错文案）
  + `report()`（计数）。检查 ID 见 §3。
- 回归测试：`tests/test_leakage.py` 的 6 个新增反例用例 + `tests/test_m5_1_audit.py`。

### B-03 删边未同步训练监督（P0，Fixed）

- 位置：`src/cofi_gene/perturbation.py::thin_channel`（旧 M5 唯一模式）。
- 旧函数只替换 `encode_edges/encode_weights`，`train` 原样返回。于是 10/30/50% 删边衡量的是"邻接信息减少"，
  而训练正例仍在教模型这些边存在——不能描述为"训练关系缺失"。
- 修复：新增 `remove_training_edges(..., supervision=)` 与 `apply_edge_removal(mode=...)`：
  - `adjacency`（M5 Mode 1，默认）：与旧行为逐位一致；
  - `supervision_aware`（M5.1 Mode 2）：同一次嵌套抽样同时删除 `encode_edges` 与 `train` 正例（权重同步），
    val/test 正例与负样本不动，记录 `num_train_positives_removed`。
  `apply_data_perturbations` 新增 `data.edge_removal_mode`、`data.drop_channels`。
- 回归测试：`tests/test_m5_1_removal_modes.py`（11 例），含 0% 严格 no-op、嵌套性、双模型同删边集合。

### B-04 遮蔽实现分裂（P0，Fixed）

- 位置：`pipeline.py::_build_channel_splits`（`mask_other_view` 分支）与 `perturbation.py::mask_cross_view_positives`。
- 证据：两处各写一遍"从另一视图删除 held-out 边"的集合运算；`pipeline` 版本只删 `encode`，
  `perturbation` 版本还删 `train`，口径不同且无测试保证一致。
- 修复：新增 `src/cofi_gene/protocols.py` 作为唯一实现；`mask_cross_view_positives()` 变成薄封装并委托
  `apply_visibility_protocol(dataset, "dual_view_strict_holdout")`；`pipeline` 的 `mask_other_view` 构建期标志保留，
  但训练期以协议为准（`data.mask_other_view` 与 `data.visibility_protocol` 冲突时报错）。
- 一致性证据：`tests/test_m5_1_protocols.py::test_mask_cross_view_positives_uses_the_same_implementation`
  对每个通道逐数组比较两个入口。

### B-05 训练入口缺少预检与完成标志（P0，Fixed）

- 修复：`train_link_prediction(..., audit_mode="strict"|"warn"|"off", audit_roundtrip=False)`：
  - `strict`（默认）在任何 FAIL 时终止，不进入训练循环；`warn` 记录后继续；`off` 跳过。
  - 记录 `metrics.json["audit"]`（状态、failed/warning ID、可见性摘要）并写 `run_dir/audit.json`。
  - 新增 `metrics.json["status"]="complete"` 与 `finished_at`（`--resume` 依赖）。
  - 新增 `metrics.json["protocol"]`（`visibility_protocol`、`edge_removal_mode`、`drop_channels`、
    `edge_drop_fraction`、`negative_strategy`、`changed`）。
- 回归测试：`tests/test_m5_1_train_gate.py`（7 例）。

### B-06 只有均匀随机负样本（P1，已补策略与基线）

- 修复：`negative_sampling` 新增 `degree_matched`、`two_hop`（含 fallback 与统计）；新增 `baselines.py`
  （PA / CN / Adamic–Adar / Resource Allocation / 特征余弦）与 `scripts/run_baselines.py`。
  所有困难策略与基线只用**协议可见的训练图**（`encode_edges`），并复用同一评估候选集。
- 回归测试：`tests/test_m5_1_negative_splits.py`、`tests/test_m5_1_baselines.py`。

### B-07 历史 test split 已被使用（P1，口径约束）

历史 M5（40 单元）与 M6（B0–B7 × 3 seed）的 test 指标已被查看并用于研究判断。因此：

- 新的 Stage 1–4 结果只能称为**开发诊断集**；
- 若要报告"最终确认"，必须在 Stage 5 前锁定协议，并优先使用未参与前期选择的评估集；
- 报告不得再写"该测试集从未被查看"。

### B-08 `optim.negatives_ratio` 不生效（P2，文档化）

`train.py` 不读取 `optim.negatives_ratio`；负样本在构建期固定（D-015 的设计选择），改该键只影响校验。
M5.1 保留此行为（不引入训练期重采样，避免改变既有语义），并在 `DATA_CONTRACT.md` 与训练指南写明：
**要改变负样本比例请重建数据集**（`PipelineConfig.negatives_ratio`）。

## 3. 校验项与检查 ID

| ID | 检查 | 失败后果 |
| --- | --- | --- |
| `C01-node-features` | 节点/特征数一致、`X` 全有限 | 训练中止 |
| `C02[ch].{train,val,test,encode_edges}` | 形状 `[2,E]`、整数、范围、无 self-loop、无重复/反向重复 | 训练中止 |
| `C03[ch].positive-disjoint` | 正边 train/val/test 两两互斥 | 训练中止 |
| `C04[ch].heldout-absent` | `encode ∩ (val∪test) = ∅` | 训练中止 |
| `C05[ch].encode-subset` | `encode ⊆ train`（允许遮蔽/删边收缩，不允许凭空多边） | 训练中止 |
| `C06[ch].*_weights` | 权重长度与边数一致 | 训练中止 |
| `C07[ch].neg_*` | 负边形状/范围/无 self-loop/无重复 | 训练中止 |
| `C08[ch].neg_*-positive` | 负边不使用任何已知正边（含隐藏正边） | 训练中止 |
| `C09[ch].neg-disjoint` | 负边 train/val/test 两两互斥（**M5.1 修复 B-01**） | 训练中止 |
| `C10-target-splits` | 下游目标分组互斥 | 训练中止 |
| `A01-protocol` | 协议名与删边模式合法 | 训练中止 |
| `A02-strict-holdout` | 严格遮蔽后可训练、残留可见度 = 0 | 训练中止 |
| `A04-negative-protocol` | 数据集记录负采样协议且声明互斥 | FAIL（旧数据集） |
| `A05-feature-provenance` | `meta` 有 `feature_reducer`/`provenance` | WARN（不可核验） |
| `A06-dataset-hash` | `dataset.npz` 存在并记录 sha256 | FAIL |
| `A07-roundtrip` | save→load 后划分/数组逐位不变 | FAIL |

WARN 不终止训练，但会出现在 `metrics.json["audit"]["warning_ids"]` 与阶段汇总中。

## 4. 新增实验协议（独立命名，不与 Bug Fix 混淆）

| 编号 | 名称 | 允许的信息 | 不允许的信息 | 与旧实验的差别 |
| --- | --- | --- | --- | --- |
| E-01 | `cross_view_transfer`（Protocol A） | 本通道训练边 + 另一来源中**已存在**的同一基因对证据 | 本通道 val/test 正边进入编码/监督 | 与旧默认一致，但现在会**报告**跨视图可见对（A/B 各一列） |
| E-02 | `dual_view_strict_holdout`（Protocol B） | 两通道训练边中、不在任一通道 held-out 并集里的部分 | 任一通道的 val/test 正边进入任一编码图或训练正例 | 旧 `mask_other_view` 只影响 `encode`；现在同时作用于 `train`，且残留可见度必须为 0 |
| E-03 | `degree_matched` 负样本 | 训练图上正例端点的度数分布 | 用含 val/test 边的图计算度数 | 新协议；与 `uniform_random` 并存，用于诊断"仅凭度数"捷径 |
| E-04 | `two_hop` 负样本 | 训练图两跳邻居 | 全图枚举 / `N×N` 矩阵 | 新协议（次优先级），二跳非边 |
| E-05 | `supervision_aware` 删边 | 删边比例与 seed（与 M5 相同的嵌套抽样） | 被删边继续作为训练正例或重新成为负例 | 旧 M5 只删邻接（现名 `adjacency`），本模式才对应"训练关系缺失" |
| E-06 | 非神经网络基线 | 训练图拓扑 / 共享特征 / 同一评估候选集 | 任何 checkpoint、测试边、完整图 | 新对照：PA、CN、Adamic–Adar、Resource Allocation、特征余弦 |

命名约束：`metrics.json["protocol"]` 与 run 名（`s{stage}_{model}_{xfer|strict}_drop{NN}_{adj|sup}_seed{N}`）
都显式编码协议，读者不必猜某个数字用了哪种遮蔽/删边口径。

## 5. 数据版本管理

| 项 | 规则 |
| --- | --- |
| 历史数据 | `data/processed/real_v1` **保留原样**，不覆盖；旧构建缺少 `meta.negative_sampling`，审计会给出 `A04` FAIL + `C09` 失败（如确有重叠） |
| 修复版数据 | 建议目录 `data/processed/real_v1_1`（同一 `PipelineConfig`，仅采样逻辑修复）；同 seed 下 train 负样本不变，val/test 会变 |
| 困难负样本版 | 建议 `data/processed/real_v1_1_deg`（`negatives_strategy: degree_matched`）与 `real_v1_1_hop`（`two_hop`），仅供评估/对照 |
| manifest 记录 | `meta.split`（含 `negatives_strategy` / `negatives_fallback` / `negative_degree_buckets` / `negative_split_exclusion`）、`meta.negative_sampling`（每通道策略、seed、fallback 率、`cross_split_overlap`）、`meta.features`（reducer 与说明）；`manifest.json` 仍记录原始文件 sha256、`git_commit`、命令与完整构建报告 |
| 哈希 | `scripts/audit_dataset.py` 的 `A06` 输出 `dataset.npz` sha256；训练 run 的 `metrics.json.dataset.dataset_sha256` 应与之相同 |

## 6. 复现与测试证据（本地，CPU）

| 命令 | 结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **300 passed**（218 → 300，M5.1 新增 82 例；无 skipped） |
| `python scripts/audit_dataset.py --dataset <processed> --out audit.json` | `status=PASS`，44 项检查，输出每通道 val/test 与跨视图可见度 |
| `python scripts/run_baselines.py --dataset <processed>` | 5 个基线 × 2 通道的 AP/AUROC + 阳性率基线 |
| `python scripts/run_experiments.py --stage 0/1/3/4/5 --dry-run` | 计划可展开；Stage 1/3/4 在合成数据上各跑通 1 格 |
| `python -m cofi_gene.train --config configs/b3.yaml --dataset <processed> --dry-run` | 打印 config + dataset 摘要 + 审计报告 |

## 7. 未解决 / 待服务器确认

1. **实现入库**：`codex/m5-source-specific`、`codex/m6-downstream` 未推到 GitHub；本仓库无法重跑 M5/M6。
   在补入之前，本仓库的 B4–B7 语义（共享融合向量）与服务器结果（来源特异）不一致，禁止混用。
2. **真实数据审计**：本仓库没有 `real_v1`，无法独立复现 12/354 的重复计数；修复后的 `real_v1_1` 必须由服务器构建并
   用 `scripts/audit_dataset.py` 出 PASS 报告后才能训练。
3. **特征溯源**：`gene_text` 原始文件与 PCA 拟合记录不在库中；`A05` 只能确认 `meta` 声明，不能确认外部文件哈希。
4. **预算**：D-018 仍未锁定；M5 有 26/40 单元顶到 200 epoch，任何"最优 epoch 在上限"的 run 都要在报告中标注。
5. **测试集语义**：见 B-07，历史 test split 不能作为"从未查看"的独立确认集。


