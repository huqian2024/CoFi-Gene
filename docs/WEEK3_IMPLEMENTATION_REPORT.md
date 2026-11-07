# Week 3 实施报告（B4 Feature-wise Gated Fusion / B5 对称双向 Cross-Graph Attention）

最后更新：2026-11-07
范围：`AGENTS.md` 固定范围 B0–B7 中的 **B4、B5**；**未实现 B6/B7（Graph Feedback）**。
本报告只记录**实际执行并验证过**的内容；未验证项会显式标注。

本轮依据的权威执行指令（Week 3 Master Execution Instruction）与本仓库既有计划的差异，见 §14：

- B4 门控 = **单个 `Linear(2d→d)` + Sigmoid**（无隐藏层），输出**一个**融合向量 `H`。
- B5 注意力 = **4 头 × 32 维**，对称双向，每个 Query 恰好 **2 个** Key/Value 候选。
- Week 3 只做**实现 + 测试 + 冒烟训练**，且冒烟训练**不得使用 test split**（本轮新增 `--skip-test`）。

---

## 1. Week 2 审计结论

审计方式：读代码 + 跑测试 + 跑训练命令，以**本地工作树**为准（远程 `main` 停在 M0，落后于本地分支）。

| 组件 | 结论 | 证据 |
| --- | --- | --- |
| STRING 双通道读取（只按单通道列，不用 combined score） | ✅ 已实现、已测试 | `string_io.py`、`tests/test_string_io.py` |
| 蛋白 ID → 基因符号对齐与冲突计数 | ✅ 已实现、已测试 | `gene_ids.py`、`tests/test_gene_ids.py` |
| 图 A / 图 B 构建、去重、自环、重合率 | ✅ 已实现、已测试 | `graph_build.py`、`tests/test_graph_build.py` |
| 边 train/val/test 划分、按扰动目标分组、缺边模拟 | ✅ 已实现、已测试 | `splits.py`、`tests/test_splits.py` |
| 负采样（避开全部已知正边、构建期固定） | ✅ 已实现、已测试 | `negative_sampling.py` |
| 防泄漏强制校验（作为不变量在加载/构建时抛错） | ✅ 已实现 | `CoFiDataset.validate()`、`tests/test_leakage.py`（7 个对抗用例） |
| 特征 3072→128（固定符号 SVD/PCA） | ✅ 已实现、已测试 | `features.py`、`tests/test_features.py` |
| PIE DE 读取与唯一标签规则 | ✅ 已实现、已测试 | `pie_io.py`、`tests/test_pie_io.py` |
| 数据集容器 + manifest / sha256 / git 记录 | ✅ 已实现 | `dataset.py`、`manifest.py`、`scripts/build_dataset.py` |
| B0 / B1 / B2 / B3 上游训练与评估 | ✅ 已实现、端到端跑通、同 seed 可复现 | `models/{gcn,fusion,heads}.py`、`train.py`、`tests/test_train.py` |
| B4–B7 | ❌ 注册表主动抛 `NotImplementedError` | 改动前 `models/__init__.py` |
| 真实 STRING/PIE 数据 | ❌ 未下载、未核验 | `docs/DATA_CONTRACT.md` 第 3 节仍为空 |
| GPU（RTX 5090） | ❌ 未验证 | 本机 `torch 2.15.0+cpu`、`torch.cuda.is_available() == False` |
| 下游 DE MLP / 缺边鲁棒性 / 错位对照 / 3 seed | ❌ 未开始（M5–M7） | — |

审计前基线：`python -m pytest -q` → **99 passed**。

### 1.1 Week 2 冻结策略遵守情况

不变（已核对）：数据源与版本、基因映射与顺序、两图构建、边权、特征预处理与 128 维表示、边划分、
负采样协议、B0–B3 配置与行为、既有实验产物（`runs/b3_seed42_f2ebde49513f/` 未被触碰）。

可验证证据：本轮改动后，`configs/b0..b3.yaml` 的 **resolved-config hash 与 Week 2 完全一致**
（`bd7bb75ec0f2` / `f3160ec62e2d` / `0ea7242eb7f9` / `f2ebde49513f`），因此 B0–B3 的 run 名不会变。
做法：Week 3 新增的可选配置键**故意不写入 `config.py: DEFAULTS`**。

### 1.2 审计中发现的 Week 2 观察项（已上报，按要求**未自动修复**）

| # | 观察 | 证据 | 影响 | 建议（等待批准） |
| --- | --- | --- | --- | --- |
| W2-1 | `optim.negatives_ratio` 被校验但**从未被使用**；负样本在数据构建期生成并写入数据集（D-015） | `config.py` 校验 + `train.py` 只读 `part.neg_*` | 改这个键不会有任何效果，容易误导 | 在 `config.py` 注释标注"仅记录/预留"，或在 train.py 显式 `raise` 若与数据集不一致 |
| W2-2 | `optim.eval_every` / `optim.early_stop_patience` / `runtime.log_every` 未做类型与范围校验 | `validate_config()` 只校验 lr/wd/epochs/negatives_ratio/grad_clip | 传 `eval_every: 0` 会被 `max(1, …)` 静默修正 | 追加三条范围校验 |
| W2-3 | 小合成图上 `lr=0.01` 会出现 loss 尖峰（本轮 B4 `6.3→29→0.28`、B5 `0.96→141→0.36`，均为有限值并已收敛） | `runs/week3_smoke/*/metrics.json` 的 `history` | 训练预算（D-018）仍是占位值，不能据此下科学结论 | 先锁 D-018，再谈模型差异；**不是代码 bug** |

> 以上三项都**不阻塞** B4/B5 实现，因此本轮继续推进，未改动任何数据/划分/训练协议语义。

---

## 2. 改动前的实际实现状态（一句话）

数据管线、划分、防泄漏、特征压缩、B0–B3 训练/评估已在**合成数据 + CPU** 上跑通并可复现；
B4–B7 由注册表主动拒绝；没有任何真实数据或 GPU 验证。

## 3. 本次创建的文件

| 文件 | 作用 |
| --- | --- |
| `src/cofi_gene/models/attention.py` | B5 `CrossGraphAttention`（4 头 × 32 维、来源特异 Q/K/V、双向对称、可选残差） |
| `configs/b4.yaml` | B4 配置（继承 `_base.yaml`，与 B3 完全同一 optim/data/runtime） |
| `configs/b5.yaml` | B5 配置（`attention_heads: 4`、`attention_residual: false`，其余继承） |
| `tests/test_week3_models.py` | Week 3 单元/集成测试 |
| `docs/WEEK3_IMPLEMENTATION_REPORT.md` | 本报告 |

## 4. 本次修改的文件

| 文件 | 改动 | 是否影响 B0–B3 |
| --- | --- | --- |
| `src/cofi_gene/models/fusion.py` | 新增 `GatedFusion`（单 `Linear(2d→d)` + Sigmoid，单输出 `H`） | 否（`ConcatFusion` 未改） |
| `src/cofi_gene/models/__init__.py` | `IMPLEMENTED_MODELS=(b0..b5)`、`PLANNED_MODELS=(b6,b7)`；B4/B5 使用 per-channel edge head；新增 `encode()`、`fusion_representations()`、`head_for()`；`score_edges(..., channel=)` | 否（B0–B3 仍走共享 `head`，行为与输出完全一致） |
| `src/cofi_gene/train.py` | 打分传 `channel=`；B5 超参接线；`metrics.json` 的 `model` 段增加 `fusion_kind`/`edge_heads`；新增 `evaluate_test` 参数与 `--skip-test` 开关（默认仍评测 test） | 否（默认路径逐行等价，仅多一次 `channel=` 传参，B0–B3 的 `channel_heads is None` → 同一 head） |
| `src/cofi_gene/config.py` | `_ALLOWED["model"]` 增加可选键 `attention_heads` / `attention_dropout` / `attention_residual` 并严格校验；**不写入 `DEFAULTS`** | 否（resolved config 不变，见 §1.1） |
| `tests/test_models.py` | 注册表用例的玩具维度由 `hidden_dim=6` 改为 `hidden_dim=8 / feature_dim=4`（6 无法被 4 头整除） | 否（断言集合不变，仅维度；B0–B3 断言同样通过） |
| `tests/test_config.py` | 覆盖 b0–b5 配置；`planned model` 断言改用 b6；新增可选键校验 | 否 |
| `tests/test_train.py` | 端到端/确定性用例扩展到 b4/b5；新增 `--skip-test` 两例 | 否（原用例全部保留通过） |
| `.github/workflows/ci.yml` | `torch-cpu` job 追加 B4/B5 训练命令与产物检查 | 否 |
| `AGENTS.md` / `README.md` / `docs/STATUS.md` / `docs/ROADMAP.md` / `docs/HANDOFF.md` / `docs/TEAM_HANDOFF.md` | 状态与命令同步（事实性更新，不改研究范围） | 否 |
| `docs/DECISIONS.md` | 仅在末尾追加 D-019/D-020/D-021 及本轮修订说明，未改既有条目 | 否 |

## 5. B4 架构与公式（Feature-wise Gated Fusion）

```
X, G_A -> GCN_A -> H_A [N,128]
X, G_B -> GCN_B -> H_B [N,128]
H_cat = [H_A ; H_B]               [N,256]
g     = sigmoid(Linear(H_cat))    [N,128]      # 逐基因、逐特征
H     = g * H_A + (1 - g) * H_B   [N,128]      # 单一融合输出
H --(通道 A 专属 head)--> A 图边 logits
H --(通道 B 专属 head)--> B 图边 logits
```

实现要点：

- 门控网络就是**一层** `nn.Linear(2d, d)` + `Sigmoid`，没有隐藏层、没有 ReLU、没有额外注意力。
- 门是**特征级**的：每个基因得到自己的 128 个门值（`GatedFusion.gate_weight()` 可导出检查），
  不是"每个图一个标量权重"。
- 零初始化下 `g = 0.5`，即初始等价于两图均值，不会偏向任一来源。
- 输出维度：`hidden_dim == fusion_dim == 128`，因此**不需要**再叠一层投影（避免出现"未训练的投影"，
  见 §13 的 L-2）。
- 与 B3 的关系：B3 的 `ConcatFusion` 是 `Linear(256→128)`，B4 的门控也是 `Linear(256→128)`，
  因此 **B3 与 B4 参数完全相同（均为 70,272，feature_dim=16）**，两者只差"固定拼接投影 vs 学习式门控"，
  这条对比没有容量混淆（`tests/test_week3_models.py::test_b4_is_parameter_matched_with_b3`）。

## 6. B5 架构与公式（对称双向 Cross-Graph Attention，4 头 × 32 维）

```
X, G_A -> GCN_A -> H_A [N,128]
X, G_B -> GCN_B -> H_B [N,128]
[H_A, H_B] -> 4 头双向跨图注意力 -> Z_A [N,128], Z_B [N,128]
[Z_A ; Z_B] [N,256] -> Linear -> H [N,128]
H --(通道 A 专属 head)--> A 图边 logits
H --(通道 B 专属 head)--> B 图边 logits
```

单基因 `i`、query 来源 `s ∈ {A,B}`、候选来源 `t ∈ {A,B}`、头 `m ∈ {1..4}`、`d_head = 32`：

```
Q_s^m = h_s W_q^{s,m},  K_t^m = h_t W_k^{t,m},  V_t^m = h_t W_v^{t,m}
score_{s,t}^m = (Q_s^m · K_t^m) / sqrt(32)
alpha_{s,·}^m = softmax_{t∈{A,B}} score_{s,·}^m          ← 恰好两个候选
Z_s = concat_m ( sum_t alpha_{s,t}^m V_t^m )              ← 4 × 32 = 128
```

- **候选数恒为 2**（自身图 + 另一图）：若只有一个候选，softmax 恒为 1、梯度为 0，这是计划明确禁止的设计。
  测试断言 `weights.shape[-1] == 2`、两侧权重严格落在 (0,1)、每行和为 1。
- **对称双向**：A 与 B 都作为 Query 来源（2×2 的 score 矩阵），信息 A→B 与 B→A 都流动。
- **来源特异投影**：`W_q^{s}`、`W_k^{s}`、`W_v^{s}` 按来源分开。
- 每个基因只在"两个证据视图"之间做注意力；邻域信息已由 GCN 编码，**不是**全基因 Transformer。
- 最终 `[Z_A;Z_B]` 经一个 `Linear(256→128)` 投影，得到统一 `[N,128]`（AGENTS.md 4.2 的"拼接 + 线性投影"）。
- 残差（`Z_s = h_s + attn`）实现为**可配置开关** `attention_residual`，默认 **False**（与本轮执行指令的公式一致）；
  `AGENTS.md` 4.2 曾提到残差，两者差异已在 §14 列为待确认项 D-020，未擅自当作既定结论。

## 7. 张量形状规格

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `X` | `[N, F]` | 节点特征（真实数据 F=128；本轮合成数据 F=16） |
| `adjacencies[ch]` | sparse `[N, N]` | `D^-1/2 (A+I) D^-1/2`，**只用编码图（训练子图）**；val/test 正边不进入 |
| `H_A`, `H_B` | `[N, 128]` | 两个独立 GCN 编码器输出 |
| B4 `H_cat` | `[N, 256]` | `[H_A ; H_B]` |
| B4 `g` | `[N, 128]` | `sigmoid(Linear(...))`，逐基因逐特征，∈ (0,1) |
| B4 `H` | `[N, 128]` | `g*H_A + (1-g)*H_B`，两个通道 key 指向**同一个**向量 |
| B5 `Q, K, V` | `[N, 2, 4, 32]` | `[节点, 来源, 头, d_head]` |
| B5 `scores / weights` | `[N, 2, 4, 2]` | `[节点, query 来源, 头, 候选来源]`，最后一维 softmax 和为 1 |
| B5 `Z_A`, `Z_B` | `[N, 128]` | 每头 32 维，4 头拼接 |
| B5 `H` | `[N, 128]` | `Linear([Z_A ; Z_B])` |
| `encode(x, adj)` | `[N, 128]` | 唯一的下游冻结嵌入，所有 B0–B5 一致 |
| 边打分 | `[2, K] → [K]` | 每通道独立 head；K = 该通道正样本数或负样本数 |

## 8. 训练集成细节

- **共享管线**：B4/B5 复用 `cofi_gene.train.train_link_prediction`，不新增训练循环。
  优化器（Adam）、lr、weight decay、grad clip、epochs、eval_every、early stop、负样本、
  评估实现、seed 约定全部来自同一份 `_base.yaml`，与 B3 逐字段一致（测试断言 lr/epochs 相同）。
- **独立边预测头**（Part 7.1）：B4/B5 持有 `nn.ModuleDict` 形式的 per-channel head（`edge_heads=per_channel`），
  A/B 各自实例化、各自计算 BCE；B0–B3 保持 Week 2 的单一共享 head（`edge_heads=shared`），
  因此基线行为未被改动。默认 `edge_head: dot`（既有实现，未引入新解码器设计）。
- **监督对象**：两个通道都解码**同一个最终 128 维嵌入**。若改为分别解码 `Z_A`/`Z_B`，
  `[Z_A;Z_B]→128` 投影在训练中拿不到梯度，下游就会用到未训练的随机投影（§13 L-2）——这一点已由测试守护。
- **损失**：与 B3 相同——每通道 `BCEWithLogits(正样本 ∪ 固定负样本)`，再按通道平均；无额外辅助损失。
- **防泄漏**：GCN 只用 `encode_edges`（训练子图）；`CoFiDataset.validate()` 在训练前强制检查
  "val/test 正边不在编码图里、负样本不是已知正边、下游目标不跨 split"。
- **模型选择**：只用 val 早停；test 仅在 `evaluate_test=True`（默认）时**评测一次**。
- **冒烟训练不使用 test**：新增 `train_link_prediction(..., evaluate_test=False)` 与 CLI `--skip-test`，
  `metrics.json` 里 `test: null` 且 `notes` 注明 "test split NOT scored"。
- **记录**：`metrics.json` 现在含 `model.fusion_kind`（none/concat/gate/attention）、
  `model.edge_heads`（shared/per_channel）、`parameter_count`、`output_dim`、完整 `history`、
  `dataset.dataset_sha256`、`git_commit`、依赖版本。

## 9. 单元测试结果（本轮实际执行）

```
$ .venv/bin/python -m pytest -q
147 passed in 3.29s          # Week 2 结束时 99 passed，本轮 +48
```

`tests/test_week3_models.py`（**41 例**）覆盖：

| 验收点 | 用例 |
| --- | --- |
| B4 输出 `[N,128]` | `test_week3_models_emit_one_128_dim_embedding_per_channel[b4]` |
| B5 输出 `[N,128]` | 同上 `[b5]` |
| 门控 = 单 Linear + Sigmoid（无隐藏层） | `test_gated_fusion_is_a_single_linear_layer` |
| 门控公式 `H = g*H_A+(1-g)*H_B` | `test_gated_fusion_implements_the_documented_equation` |
| 门值域 ∈ [0,1] | 同上 + `test_gated_fusion_at_half_gate_is_the_plain_average` |
| 门是特征级（每基因 128 值） | `test_gate_is_feature_wise_not_one_scalar_per_graph` |
| 梯度到门控网络、也到两个编码器 | `test_gated_fusion_reaches_both_encoders_with_gradients`、`test_week3_models_backpropagate_finite_gradients_everywhere` |
| B5 头数=4、每头 32 维 | `test_attention_uses_four_heads_of_32_dims_by_default` |
| 每个 Query 恰好 2 个候选 | `test_attention_has_exactly_two_candidate_keys_per_query` |
| 注意力权重和为 1、有限、非退化 | 同上（`sum(dim=-1)==1`、两侧 ∈ (0,1)、`isfinite`） |
| 注意力与文档公式逐元素一致 | `test_attention_scores_match_the_documented_equations`（手工重算 `softmax([QA·KA, QA·KB]/√32)`） |
| 双向：A→B 与 B→A 都流动 | `test_attention_mixes_the_other_graph_into_each_query` |
| 4 头学到不同偏好 | `test_attention_heads_learn_different_candidate_preferences` |
| 残差默认关闭、可开启 | `test_attention_residual_is_off_by_default_and_opt_in` |
| 来源特异投影 | `test_attention_uses_source_specific_projections` |
| 无 NaN/Inf | `test_week3_models_backpropagate_finite_gradients_everywhere`、`..._deterministic_in_eval_mode` |
| dtype 处理（默认 float32） | `test_week3_models_use_float32_by_default_and_survive_a_step` |
| device 处理（CPU/CUDA 自适应） | `test_week3_models_follow_the_requested_device`（无 CUDA 时在 CPU 上执行同一路径） |
| 优化器确实更新参数 | `test_week3_models_use_float32_by_default_and_survive_a_step`（更新前后嵌入必须不同） |
| 独立 A/B 边预测头 | `test_week3_models_keep_graph_specific_edge_heads`、`test_week2_models_score_identically_with_or_without_a_channel_argument` |
| checkpoint 存取 | `test_week3_checkpoint_roundtrip_restores_embeddings[b4/b5]` |
| 工厂 + 配置实例化 | `test_week3_models_are_built_from_the_shipped_configs`、`test_registry_advertises_b4_and_b5_only_as_implemented` |
| B3/B4 参数量对齐 | `test_b4_is_parameter_matched_with_b3` |
| B6/B7 仍被拒绝 | `test_registry_advertises_b4_and_b5_only_as_implemented` |

## 10. 回归测试结果

```
$ .venv/bin/python -m pytest -q         # 全量
147 passed
```

- B0/B1/B2/B3 的原有用例（训练端到端、确定性、AUPRC、CUDA 缺失报错、CLI dry-run）**全部保留并通过**；
  本轮没有改动 B0–B3 的数学定义、数据、负样本或指标代码。
- 唯一触碰的既有测试是 `tests/test_models.py::test_registry_builds_every_implemented_model` 的**玩具维度**
  （`hidden_dim=6` → `hidden_dim=8`, `feature_dim=4`），原因是 6 不能被 B5 的 4 头整除；
  断言集合与语义未变。
- B0–B3 的 resolved-config hash 与 Week 2 完全一致（§1.1），因此以 config hash 命名的历史 run 名不需要重建。

## 11. 冒烟训练结果（合成数据 + CPU，**不是**科学结论）

命令（`runs/week3_smoke/`，`--skip-test` 保证 test split 完全没被读取）：

```bash
python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \
    --processed data/processed/synthetic_v1
python -m cofi_gene.train --config configs/b4.yaml --dataset data/processed/synthetic_v1 \
    --out-dir runs/week3_smoke --epochs 40 --seed 7 --run-name week3_smoke_b4 --skip-test
python -m cofi_gene.train --config configs/b5.yaml --dataset data/processed/synthetic_v1 \
    --out-dir runs/week3_smoke --epochs 40 --seed 7 --run-name week3_smoke_b5 --skip-test
# 回归对照（B0–B3 协议未变）
python -m cofi_gene.train --config configs/b3.yaml --dataset data/processed/synthetic_v1 \
    --out-dir runs/week3_smoke --epochs 40 --seed 7 --run-name week3_smoke_b3 --skip-test
```

| run | fusion | 参数量 | epochs | train loss | best_epoch | val_mean_AP | test | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `week3_smoke_b3` | concat | 70,272 | 40 | 3.7378 → 0.2928 | 20 | 0.8610 | `null`（未评测） | 285,733 B |
| `week3_smoke_b4` | gate | 70,272 | 40 | 6.3447 → 0.2839 | 20 | 0.9149 | `null`（未评测） | 285,861 B |
| `week3_smoke_b5` | attention | 169,344 | 40 | 0.9556 → 0.3601 | 30 | 0.8393 | `null`（未评测） | 685,033 B |

逐条核对冒烟验收点（针对 B4 与 B5）：

| 要求 | 结果 | 证据 |
| --- | --- | --- |
| 前向可用 | ✅ | 40 epoch 全部完成，`history` 完整 |
| 反向可用 | ✅ | loss 从首轮到末轮显著变化；所有参数 `grad` 非空（单测） |
| loss 有限 | ✅ | `all(isfinite(history.train_loss))` 为 True（存在尖峰：B4 最大 28.98、B5 最大 141.56，均为有限值且随后收敛） |
| 优化器更新参数 | ✅ | 1-epoch 与 40-epoch 的 checkpoint 逐张量比较：B4 `10/10`、B5 `22/22` 张量已改变 |
| 分别保存冒烟 checkpoint | ✅ | `runs/week3_smoke/<run>/model.pt`（与 Week 2 产物目录完全分开） |
| 未使用 test split | ✅ | `metrics.json.test == null`，`notes[0] == "test split NOT scored: ..."` |
| 记录耗时/告警 | ✅ | 每 run 约 **1.7 s**（CPU，40 epoch，60 基因）；无 Python 告警 |

> 如何解读：60 个基因、16 维特征、val 只有 9–10 条正边，上表的 val AP 差异**没有统计意义**，
> 只说明"B0–B5 都能前向、反向、更新参数、落盘、可复现"。真实结论需要真实数据 + 锁定 D-018 + 3 seed。

## 12. 运行环境与耗时

| 项 | 值 |
| --- | --- |
| 平台 | Linux（本机开发机） |
| Python | 3.12.2 |
| torch | **2.15.0+cpu**（`torch.cuda.is_available() == False`） |
| numpy / pyarrow | 2.5.4 / 26.0.0 |
| GPU | 本机为 RTX 4060 Laptop；**5090 训练机未使用、未验证** |
| 测试耗时 | `pytest -q` ≈ 3.3 s（147 例） |
| 冒烟耗时 | 约 1.7 s / run（40 epoch，60 基因，CPU） |
| 装置/精度 | 全部 float32、CPU；`--device cuda` 在无 CUDA 时显式报错 |

**GPU 未验证**：本轮没有在任何 GPU 上运行；`AGENTS.md` 要求的"5090 + 真实规模"验证仍然缺失。

## 13. 已知限制

- **L-1 无真实数据、无 GPU**：所有结果来自 60 基因合成图 + CPU。真实规模（数千基因 / 数十万边）的显存、
  每 epoch 时间、`torch.sparse` 在 5090（sm_120）上的行为都未验证。
- **L-2 B5 的投影必须被训练**：`[Z_A;Z_B]→128` 投影只有在"head 解码最终嵌入"时才拿到梯度。
  若未来把 head 改回直接解码 `Z_A`/`Z_B`，该投影会变成**未训练的随机投影**并污染下游对比；
  单测 `test_week3_models_backpropagate_finite_gradients_everywhere` 会立刻失败，请不要放宽它。
- **L-3 容量不匹配**：B3 = B4 = 70,272（参数对齐，可直接比较），但 B5 = 169,344。
  B5 相对 B3/B4 的增益可能部分来自容量，报告中必须如实说明；若要"容量可比的注意力对照"，
  那是**新增模型**，需先批准。
- **L-4 超参仍是占位值**（D-018）：`lr=0.01` 在小图上会尖峰（见 §11），不能据此比较模型优劣。
- **L-5 单 seed**：冒烟只跑了 seed=7；`AGENTS.md` 要求主要结果 3 seed 均值±标准差。
- **L-6 门控/注意力形式仍待确认**（D-019/D-020/D-021）：本轮按执行指令实现，但尚未成为团队锁定决定。
- **L-7 B5 是"每基因两 token"的轻量注意力**，不是邻域注意力：不能用"哪些邻居参与"的信息（范围控制所需）。
- **L-8 `fusion_representations()` 会重算编码器**：仅用于分析，不进计算图、不参与训练，但大图调用有额外耗时。
- **L-9 B6/B7 不存在**：`build_model("b6"/"b7")` 抛 `NotImplementedError`（刻意，Feedback 属 M4）。

## 14. 未解决的研究决定（不得写成已定结论）

### 14.1 本轮执行指令与仓库既有计划的差异（已按执行指令实现，需人类确认）

| 条目 | `AGENTS.md` 原写法 | 本轮执行指令 | 本轮实现 | 建议 |
| --- | --- | --- | --- | --- |
| B4 门控 | "得到两份来源特异、带残差的更新表示" | 单个 `Linear` + Sigmoid，输出**一个** `H` | 按执行指令：单 `Linear`，单 `H` | 若确认为最终方案，请更新 `AGENTS.md` 4.2 的描述 |
| B4 门控网络 | 未指定 | 简单线性门（不要隐藏层） | 单 `Linear(2d→d)` + Sigmoid | 建议锁定为 D-019 |
| B5 头数 | 未指定 | **4 头 × 32 维** | 默认 4 头（可配置） | 建议锁定为 D-020 |
| B5 残差 | "加入残差连接" | 公式中**未含**残差 | 默认关闭，`attention_residual: true` 可开启 | 需两人决定，见 D-020 |
| B5 输出投影 | "拼接 + 线性投影至 128" | 同 | `Linear([Z_A;Z_B]→128)` | 一致，无冲突 |

### 14.2 待确认/阻塞项

1. **D-018 训练预算**（lr / epochs / early-stop）：不定则 B0–B7 消融表不成立。
2. **D-019 / D-020 / D-021**：门控形式、注意力形式（含残差开关）、双图监督方式，已在
   `docs/DECISIONS.md` 末尾追加，等待两人确认。
3. **D-005 / D-009 / D-012 / D-014 / D-017**：真实数据相关（STRING 版本/阈值、符号冲突、PCA 拟合范围、DE 标签），
   不定则无法产出真实数据的 B4/B5 结果。
4. **是否需要"容量可比的注意力对照"**：属研究范围变更，未经批准不做。
5. **W2-1 / W2-2 两个 Week 2 观察项**（§1.2）：需要批准后才改，本轮未动。

## 15. Week 4 准备建议

1. **锁 D-018**，并在真实数据上跑 `configs/b0..b5.yaml` 单 seed 冒烟，记录显存 / 每 epoch 时间 / 参数表；
   同时补 `scripts/check_environment.py --strict` 在 5090 上的留档（M0 收尾）。
2. 用**同一划分 + 同一预算**产出 B0–B5 上游 Link Prediction AUPRC 表（D-003 主指标），
   并随表记录 `fusion_kind` / `edge_heads` / `parameter_count`，便于解释（尤其是 B5 的容量差）。
3. 为 B5 增加**注意力权重可视化**（每基因 A/B 权重分布；A 图边与 B 图边上是否系统性偏好某一来源），
   供第 8 周"局部案例图"使用，不引入新模型。
4. 决定 D-020 的残差开关后再跑 3 seed（`attention_residual` 不同 → 视为不同配置，需分开记录）。
5. **不要**提前实现 B6/B7（M4），也**不要**在 D-018 锁定前跑全量多 seed 实验。



