# Week 4 实施报告（M4：B6/B7 One-Step Graph Feedback）

最后更新：2026-11-15
范围：`AGENTS.md` 固定范围 B0–B7 中的 **M4（B6 = B5 + 反馈、B7 = B4 + 反馈）**。
本报告只记录**实际执行并验证过**的内容；未验证项（真实 STRING/PIE、RTX 5090）会显式标注。

| 项 | 值 |
| --- | --- |
| 分支 | `feat/week4-graph-feedback`（基于 `main` = `eddf1d7`），合并前不 push |
| 提交 | `6d2c584` 实现；`b66c392` 防泄漏审计/等价性测试；本报告的文档与 CI 提交见 `git log -1 -- docs/WEEK4_IMPLEMENTATION_REPORT.md` |
| 测试 | `python -m pytest -q` → **186 passed**（CPU，4.25 s；Week 3 结束时 147） |
| 验证层级 | **仅 CPU + 合成数据**（60 基因 / 16 维特征）；真实数据与 GPU **未验证** |

---

## 1. 目标与完成范围

M4 的目标：检验"把增强后的表示反馈到图结构、再编码一次"是否有额外收益（`AGENTS.md` 4.3）。
本轮**只做代码 + CPU 验证**，不做真实数据实验，也不改动 B0–B5 的数学定义与训练目标。

已完成：

- 新增独立反馈模块 `src/cofi_gene/models/feedback.py`（纯函数 + 无参数 `nn.Module`）。
- B6 = B5（对称双向 Cross-Graph Attention）+ 单轮边权反馈；B7 = B4（Feature-wise Gated Fusion）+ 单轮边权反馈。
- 模型只接收**已遮蔽的训练编码图**（`set_encoding_graphs()`），反馈**只重加权既有边**，不补边（D-022）。
- 注册表、配置接口、训练入口、`configs/b6.yaml`、`configs/b7.yaml` 接通；B0–B5 的 resolved config hash 与行为不变。
- 新增 39 个测试（`tests/test_week4_feedback.py` 24 例、`tests/test_week4_leakage.py` 7 例，以及 8 个既有参数化矩阵的扩展）。

明确**未做**：真实 STRING/PIE 数据、GPU 训练、缺边鲁棒性（M5）、下游 DE MLP（M6）、3 seed 汇总（M7）、
候选边补边（D-022 明确不做）、warmup 后冻结式反馈（D-023 记录为待确认默认）。

## 2. B6/B7 的完整计算流程

对每个通道 `c ∈ {A, B}`（两图共享同一节点集与同一 `X`）：

```
# ---- 第一次编码（编码图 = 训练子图）----
H_c^(1) = GCN_c(X, Â_c)                    Â_c = D_c^-1/2 (A_c + I) D_c^-1/2

# ---- 该通道自己的边预测头打分（与 LP 损失同一 head，不新增参数）----
s_e     = head_c(H_c^(1)[u], H_c^(1)[v])           e = (u, v) ∈ E_c^enc
p_e     = sigmoid(detach(s_e))                     # 关键：stop-gradient

# ---- 有幅度限制的边权重更新（拓扑不变）----
w'_e    = clamp( w_e · (1 + α (2 p_e − 1)),
                 min = (1 − m) w_e,
                 max = (1 + m) w_e )               α = 0.1, m = 0.5

# ---- 重建归一化邻接并二次编码（同一套 GCN 参数）----
Â'_c    = D'^-1/2 (A'_c + I) D'^-1/2               A'_c 与 A_c 边集完全相同，只有权重变
H_c^(2) = GCN_c(X, Â'_c)

# ---- 各自融合 ----
B6: [Z_A ; Z_B] = Attention(H_A^(2), H_B^(2)) → Linear(256→128) → H    （Z 来自 B5 模块）
B7: H = g ⊙ H_A^(2) + (1 − g) ⊙ H_B^(2)，g = sigmoid(Linear([H_A^(2); H_B^(2)]))   （B4 模块）

# ---- 监督 ----
logits = head_c(H, e)  for e ∈ train 正边 ∪ 固定负样本；BCE, 按通道平均
```

`H` 恒为 `[N, 128]`，与 B0–B5 一致；下游 `encode()` 接口不变。

## 3. Edge Weight Feedback 的实现与参数

| 项 | 值 / 位置 |
| --- | --- |
| 纯函数 | `feedback_weights(weights, scores, alpha=0.1, margin=0.5)`（`models/feedback.py`） |
| 模块 | `EdgeWeightFeedback(alpha, margin)`：`forward(embeddings, edge_index, edge_weight, head, num_nodes) → (w', Â')` |
| 超参 | `α = 0.1`、`m = 0.5`（`configs/b6.yaml`、`configs/b7.yaml` 显式写入；配置键为可选键，**不进** `config.py: DEFAULTS`） |
| 参数量 | 反馈模块**无参数**；B6/B7 参数量与 B5/B4 完全相同 |
| 数值性质 | 结果非负、有限；`α = 0` 时返回输入权重本身（严格 no-op）；`m = 0` 时退化为恒定权重 |
| 边界处理 | 空边图 → 只剩自环（`N` 个），输出有限；孤立节点经自环保持度数 ≥ 1；权重自动对齐到 scores 的 dtype/device |
| 结构校验 | `validate_encoder_graph()`：拒绝自环、重复无向边、越界索引、形状不是 `[2, E]`；在 `set_encoding_graphs()` 时执行一次 |
| 图的来源 | 只接收 `CoFiDataset` 的 `encode_edges`（训练子图，或 `mask_other_view=True` 时再遮蔽对方视图隐藏边）及其权重 |
| 额外约束 | 每通道独立重加权，A 的边不会写入 B；不新增候选边；不读 val/test 边或标签 |

## 4. B6/B7 与 B5/B4 的继承关系

- `BASE_MODEL_OF = {"b6": "b5", "b7": "b4"}`；构造函数里 `base_name` 决定融合模块与输出维度，
  因此 **B6 复用 `CrossGraphAttention`（4 头 × 32 维、残差默认关）**，**B7 复用 `GatedFusion`（单层 `Linear(2d→d)` + Sigmoid）**，
  这两处源码**一行未改**（`git diff` 可核对）。
- `_FUSION_KIND`：`b6 → attention`、`b7 → gate`，`metrics.json` 的 `model.fusion_kind` 据此记录。
- `PER_CHANNEL_HEAD_MODELS` 扩展为 `b4/b5/b6/b7`：B6/B7 每通道一个独立 edge head（与 B5/B4 一致），
  同一 head 既用于反馈打分，也用于 LP 损失——不引入额外解码器。
- 新逻辑只在 `feedback is not None` 分支生效：不调用 `set_encoding_graphs()` 时，B0–B5 的
  `forward` 路径与 Week 3 逐行等价（`_encode` + `_fuse` 拆分是纯重构，147 个既有用例原样通过）。

## 5. GCN 参数共享、梯度 detach 与参数量

**参数共享**：两次编码使用同一个 `self.encoders[c]` 实例。测试 `test_the_second_pass_reuses_the_same_encoder_module`
在 `encoders["A"]` 上挂 forward hook，单次 `forward` 计数为 **2**，证明不是"第二个编码器"。

**梯度路径**：`p = sigmoid(detach(s))` 阻断了"调低自己被打分的边权来提高分数"的退化解；
梯度只经**第二次编码**与 **LP 打分头**回流。测试 `test_feedback_weights_are_detached_from_the_score_head`
断言返回权重的 `grad_fn is None`、重建邻接的 `values().requires_grad == False`；
`test_backward_reaches_encoders_heads_and_fusion` 断言 `x.grad` 与全部参数梯度有限且非全零。

**参数量**（`model.parameter_count()` 实测）：

| feature_dim | B4 | B5 | B6 | B7 |
| --- | --- | --- | --- | --- |
| 16（合成数据） | 70,272 | 169,344 | **169,344（= B5）** | **70,272（= B4）** |
| 128（真实数据目标维度） | 98,944 | 198,016 | **198,016（= B5）** | **98,944（= B4）** |

反馈不引入任何参数，因此 `B5 vs B6`、`B4 vs B7` 的差异**只来自结构反馈**，不存在容量混淆。

## 6. 防泄漏设计与测试证据

设计（红线条款见 `TRAINING_HANDOFF.md` §8 与 `docs/DATA_CONTRACT.md` §1）：

1. 模型不持有 dataset，只持有 `set_encoding_graphs()` 注册的 `encode_edges` / `encode_weights`。
2. `encode_edges` 由管线保证等于训练子图（`mask_other_view=True` 时再去掉对方视图隐藏边）；
   `CoFiDataset.validate()` 在加载/构建时强制 `encode ∩ (val ∪ test) = ∅`。
3. 反馈邻接由 `normalized_adjacency(encode_edges, w')` 重建 → **边集不变**，无法凭空引入隐藏边。
4. 负样本在构建期生成且排除 `train|val|test` 全部正边 → 不会出现在编码图中。
5. 验证/测试阶段：`evaluate_split` 用 `model.eval()` + `torch.no_grad()`，反馈重新计算但只用注册图；
   前后参数与邻接逐位不变（已断言）。
6. 跨通道：A/B 各自重加权，不复制对方边。

测试证据（`tests/test_week4_leakage.py`，共 7 例，其中 1 例按 b6/b7 参数化 → 8 个执行项）：

| 用例 | 断言 |
| --- | --- |
| `test_feedback_adjacency_is_exactly_the_encoder_graph` | 反馈邻接的边集 == 该通道 `encode_edges`，且与 val/test 正边不相交 |
| `test_negatives_and_hidden_edges_never_enter_the_encoder_graph` | `encode` 与 val/test、以及三个 split 的负样本均不相交 |
| `test_swapping_hidden_positives_leaves_training_bit_identical` | **只交换 val/test 正边**（编码图、train 正边、负样本不变）→ B6/B7 的 `train_loss` 历史与最终 `state_dict` 逐位相同，而 `val` 指标确实改变 |
| `test_validation_does_not_touch_parameters_or_the_encoder_graph` | 验证前后参数、基础邻接、注册图逐位不变，且恢复 `train()` 模式 |
| `test_feedback_reads_the_registered_graph_not_the_training_positives` | 改编码边权重 → 输出变化；模型不可达 dataset |
| `test_masked_protocol_is_respected_when_both_views_hide_a_pair` | `mask_other_view=True` 时 `encode ⊆ train`，且与自身/对方视图的 val/test 都不相交；反馈邻接 == 该编码图 |

## 7. alpha = 0 等价性验证

`α = 0` 时 `w' = w`，第二次编码与第一次完全相同，故 B6 退化到 B5、B7 退化到 B4。

| 用例 | 条件 | 结果 |
| --- | --- | --- |
| `test_alpha_zero_reproduces_b5_and_b4_in_eval_mode` | `eval()` + 同 seed + `load_state_dict` 共享参数 + 同邻接 | **`torch.equal` 逐位相等**（B6≡B5、B7≡B4，A/B 两通道） |
| `test_alpha_zero_matches_in_train_mode_when_dropout_is_zero` | `train()` 且 dropout=0 | 同样逐位相等 |
| `test_feedback_changes_the_second_encoding_and_margin_zero_disables_it` | `α = 0.1` vs `α = 0`；`m = 0` vs `α = 0` | 前者输出不同（反馈确实生效），后者与 `α = 0` 完全一致（clamp 冻结） |
| `test_alpha_zero_is_an_exact_noop` / `test_margin_zero_freezes_the_weights` | 纯函数层 | `torch.equal(w', w)` 成立 |

因此 `feedback_alpha: 0` 可作为"B6/B7 与 B5/B4 的实现回归开关"，用于排查后续改动。
注意：**`α=0` 只保证边权不变**；训练模式下两次编码的随机性来自 dropout，需要固定 seed 或 `eval()`
才能比较（测试即按此方式设计）。

## 8. 合成数据 Smoke Test（可复现命令）

数据：`data/processed/synthetic_v1`（60 基因 / 16 维；A 编码边 81、B 编码边 72），CPU，seed=42，3 epoch，不评 test：

```bash
python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \
    --processed data/processed/synthetic_v1

for m in b6 b7; do for r in 1 2; do
  python -m cofi_gene.train --config configs/$m.yaml \
      --dataset data/processed/synthetic_v1 --out-dir runs/week4_verify/$m-$r \
      --run-name ${m}_s42_r$r --seed 42 --epochs 3 --skip-test
done; done
```

实测结果（两次独立运行）：

| 模型 | best_epoch | val_mean_ap（两次） | history / val / state_dict | `dataset_sha256` | `test` |
| --- | --- | --- | --- | --- | --- |
| B6 | 3 / 3 | 0.635594 / 0.635594 | 逐位相同 | 相同 | `null`（`--skip-test`） |
| B7 | 3 / 3 | 0.886939 / 0.886939 | 逐位相同 | 相同 | `null`（`--skip-test`） |

两次运行的 `config` 仅 `runtime.out_dir` / `runtime.run_name` 不同；`model` / `data` / `optim` 段完全相同。
另一次更短的冒烟（seed=7、2 epoch）得到 B6 `0.6028`、B7 `0.8308`。

> ⚠️ 上述 AP 只说明**代码链路可训练、可复现**，**不代表任何真实数据性能**，也不构成 B6 vs B5、B7 vs B4 的结论。

产物：`runs/<out-dir>/<run-name>/{metrics.json, config.json, model.pt}`；`metrics.model` 中记录
`uses_feedback: true` 与 `feedback: {alpha, margin, base_model}`。`runs/` 已在 `.gitignore`，不入库。
`test_checkpoint_roundtrip_rebuilds_and_forwards` 验证保存的 checkpoint 能按记录配置重建并 forward。

## 9. CPU 全量测试

```
$ .venv/bin/python -m pytest -q
186 passed in 4.25s          # Week 3 结束时 147 → Week 4 结束 186（+39）
```

环境：Python 3.12.2、numpy 2.5.4、pyarrow 26.0.0、**torch 2.15.0+cpu**、无 CUDA、无 PyG。
新增用例分布：

| 文件 | 用例数 | 覆盖 |
| --- | --- | --- |
| `tests/test_week4_feedback.py` | 24 | 公式/边界、detach、邻接重建、注册表、空图/孤立点/极端 logits、dtype 对齐、forward/backward、`α=0` 等价性、编码器复用、checkpoint 重载、trainer 元数据 |
| `tests/test_week4_leakage.py` | 7（含 b6/b7 参数化 → 8 项） | 见 §6 表格 |
| 既有文件扩展 | 8 | `test_train.py`（b6/b7 加入端到端与确定性矩阵）、`test_models.py`、`test_config.py`、`test_week3_models.py`（注册表/融合类型断言） |

配置未变（实测 resolved hash）：`b0 bd7bb75ec0f2`、`b1 f3160ec62e2d`、`b2 0ea7242eb7f9`、
`b3 f2ebde49513f`、`b4 eb0a3fe0c5a8`、`b5 29ee907107e4`；新增 `b6 a4456156d5fd`、`b7 5df6a41740ac`。

## 10. 已知局限与实验风险

1. **真实数据 / GPU 完全未验证**：所有"跑通"均为 60 基因、16 维的合成数据 + CPU。真实 3–5k 基因、
   128 维、A/B 边数量级（10^5–10^6）下的耗时、显存、收敛行为均未知。
2. **反馈计算成本**：每次 forward 多一次编码 + 每通道一次邻接重建（对称化 + 归一化），
   约 **2× 前向开销**；无 AMP、无 batch/分块，显存随节点/边数增长。
3. **训练边打分偏乐观**：反馈使用与 LP 损失共享的 head，而它打分的是训练正边（模型已拟合过），
   `p` 系统性偏高 → 边权整体上调（受 `m=0.5` 限制）。若真实数据上出现"权重普遍膨胀、反馈收益随 epoch 消失"，
   应记录该现象，不要靠调 `α` 掩盖。
4. **D-023（待确认）**：当前实现是"每个 forward 重算一次反馈"；`AGENTS.md` 4.3 的"只更新一次图结构"
   也可解读为"warmup 后冻结一次"。后者需要额外的 warmup 预算（影响 D-018），本轮未实现。
5. **D-016（待确认）**：`mask_other_view` 默认 `false`（"另一视图可见"口径）。因此某条在 A 中被隐藏的边
   可能出现在 B 的编码图中，B6 的跨图注意力可间接利用；这是**协议选择**，不是实现泄漏，但报告措辞必须按 D-016 口径。
6. **选项/超参未锁定**：`α=0.1`、`m=0.5` 是工程默认（D-022 记录），未做敏感性分析。
7. **既有问题未处理**：`optim.negatives_ratio` 只被校验、不生效（负样本在构建期固定，W2-1）。
8. 未做候选边补边（D-022 有意排除），也未做"反馈轮数 > 1"。

## 11. 与队友（真实数据 / GPU 训练）的交接事项

权威操作手册是 [`TRAINING_HANDOFF.md`](../TRAINING_HANDOFF.md)（本文不重复命令），M4 相关补充：

- 批次顺序建议：`B3 → B4/B5 → B6/B7 → B0–B2`，全部使用**同一份** `data/processed/<name>`、
  同一 `--seed`、同一训练预算（`configs/_base.yaml`，D-018 未锁定前不要中途改）。
- B6/B7 的 `metrics.json` 必须包含 `model.uses_feedback == true` 与
  `model.feedback == {alpha, margin, base_model}`；缺这两项说明跑的不是反馈模型。
- 自检开关：把 `feedback_alpha` 改成 `0`（或在配置里覆盖）应与 B5/B4 的 loss/指标逐位一致；
  若不一致，先查数据版本或 dropout/seed，不要直接下结论。
- 报数字时同时给 `positive_rate`、`n_positives/n_negatives` 与数据集 `sha256`；
  **不得**把合成数据 AP 写进真实数据表格。
- 若显存不足：反馈会多一次前向，不要用减小 `hidden_dim` 的方式"绕过"（会破坏与 B4/B5 的对照），
  先记录规模与报错并上报。

## 12. 相关提交

| 提交 | 内容 |
| --- | --- |
| `6d2c584` | feat(week4): B6/B7 单轮边权反馈（含 `models/feedback.py`、注册表/训练入口/配置、`configs/b6–b7.yaml`、初版测试、D-022/D-023、STATUS 引用修正） |
| `b66c392` | test(week4): 防泄漏审计、B5/B4 等价性、极端条件与 checkpoint 重载用例 |
| 本报告的提交 | docs(week4): 实施报告 + STATUS/README 同步 + CI 增加 B6/B7 smoke（`git log -1 -- docs/WEEK4_IMPLEMENTATION_REPORT.md`） |

复核命令：`git log --oneline eddf1d7..HEAD`。


