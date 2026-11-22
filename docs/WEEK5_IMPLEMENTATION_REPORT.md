# Week 5（M5）实施报告：缺边鲁棒性 + 基因错位对照（代码侧）

日期：2026-11-22 ｜ 范围：`AGENTS.md` §6 实验二、实验三的**代码与协议** ｜ 未新增模型（B0–B7 定义不变）

## 0. 边界（本次只做 coding 侧）

| 归属 | 内容 |
| --- | --- |
| 本次交付（coding） | 协议扰动模块、模型侧错位对照、训练器/JSON 记录、批量驱动脚本、测试与 CI 冒烟、文档 |
| 不属本次（训练/数据方） | 真实 STRING/PIE 上的 10/30/50% 曲线实跑、3 seed 汇总、D-016 口径的最终批准、生物学解释 |
| 明确未改动 | `string_io` / `pipeline` / `pie_io` / `features` / `splits` 既有拆分语义、B0–B7 数学定义、负样本、已有 `metrics.json` 字段 |

默认行为保持原样：三个新开关都**不写入 `DEFAULTS`**，因此 B0–B7 的 resolved config、`resolved_config_hash`、run 名与 `metrics.json` 在未开启扰动时逐字节不变（有单测与 `edge_drop_fraction=0` 等价性用例守护）。

## 1. 三个协议扰动

### 1.1 缺边鲁棒性（`data.edge_drop_fraction`）

只对每个通道的 **`encode_edges`（训练/编码图）** 做确定性删边：

```
kept, removed = simulate_missing_edges(encode_pairs, fraction, seed=channel_seed(seed, channel_index))
```

- `val`/`test` 正边与全部负样本**完全不参与**抽样，所以 10/30/50% 三档共享同一评估集（`n_positives`/`n_negatives` 恒定，单测断言）。
- 每通道种子 `channel_seed(seed, index) = SeedSequence([seed, index, 0xC0F1])`：只依赖 `(run seed, 通道序号)`，与训练 RNG 流无关；同一模型/通道在所有比例下用**同一个 seed**，因此删边集合随比例**嵌套**（10% ⊂ 30% ⊂ 50%），曲线是单调信息损失而不是重新抽样噪声。
- 权重按 `(min,max)` 配对保留，删边后不会发生权重错位。
- 协议要求**重训**：每个 `(model, ratio)` 单元都是一次完整训练，而不是在训练好的图上评测期删边。

### 1.2 双视图遮蔽（`data.mask_other_view`，默认关闭）

对应 D-016 的“双视图均遮蔽”口径：若基因对 `p` 是通道 X 的 val/test 正边，则 `p` 不得在另一通道 Y 中可见：

1. 从 Y 的 `encode_edges` 删除（消息传递不可见）；
2. 从 Y 的 `train` 正边删除（否则 Y 的监督仍在教共享表示“p 是边”，会把 B 通道的恢复能力抬得虚高）。

`val`/`test`/负样本/节点集不变；`CoFiDataset.validate()` 在扰动后再次运行，仍然全部通过。**该口径尚未由两人正式锁定（D-016）**，所以默认仍为“另一视图可见”。

### 1.3 基因错位对照（`data.misalign_cross_view`，默认关闭）

只允许 B3–B7（有跨图交互的模型；B0 无融合、B1/B2 单视图，配置校验直接拒绝）。

```
perm = derangement(N)                      # 由 (N, seed) 决定，无不动点
per_channel[other_view] = per_channel[other_view][perm]   # 仅在 fusion 之前
```

- 编码器、两图结构、监督、评估边都不变，只有融合消费的 A/B 基因配对被打乱，正是 `AGENTS.md` 实验三要检验的“模型是否真的用到了正确的跨图对应”。
- 必须是 derangement：恒等映射是静默 no-op，不能当对照（单测拒绝）。
- 置换用 **non-persistent buffer**，因此 `state_dict()` / checkpoint 格式完全不变；可复现性靠 `metrics.json` 中记录的 `cross_view.permutation_sha256` + 由 `(seed, N)` 重算。

## 2. 记录与可复现

开启任一扰动时，`metrics.json` 增加一个 `perturbation` 块（未开启时不出现该键）：

```json
{
  "edge_drop_fraction": 0.3,
  "mask_other_view": false,
  "misalign_cross_view": false,
  "seed": 42,
  "channels": {
    "A": {"num_encode_edges_before": 144071, "num_encode_edges_after": 100850,
          "num_encode_edges_removed": 43221,
          "num_train_positives_before": 115257, "num_train_positives_after": 115257,
          "edge_drop": {"fraction": 0.3, "seed": 4204904804, "...": "..."},
          "cross_view_mask": null}
  },
  "cross_view": {"channel": "B", "permutation_sha256": "…"}
}
```

`dataset.dataset_sha256` 仍指向**未扰动**的 `dataset.npz`（扰动是运行期协议，不落盘），文档与 JSON 都写明这一点，避免误读“数据集被改过”。

## 3. 新增 / 修改文件

| 文件 | 变更 |
| --- | --- |
| `src/cofi_gene/perturbation.py` | **新增**（263 行）：`channel_seed`、`thin_channel`、`mask_cross_view_positives`、`cross_view_permutation`、`permutation_fingerprint`、`apply_data_perturbations`（纯 numpy，无 torch） |
| `src/cofi_gene/models/__init__.py` | `CROSS_VIEW_MODELS`；`cross_view_channel`/`cross_view_perm`（non-persistent buffer）；`set_cross_view_permutation` / `clear_cross_view_permutation` / `_misalign`，在 `_fuse` 与 `fusion_representations` 之前生效 |
| `src/cofi_gene/config.py` | 允许并校验可选 `data.edge_drop_fraction` / `mask_other_view` / `misalign_cross_view`（不进 `DEFAULTS`） |
| `src/cofi_gene/train.py` | 训练前应用数据扰动并复验泄漏防线；注册错位置换；写入 `perturbation` 块与 note；CLI 增加 `--edge-drop-fraction/--mask-other-view/--misalign-cross-view` |
| `scripts/run_robustness.py` | **新增**（249 行）：`plan_runs` / `summarize` / `render_markdown` / `main`，B0–B7 × 比例批量训练并写 `summary.json` + `summary.md`（每跑完一格就落盘） |
| `tests/test_week5_perturbation.py` | **新增 12 例**（无 torch）：删边只动编码图、权重对齐、确定性/嵌套、0 比例 no-op、扰动后仍过 `validate()`、遮蔽的两视图行为、derangement、配置校验 |
| `tests/test_week5_cross_view.py` | **新增 14 例**（torch）：只对 B3–B7 开放、拒绝恒等/非双射、`state_dict` 不变、置换恰好作用一次于融合前、编码器输出不变、错节点数报错、五个模型各训练 1 epoch、0 比例与基线逐位一致、评测集规模不变 |
| `tests/test_week5_robustness_script.py` | **新增 6 例**：计划/命名/参数校验、汇总表与 markdown（含缺失格显示 `—`）、`--dry-run` 不写盘、单格端到端 |
| `.github/workflows/ci.yml` | `torch-cpu` job 增加：drop=0.3、misalign、drop=0.0 三次冒烟 + 一次 `b1` 两格 sweep，并断言 JSON 内容 |

## 4. 用法

```bash
# 单次训练（适合手动抽查）
python -m cofi_gene.train --config configs/b5.yaml --dataset data/processed/real_v1 \
    --edge-drop-fraction 0.3 --seed 42 --run-name b5_drop30_seed42
python -m cofi_gene.train --config configs/b5.yaml --dataset data/processed/real_v1 \
    --misalign-cross-view --seed 42 --run-name b5_misalign_seed42
python -m cofi_gene.train --config configs/b3.yaml --dataset data/processed/real_v1 \
    --mask-other-view --seed 42 --run-name b3_maskother_seed42

# 完整曲线（B0–B7 × {0, 0.1, 0.3, 0.5}，每格一次完整训练）
python scripts/run_robustness.py --dataset data/processed/real_v1 --device cuda \
    --out-dir runs/m5_robustness --summary-name seed42
# 只跑关键对照
python scripts/run_robustness.py --dataset data/processed/real_v1 --device cuda \
    --models b1 b2 b4 b5 b6 b7 --misalign-cross-view --ratios 0 --summary-name misalign42
```

产物：`runs/m5_robustness/<model>_drop<NN>[_maskother|_misalign]_seed<S>/`（`metrics.json`/`config.json`/`model.pt`）+ `summary.json`（含每格原始指标）+ `summary.md`（AUPRC/AUROC 两张透视表，可直接转成鲁棒性曲线）。

## 5. 验证证据

| 项 | 结果 |
| --- | --- |
| 全量测试 | `python -m pytest -q` → **218 passed**（Week 4 为 186，本次 +32） |
| CI 冒烟（本地等价复跑） | `b3 drop=0.3`、`b5 misalign`、`b4 drop=0.0`、`b1` 两格 sweep 全部成功；断言 `Week 5 robustness smoke OK` |
| 0 比例等价 | `edge_drop_fraction=0.0` 与基线配置的 `train_loss` 历史、`val` 指标逐位相同，且不写 `perturbation` 块 |
| 泄漏防线 | 三档删边后 `CoFiDataset.validate()` 全部通过；`encode` 与自身 `val/test` 仍不相交；遮蔽只减少数据，不改变 val/test/negatives |
| 错位幅度 | `fusion_representations` 在置换后与“手工把 B 行打乱后喂给 fusion”逐元素一致（`atol=1e-6`） |

## 6. 未完成与风险（不得写成结论）

1. **真实数据曲线未跑**：本次只交付代码与协议；`runs/m5_robustness/` 的任何真实数据结果需由训练方执行后归档。
2. **D-016 / D-018 未锁定**：遮蔽口径与训练预算仍待两人确认；本实现默认“另一视图可见 + 现有预算”，不构成结论。
3. **单 seed**：曲线点仍是一次训练，方差未知；3 seed 属 M7。
4. **计算成本**：B6/B7 每步两次编码，`4 个比例 × 8 模型` 的成本约为 B4/B5 的 1.5–2 倍，建议先跑 `b1/b2/b3/b4/b5`，再补 `b6/b7`。
5. **B0 是平坦基线**：B0 不使用图，删边对其指标无影响，这是对照的应有行为，不要误读为“抗删边能力强”。
6. **1:1 负样本**：曲线的 AP 基线仍是 0.5 附近的既有口径（构建期固定负样本），不要与全库随机候选场景下的 AP 混用。
7. **错位对照的解释边界**：置换只破坏融合的基因配对，不动编码器；若某模型指标几乎不变，说明它主要靠单视图信息，而不是“注意力对错位免疫”。

