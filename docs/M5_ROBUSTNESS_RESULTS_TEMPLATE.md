# M5 鲁棒性 / 错位对照结果模板（复制后填写）

> **用法**：`cp docs/M5_ROBUSTNESS_RESULTS_TEMPLATE.md docs/M5_ROBUSTNESS_RESULTS_<YYYY-MM-DD>.md`，
> 填完数字后 push，并在 `docs/STATUS.md` 的"下一步"处加一行链接。
> **只提交文档与指标**：`data/`、`runs/`、`model.pt`、`manifest.json` 一律不入库。
> 协议与代码见 `docs/WEEK5_IMPLEMENTATION_REPORT.md`、`docs/DECISIONS.md` D-024/D-025/D-026。

## 1. 运行信息

| 项 | 值 |
| --- | --- |
| 代码提交（`git rev-parse HEAD`） | `<填写>` |
| 分支 | `codex/real-data-first-round`（已 merge `origin/main`） |
| 数据集目录 | `data/processed/real_v1` |
| `dataset.npz` SHA-256 | `fee691ce1da49aa055c12003635a527bb3acabe9f087c99c239dc4e665f8685d`（如不同请写明原因） |
| seed | 42 |
| 设备 / 环境 | `<GPU 型号>` / `<torch>` / `<CUDA>` / `<驱动>` |
| 训练预算 | epochs `<...>`、eval_every `<...>`、patience `<...>`、grad_clip `<...>`（D-018 当前状态：`<未锁定/已锁定>`） |
| 主曲线命令 | `<原样粘贴 scripts/run_robustness.py ... --summary-name seed42>` |
| 错位对照命令 | `<原样粘贴 ... --misalign-cross-view ...>` |
| 耗时 | 主曲线总时长 `<...>`；单格平均 `<...>` |

## 2. 评估集核对（必须与首轮一致）

| 通道 | `n_positives` | `n_negatives` | 阳性率 |
| --- | --- | --- | --- |
| A | `<...>` | `<...>` | 0.5 |
| B | `<...>` | `<...>` | 0.5 |

> 这四列在所有 ratio 下必须完全相同；若不同，说明评估集被改动，结果作废。

## 3. 主曲线：缺边 0 / 10 / 30 / 50%（AUPRC）

直接从 `runs/m5_robustness/seed42.md` 复制表格；每个通道一张。

**通道 A**

| model | drop 0.00 | drop 0.10 | drop 0.30 | drop 0.50 |
| --- | --- | --- | --- | --- |
| b0 |  |  |  |  |
| b1 |  |  |  |  |
| b2 |  |  |  |  |
| b3 |  |  |  |  |
| b4 |  |  |  |  |
| b5 |  |  |  |  |
| b6 |  |  |  |  |
| b7 |  |  |  |  |

**通道 B**（B1 不参与 B 通道评估，填 `—`）

| model | drop 0.00 | drop 0.10 | drop 0.30 | drop 0.50 |
| --- | --- | --- | --- | --- |
| b0 |  |  |  |  |
| b2 |  |  |  |  |
| b3 |  |  |  |  |
| b4 |  |  |  |  |
| b5 |  |  |  |  |
| b6 |  |  |  |  |
| b7 |  |  |  |  |

可选：AUROC 表从 `seed42.md` 直接复制（辅指标）。

## 4. Gene ID 错位对照（实验三）

| model | 正常 AP（A / B） | 错位后 AP（A / B） | ΔAP（A / B） |
| --- | --- | --- | --- |
| b3 |  |  |  |
| b4 |  |  |  |
| b5 |  |  |  |
| b6 |  |  |  |
| b7 |  |  |  |

> 结论只能写"下降 / 持平 / 上升"，并注明单 seed；不要写成"注意力学会了正确的跨图对应"这类机制断言。

## 5. 双视图遮蔽（可选，D-016 未锁定）

| model | ratio | 正常 AP | `--mask-other-view` AP | 备注 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

> D-016 未正式确认前，这一节只能作为补充实验，不能进主表或结论段落。

## 6. 异常与限制（必须填写）

- OOM / 训练失败：`<完整 traceback 或"无">`
- 是否命中 epoch 上限（`best_epoch == 200`）：`<是/否，涉及哪些模型>`
- loss 尖峰或指标异常：`<描述>`
- 与首轮 `docs/ALL_MODELS_RESULTS_2026-11-19.md` 的差异：`<描述>`
- 单 seed 说明：`<是>`；方差与 3 seed 属 M7。

## 7. 自检清单（push 前逐条勾）

- [ ] `summary.json` 存在且 `dataset.sha256` 与上表一致
- [ ] 每个 ratio 的 `n_positives` / `n_negatives` 与 ratio=0 相同
- [ ] ratio>0 的 run 都带 `perturbation` 块；ratio=0 的 run 不带
- [ ] 错位 run 的 `perturbation.cross_view.permutation_sha256` 有值（16 位）
- [ ] 未提交任何数据 / 权重 / 绝对路径（检查 `manifest.json` 未入库）
