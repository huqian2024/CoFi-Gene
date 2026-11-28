# Week 7 实施报告（代码侧：固定超参、三种子聚合、三类图）

日期：2026-11-28 ｜ 分支：`codex/m5.1-evaluation-audit` ｜ 环境：Python 3.12 / torch 2.15.0+cpu
范围：**只做本地代码工作**。按任务约定，正式训练、GPU 实验、多种子实跑由服务器负责人执行
（指南见 `docs/WEEK7_SERVER_TRAINING_GUIDE.md`）。本报告不含任何真实数据指标。

## 1. 本轮已确定的决策

| 决策 | 结论 | 影响 |
| --- | --- | --- |
| D-033（收口 D-018） | 预算锁定 `lr=0.01 / 200 epoch / eval_every=10 / patience=50 / grad_clip=5 / dropout=0.2`，B0–B7 与所有 seed 共用 | 与 M5/M6 可比；报告必须写"最优 epoch 落在上限"的 run 数，不能称"已收敛" |
| D-034 | Stage 5 三种子 = `real_v1_1` + `cross_view_transfer`（Protocol A），seed 42/43/44，划分固定 | 与历史可比；必须附 `audit.json` 的跨视图可见比例 |
| D-035 | 图 1（下游 DE AUPRC）等 M6 代码入库后从原始 `metrics.json` 生成，**不使用文档手抄值** | 暂时用上游 AP 图占位；下游图标记"待 M6 复现" |

## 2. 新增代码

| 文件 | 作用 |
| --- | --- |
| `src/cofi_gene/aggregate.py` | 读取 `metrics.json` → 每模型×通道 mean/样本 std、预定义**配对差**（同 seed 先做差）、预算饱和统计、provenance 与 `integrity` 阻断标记 |
| `src/cofi_gene/figures.py` | 零依赖 SVG 渲染：分组柱状图（带误差线/允许缺值）、折线图、反馈案例图（节点环 + 边宽∝权重 + 红/蓝=升/降 + Top-|Δw| 面板） |
| `src/cofi_gene/feedback_report.py` | 用训练同一条公式（`feedback_weights`）导出每边 `before/after/Δ/score`，支持 `top_k`、`focus_genes`、`select_case` |
| `src/cofi_gene/models/__init__.py` | 新增 `encoding_weights()` 公开访问器（报告不再读私有属性） |
| `src/cofi_gene/evaluate.py` | 抽出 `build_model_from_checkpoint()`，冻结评估与反馈导出共用同一重建逻辑 |
| `scripts/aggregate_seeds.py` | 聚合 CLI：`--runs ... --metric --out-dir --json`，无 `metrics.json` 时返回码 2 |
| `scripts/make_figures.py` | 出图 CLI：`--kind {upstream,downstream,robustness,feedback}`；表格混数据/混协议时**拒绝出图**（`--allow-warnings` 可显式覆盖） |
| `scripts/export_feedback_case.py` | 从 B6/B7 checkpoint 导出反馈案例 JSON（可选 `--focus-genes/--top-k/--protocol`） |

设计约束（与 M5.1 一致）：不新增模型、不改 B0–B7 定义；绘图不引入 matplotlib；一切数字来自已有文件。

## 3. 本地端到端冒烟（合成数据，**不是科学结果**）

```bash
python scripts/run_experiments.py --stage 5 --dataset /tmp/m51b/p \
  --only-models b0 b1 --only-seeds 42 43 44 --epochs 2 --out-dir /tmp/w7/stage5
python scripts/aggregate_seeds.py --runs /tmp/w7/stage5 --out-dir /tmp/w7/agg
python scripts/make_figures.py --kind upstream \
  --input /tmp/w7/agg/seed_aggregate.json --out /tmp/w7/fig1_upstream.svg
python scripts/run_experiments.py --stage 4 --dataset /tmp/m51b/p \
  --only-models b1 b5 --epochs 1 --out-dir /tmp/w7/stage4
python scripts/make_figures.py --kind robustness \
  --input /tmp/w7/stage4/stage4_summary.json --channel A --out /tmp/w7/fig2_robustness_A.svg
python scripts/run_experiments.py --stage 1 --dataset /tmp/m51b/p \
  --only-models b7 --epochs 1 --out-dir /tmp/w7/stage1
python scripts/export_feedback_case.py \
  --checkpoint /tmp/w7/stage1/s1_b7_xfer_drop00_adj_seed42/model.pt \
  --dataset /tmp/m51b/p --channel A --top-k 8 --out /tmp/w7/case_b7_A.json
python scripts/make_figures.py --kind feedback \
  --input /tmp/w7/case_b7_A.json --out /tmp/w7/fig3_feedback_A.svg
```

实际输出（节选）：

```
[aggregate] b0: A: 0.8967 ± 0.0227 (n=3) | B: 0.7423 ± 0.0044 (n=3)
[aggregate] b1: A: 0.8513 ± 0.0076 (n=3)
[aggregate] budget: 6/6 run(s) at the epoch cap [2]
[feedback-case] model=b7 channel=A selected=8/81 up=8 down=0 max|Δw|=99.7001 (α=0.1, m=0.5)
```

- 三张 SVG 均生成（4.3 KB / 3.5 KB / 10.5 KB），含坐标轴、误差线、图例与 Top-|Δw| 面板；
- 6/6 run 在 2 epoch 上限饱和 → 聚合器按 D-033 如实报警（正是 Week 7 需要暴露的现象）；
- B1 只有 A 通道，柱状图对缺失的 B 留空而不是伪造 0 或悄悄丢掉通道。

## 4. 三类图的当前状态

| 图 | 计划 | 现状 | 数据来源 |
| --- | --- | --- | --- |
| 图 1 下游 DE AUPRC（带误差线） | 必做 | **代码就绪、数据待 M6 入库**（D-035）；上游 AP 图已可立即生成 | M6 `metrics.json` → 标准表 → `--kind downstream` |
| 图 2 Graph Robustness 曲线 | 必做 | ✅ 可生成 | `run_robustness.py` summary 或 `run_experiments` stage 4 summary |
| 图 3 Graph Feedback 局部案例 | 必做 | ✅ 可生成 | B6/B7 checkpoint → `export_feedback_case.py` |

图 1 的"标准表"契约（M6 入库后必须满足，或加一层薄适配）：

```json
{
  "metric": "macro_target_ap",
  "models": {
    "b0": {"mean": 0.294553, "std": 0.001138, "n": 3},
    "b7": {"mean": 0.289564, "std": 0.001619, "n": 3}
  }
}
```

## 5. 测试

`python -m pytest -q` → **337 passed**（M5.1 的 300 例 + Week 7 新增 37 例，无 skip）。新增：

| 测试文件 | 覆盖 |
| --- | --- |
| `tests/test_week7_aggregate.py` | mean/样本 std、同 seed 配对差、符号稳定性、预算饱和、legacy/incomplete、混数据/混协议告警、重复 seed、坏文件跳过、integrity 阻断标记、AUC 指标切换、Markdown/JSON 落盘 |
| `tests/test_week7_figures.py` | 柱状图（含误差线、缺失通道留空）、折线图、案例图（节点/边/面板/图例）、非法输入与 `--only SVG` |
| `tests/test_week7_feedback_case.py` | 每条边 Δ=after−before、margin 上界、top_k、focus_genes、无反馈模型/错通道/空图报错、确定性 |
| `tests/test_week7_scripts.py` | 三个 CLI 的退出码与产物、混表拒绝/`--allow-warnings`、单图模型缺值、下游表、鲁棒性两种输入格式、导出案例 JSON |

## 6. 未做 / 边界（必须如实说明）

1. **未执行任何正式训练**：没有真实数据、没有 GPU、没有 3 seed 实跑；本轮产出是"能生成 Week 7 结果"的工具链与文档。
2. **图 1 依赖 M6 代码**：`8ff31f8` / `scripts/run_downstream_de.py` 仍不在本仓库（B-11）。
3. **SVG 转 PNG/PDF** 需手动转换（Inkscape / `rsvg-convert` / PowerPoint），仓库不引入 matplotlib。
4. 报告中的合成数字只用于验证管线，**不得写入课程结果表**。
