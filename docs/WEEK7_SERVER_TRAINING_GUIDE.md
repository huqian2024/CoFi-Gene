# Week 7 服务器训练指南（三种子 + 三类图）

面向：**真实数据 + GPU 训练负责人**。本轮的目标只有两个：**B0–B7 × 3 个模型种子**（报告 mean±std 与配对差），
以及**为三类图生成可追溯的输入文件**。不含新模型、不改研究范围。
前置：M5.1 的 `data/processed/real_v1_1` 已构建且审计 PASS（见 `docs/M5_1_SERVER_TRAINING_GUIDE.md` §5）。

## 1. 本轮已锁定的决策（不要自行更改）

| 编号 | 结论 |
| --- | --- |
| D-033 | 训练预算锁定为 `configs/_base.yaml` 现值：`lr=0.01`、`weight_decay=0`、`epochs=200`、`eval_every=10`、`patience=50`、`grad_clip=5`、`dropout=0.2`；**B0–B7 与所有 seed 共用，禁止按模型单独调** |
| D-034 | Stage 5 = `real_v1_1` + `cross_view_transfer`（Protocol A），seed **42/43/44**；数据划分固定，只改模型初始化/训练洗牌种子 |
| D-035 | 图 1（下游 DE）等 M6 代码入库后从原始 `metrics.json` 生成，**不得用手抄数值画图** |

## 2. 代码同步与环境

```bash
git fetch origin && git checkout main && git pull --ff-only
git log --oneline -1           # 记下 commit sha，回报时需要
git status -sb                 # 必须干净
python -m pip install -e '.[dev,data]'
python -m pytest -q            # 期望 338 passed
```

本轮**没有新增依赖**（图是纯 Python 生成的 SVG，不需要 matplotlib）。

## 3. 训练前确认（3 分钟）

```bash
# ① 数据版本与审计
python scripts/audit_dataset.py --dataset data/processed/real_v1_1 \
  --out runs/week7/audit/real_v1_1_transfer.json
# 退出码必须是 0；把打印的每通道 val/test 与 visible_in_other_encode 贴回

# ② 预算就是仓库里的值，不要用 --epochs 覆盖正式 run
grep -E 'lr|epochs|eval_every|early_stop_patience|grad_clip' configs/_base.yaml

# ③ 先看 Stage 5 计划（24 格，不训练）
python scripts/run_experiments.py --stage 5 --dataset data/processed/real_v1_1 --dry-run
```

## 4. Stage 5：三种子正式训练（24 格）

```bash
python scripts/run_experiments.py --stage 5 \
  --dataset data/processed/real_v1_1 \
  --out-dir runs/week7/stage5 \
  --device cuda --audit strict --resume
```

- 24 格 = 8 模型 × seed 42/43/44，run 名形如 `s5_b5_xfer_drop00_adj_seed43`；
- 每格是一次完整训练；`--resume` 只跳过 `metrics.json` 里 `status=complete` 的格子；
- 失败单格重跑：把 `--resume` 去掉并加 `--only-models b5 --only-seeds 43`；
- **不要**用 `--epochs` 缩短正式 run（会破坏 D-033 的可比性）。

## 5. 聚合：mean±std、配对差、预算饱和

```bash
python scripts/aggregate_seeds.py --runs runs/week7/stage5 \
  --out-dir runs/week7/aggregate --name seed_aggregate
```

产物：`seed_aggregate.json` + `seed_aggregate.md`（模型×通道 mean±样本 std、11 组配对差、预算饱和表）。
**必须回报的两个数字**：

1. `budget.n_saturated` / `budget.n_runs`（有多少 run 的最优 epoch 落在 200 上限）；
2. `warnings` 里是否出现"混数据集 / 混协议 / 重复 seed"（正常应为空）。

## 6. 出图

输出目录建议 `figures/week7/`（SVG 可直接插入 PowerPoint/Word；需要 PNG 时用 Inkscape 或
`rsvg-convert -w 1600 fig.svg -o fig.png`，仓库不装绘图库）。

### 图 1a：上游 Link Prediction AP（mean ± std）

```bash
python scripts/make_figures.py --kind upstream \
  --input runs/week7/aggregate/seed_aggregate.json \
  --out figures/week7/fig1a_upstream_ap.svg
```

单图模型（B1 只有 A、B2 只有 B）在另一通道上**留空**，这是预期行为。

### 图 2：鲁棒性曲线

用 Stage 4（监督感知删边 0%/50%）或 `scripts/run_robustness.py` 的 summary：

```bash
python scripts/make_figures.py --kind robustness \
  --input runs/week7/stage4/stage4_summary.json --channel A \
  --out figures/week7/fig2_robustness_A.svg
```

### 图 3：Graph Feedback 局部案例

```bash
# 用 Stage 5 的 B7 checkpoint（verification：metrics.json 里看 best_epoch）
python scripts/export_feedback_case.py \
  --checkpoint runs/week7/stage5/s5_b7_xfer_drop00_adj_seed42/model.pt \
  --dataset data/processed/real_v1_1 --channel A --top-k 12 \
  --focus-genes <GENE1> <GENE2> <GENE3> \
  --out runs/week7/cases/case_b7_A.json

python scripts/make_figures.py --kind feedback \
  --input runs/week7/cases/case_b7_A.json \
  --out figures/week7/fig3_feedback_A.svg
```

- `--focus-genes` 用来把案例限制在可解释的局部（不填则取全局 |Δw| 最大的边）；
- 案例 JSON 里每条边都有 `weight_before/weight_after/delta/score`，报告可引用具体数值；
- 反馈只调整**既有训练边**的权重，不补边；图中红色=升权、蓝色=降权，线宽∝反馈后的权重。

### 图 1b：下游 DE AUPRC（等 M6 入库后）

M6 的 `scripts/run_downstream_de.py` 入库后，把逐 run 结果整理成下面这张表，再出图：

```json
{
  "metric": "macro_target_ap",
  "models": {
    "b0": {"mean": 0.0, "std": 0.0, "n": 3},
    "b1": {"mean": 0.0, "std": 0.0, "n": 3}
  }
}
```

```bash
python scripts/make_figures.py --kind downstream \
  --input runs/week7/downstream_table.json \
  --out figures/week7/fig1b_downstream_de.svg
```

若 M6 的 `metrics.json` 字段与本表不同，**不要手改数字**，把字段名回报，本地加一层适配。


## 7. 需要回传的文件

| 路径 | 内容 | 必须 |
| --- | --- | --- |
| `runs/week7/stage5/<run>/metrics.json` | 每格完整记录（含 `protocol`/`audit`/`status`） | ✅ 24 份 |
| `runs/week7/stage5/<run>/audit.json` | 该格的启动审计 | ✅ |
| `runs/week7/stage5/<run>/config.json` | 解析后的配置（核对预算是否被改） | ✅ |
| `runs/week7/stage5/s5_b7_*/model.pt` | 图 3 依赖，至少 seed42 的 B7 | ✅ |
| `runs/week7/stage5/stage5_summary.json` / `.md` | 阶段汇总 | ✅ |
| `runs/week7/aggregate/seed_aggregate.json` / `.md` | mean±std、配对差、预算饱和 | ✅ |
| `runs/week7/audit/real_v1_1_transfer.json` | 数据审计（含跨视图可见比例） | ✅ |
| `runs/week7/cases/case_b7_A.json` | 反馈案例原始数值 | ✅ |
| `figures/week7/*.svg` | 已生成的图 | ✅（能生成就交） |
| 失败格子清单 + 完整错误栈 | —— | ✅（即使为空也要说明） |

回传时同时写清：**commit sha、数据版本、GPU 型号、每格耗时（如已记录）、预算饱和数**。

## 8. 故障处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `aggregate` 返回码 2 / `no metrics.json found` | `--runs` 指错目录 | 指向 `runs/week7/stage5`（递归查找） |
| 某模型 `n=2` 或 `n=1` | 某个 seed 的 run 失败/未完成 | 用 `--only-models <m> --only-seeds <s>` 重跑该格，再重新聚合 |
| `make_figures` 拒绝出图 | summary 混了数据集/协议或重复 seed | **不要**直接加 `--allow-warnings`；先按 `warnings` 拆表或修数据 |
| 柱状图某通道为空 | 该模型不使用该通道（B1/B2） | 正常；报告里说明"B1 仅评 A、B2 仅评 B" |
| `missing frozen checkpoint` | 未先跑 Stage 5 的 B7 | 先跑 Stage 5（或至少 b7 一格） |
| `none of the focus genes ... has an encoder edge` | 选的基因在该通道无训练边 | 换基因，或去掉 `--focus-genes` 先看全局 |
| 预算饱和比例很高（如 >50%） | 200 epoch 仍在约束结果 | **如实回报**，由团队决定是否改 D-033；不得自行延长某一模型 |
| SVG 打不开 | 部分工具不识别 | 用浏览器/Inkscape 打开；转 PNG：`rsvg-convert -w 1600 in.svg -o out.png` |
| 输出目录冲突 | 复用了同一 `--out-dir` 且没带 `--resume` | 每阶段固定目录；需要重跑时用 `--only-*` 或先改名备份 |

## 9. 回报模板（直接填写）

```markdown
### Week 7 执行回报

- 代码 commit：`____________`（`main`）
- 数据：`data/processed/real_v1_1`，dataset.npz sha256 `____________`
- 数据审计：`runs/week7/audit/real_v1_1_transfer.json` → 状态 ____
  - 跨视图可见：A ____ / ____（比例 ____）、B ____ / ____（比例 ____）
- 预算：epochs ____ / patience ____ / lr ____（应与 D-033 一致）
- 环境：GPU ____ / torch ____ / CUDA ____

#### Stage 5（24 格）
- 完成 ____ / 24；失败或缺失：____________________
- 预算饱和：`budget.n_saturated` = ____ / 24
- 聚合告警：____________________（正常应为空）
- 汇总文件：`runs/week7/aggregate/seed_aggregate.md`

#### 关键数字（填 `seed_aggregate.md` 的 mean ± std）
| 模型 | A 通道 AP | B 通道 AP |
| --- | --- | --- |
| b0 | | |
| b1 | | |
| b2 | | |
| b3 | | |
| b4 | | |
| b5 | | |
| b6 | | |
| b7 | | |

配对差（同 seed 先做差）：`b5-b4` ____、`b6-b5` ____、`b7-b4` ____、`b6-b7` ____

#### 图
- [ ] `figures/week7/fig1a_upstream_ap.svg`
- [ ] `figures/week7/fig2_robustness_A.svg`
- [ ] `figures/week7/fig3_feedback_A.svg`
- [ ] 图 1b（下游 DE）：等待 M6 代码 / 已生成（说明来源）

#### 异常
- 失败格子与错误：______________________________________
- 任何与历史 M5/M6 结论方向不一致之处：__________________

#### 回传清单
- [ ] 24 份 `metrics.json` / `audit.json` / `config.json`（+ B7 `model.pt`）
- [ ] `stage5_summary.json` / `.md`
- [ ] `seed_aggregate.json` / `.md`
- [ ] `case_b7_A.json` + 三张 SVG
```
