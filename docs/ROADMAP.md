# 开发路线（以可验收 PR 为单位）

| 顺序 | 代码负责人交付 | 训练/数据负责人交付 | DoD（完成判据） |
| --- | --- | --- | --- |
| M0 | 包结构、文档、pytest、CI、环境探针 | 运行服务器探针 | `pytest` 全通过；GPU/PyG 信息记录 |
| M1 | PIE/STRING 读取、ID mapping、数据 contract 检查、split 持久化 | 原始数据版本、基因交集、DE 标签确认 | `manifest.json` + 各图规模/重合率 + 无泄漏拆分单测 |
| M2 | 共同 LP 训练器，B0/B1/B2/B3，日志与 checkpoint | 单种子小数据试跑 | 每组完成前向/反向和至少一轮评估，指标可复现 |
| M3 | B4 Gate + B5 Attention（双 token/source-aware），消融配置 | B0–B5 初跑 | 训练预算、参数统计、比较协议固定 |
| M4 | B6/B7 单轮反馈（先边权更新，再编码），重合边防泄漏 | 整套 B0–B7 试跑 | 最小反馈闭环和验收测试通过 |
| M5 | 10/30/50% 删边与错位测试，AUPRC/AUROC 统计 | 多模型单种子实验 | 保持固定评估集、产出上游表 |
| M6 | 冻结特征 + 统一 DE MLP（目标、响应、context、对照表达） | 下游初跑 | 按未见扰动目标分组；每扰动 AUPRC 与阳性比例 |
| M7 | 统一命令、3 种子、结果汇总/3 类图 | 运行全实验并备份产物 | 配置/版本/seed/权重/表格可追溯 |
| M8 | 清理、复现指南、报告支撑 | 结果解释与汇报 | 可从原始数据重建主要图表 |

## 第一张 PR（今天就能做）

- 合并本 starter 包；跑 `python -m pytest -q`、`python scripts/check_environment.py`。
- 让队友在服务器运行同一个检查，贴回非敏感环境信息（不要贴 token、ssh key、账号密码）。
- 下载一个 cell-line 的 PIE parquet 或获取其 schema，用 `inspect_pie_de.py` 做抽样检查。
- 决定基因 ID 映射规则、标签阈值、128 维压缩方案；记录到决策表。然后才开始真模型训练。

## 训练交付合同（B0–B3 已实现，B4–B7 待实现）

标准调用（已可用，需先构建数据集）：

```bash
python scripts/build_dataset.py --config configs/data.synthetic.yaml --out data/processed/synthetic_v1
python -m cofi_gene.train --config configs/b3.yaml --seed 42
```

`python -m cofi_gene.train --config ... --seed ...` 支持 `--dataset/--device/--out-dir/--epochs/--run-name/--dry-run`。
每个 run 在 `runs/<run_name>/` 下写 `metrics.json`（config、seed、dataset sha256、git commit、版本、val/test 指标、完整 history）、`config.json`、`model.pt`。
B4–B7 目前会被注册表拒绝（`NotImplementedError`），以避免用占位实现产出消融表。

训练方每个 run 需归档：git commit、config、seed、数据 manifest hash、Python/Torch/PyG/CUDA 版本、完整日志、validation/test metrics、checkpoint 路径、运行错误。测试集指标不用于调参。

## Week 2 交付结果（2026-11-01 定格，之后停止新功能开发）

| 里程碑 | 代码侧 | 数据/训练侧 |
| --- | --- | --- |
| M0 | ✅ 完成（骨架、CI、环境探针） | ⏳ 5090 上 `--strict` 未跑 |
| M1 | ✅ 代码完成（读取、对齐、contract、split 持久化、manifest） | ❌ 真实 STRING/PIE 未下载核验 |
| M2 | ✅ 代码完成（共同训练器 + B0–B3，合成数据端到端 + 确定性） | ❌ 真实数据/GPU 试跑未开始 |
| M3 | ✅ 代码完成（B4 单层门控 + B5 四头对称注意力，per-channel head，`encode()` 统一 128 维；48 个新用例） | ⏳ B0–B5 真实数据初跑未开始 | 训练预算（D-018）与 D-019/020/021 仍未锁定 |
| M4 | ⛔ 未启动 | — |
| M5–M8 | ⛔ 未启动 | — |

交付证据：分支 `feat/week2-data-pipeline`（未合 main）；提交 `59f0dd9`（功能）、`21d389f`（CI 记录）；
CI run 41283635590（`data-and-splits` ✅ / `torch-cpu` ✅）；`pytest -q` → 99 passed；合成数据 B0–B3 可训练且可复现。

### M3 的启动前置条件（全部满足才开工）

1. P0 决策锁定：STRING 版本/物种/阈值（D-005）、符号冲突策略（D-009）、DE 标签规则（D-014）。
2. 5090 上 `scripts/check_environment.py --strict` 通过并留档（含 torch/CUDA/GPU 型号）。
3. 真实数据构建成功：`scripts/build_dataset.py` 产出 `manifest.json`（基因数、两图边数、重合率、下游覆盖率）。
4. P1 决策锁定：降维方法/拟合基因集（D-012/D-017）、边权归一化（D-011）、负样本比例（D-015）、训练预算（D-018）。

未满足时**不要**先跑 B6/B7 或把 B3/B4/B5 的差异写成结论：预算或特征一变，B0–B7 全部要重跑。

## Week 3 交付结果（2026-11-07）

| 里程碑 | 代码侧 | 数据/训练侧 |
| --- | --- | --- |
| M3 | ✅ 完成：B4（Feature-wise Gated Fusion，单层门控）+ B5（对称双向 4 头 Cross-Graph Attention），per-channel edge head，`encode()` 统一输出 `[N,128]` | ⏳ 真实数据/GPU 初跑未开始；训练预算 D-018 未锁定 |

证据：`pytest -q` → **147 passed**（+48）；`models/attention.py`、`models/fusion.py::GatedFusion`、`configs/b4.yaml`、`configs/b5.yaml`、
`tests/test_week3_models.py`；合成数据（60 基因/16 维）上 B3/B4/B5 用 `--skip-test` 端到端跑通，产物在 `runs/week3_smoke/`；
详细报告见 **`docs/WEEK3_IMPLEMENTATION_REPORT.md`**；新增待确认项 **D-019 / D-020 / D-021**（门控形式、注意力形式、双图监督方式）。
参数统计（feature_dim=16, hidden=128）：B0 2,176 ｜ B1/B2 18,688 ｜ **B3 = B4 = 70,272（参数对齐）** ｜ B5 169,344。


