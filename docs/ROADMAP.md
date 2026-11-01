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

## 训练交付合同（后续实现）

计划的标准调用：`python -m cofi_gene.train --config configs/b0.yaml --seed 42`。**目前尚未实现，不能运行**。

训练方每个 run 需归档：git commit、config、seed、数据 manifest hash、Python/Torch/PyG/CUDA 版本、完整日志、validation/test metrics、checkpoint 路径、运行错误。测试集指标不用于调参。
