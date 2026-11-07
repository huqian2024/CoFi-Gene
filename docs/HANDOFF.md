# 新对话 / 本地 Agent 交接协议

## 从新对话开始（可复制提示词）

> 请接手 GitHub 仓库 https://github.com/huqian2024/CoFi-Gene 。按顺序读取根目录 `AGENTS.md`、`README.md`、`docs/STATUS.md`、`docs/DECISIONS.md`、`docs/DATA_CONTRACT.md`、`docs/ROADMAP.md`，再查看最新提交和代码。以仓库实际内容为准，不假定之前聊天进度。先总结已完成/待完成/风险，再只处理 `STATUS.md` 里的第一项，修改后跑对应测试并更新 `STATUS.md`。不要改动 B0–B7 研究边界，不得混入图隐藏边或测试扰动响应标签。

## 当前仓库快照（2026-11-07，Week 3 / M3 代码完成）

- 分支：`feat/week2-data-pipeline`（沿用，**未合入 `main`**）；`main` 停在 M0 的 `dfa496e`。
- 提交：`59f0dd9`（Week 2 数据管线 + 防泄漏 + B0–B3）、`21d389f`（CI 记录）、Week 3 之后的改动**尚未提交**（未 commit）。
- 已实现：STRING 双通道读取、ID 对齐、图构建、边/目标划分、负样本、泄漏防线、特征压缩、PIE 标签、
  数据集容器与 manifest、**B0–B5**（Week 3 新增 B4 单层门控融合、B5 4 头对称跨图注意力）、两个数据脚本、`configs/b0–b5.yaml`、
  **147 个测试**、两个 CI job（`torch-cpu` 现在也跑 B4/B5 训练命令）。
- 已验证（合成数据 + CPU）：`python -m pytest -q` → 147 passed；B0–B5 端到端可训练、同 seed 完全可复现。
- **未验证**：真实 STRING/PIE 数据、RTX 5090 的 torch/CUDA、B6/B7、鲁棒性、下游 DE。
- 待确认参数：`docs/DECISIONS.md` 的 P0（D-005/D-009/D-014）、P1（D-011/D-012/D-015/D-017/D-018）
  以及 Week 3 新增的 **D-019/D-020/D-021**（门控形式、注意力形式、双图监督方式）。
- 队友视角的交接说明见 **`docs/TEAM_HANDOFF.md`**；Week 3 实施报告见 **`docs/WEEK3_IMPLEMENTATION_REPORT.md`**。

接手时的最小操作序列：

```bash
python -m pip install -e '.[dev,data]'      # 需要训练再装 .[train]
python -m pytest -q                          # 期望 147 passed
python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \
    --processed data/processed/synthetic_v1
python -m cofi_gene.train --config configs/b3.yaml --seed 42 --epochs 4
python -m cofi_gene.train --config configs/b4.yaml --seed 42 --epochs 4
python -m cofi_gene.train --config configs/b5.yaml --seed 42 --epochs 4 --skip-test
```

## 已知的坑（都在代码里留了注释）

1. STRING 单通道分数为 0 表示"该类型无证据"，迭代器会跳过；**绝不能**用 `combined_score` 当通道。
2. `combined_score` 陷阱、未知蛋白、同符号多蛋白都写在 `cofi_gene.synthetic` 里，并由测试断言。
3. 负样本在构建期固定，所有模型共用；不要在训练里重新采样，否则跨模型不可比。
4. 本机访问 GitHub/PyPI 需代理 `127.0.0.1:7890`。
5. 小图上 lr=0.01 曾出现 loss 尖峰，超参在 M3 固定前不要拿去下结论。

## 每次编码工作结束必须更新


1. `docs/STATUS.md`：完成项、证据（commit/测试）、当前阻塞、下一步。
2. `docs/DECISIONS.md`：只有当设计/数据规则变化时新增条目。
3. 代码 & 测试同步提交；实验结果使用可追溯的 config/seed/数据版本。
4. 若做不到某项，明确标识 **未验证/待实现**，不得把方案写成完成状态。

## Pull Request 建议模板

- Purpose：本次解决的问题
- Scope：新增/变更的文件与对外接口
- Checks：命令、pass/fail、机器（CPU/GPU）
- Leakage review：数据拆分和标签隔离是否改变
- Handoff：更新 STATUS & DECISIONS 的链接

说明：跨对话的稳定上下文依赖这些已提交且可访问的文件，而不是模型默认记住当前聊天。
