# 新对话 / 本地 Agent 交接协议

## 从新对话开始（可复制提示词）

> 请接手 GitHub 仓库 https://github.com/huqian2024/CoFi-Gene 。按顺序读取根目录 `AGENTS.md`、`README.md`、`docs/STATUS.md`、`docs/DECISIONS.md`、`docs/DATA_CONTRACT.md`、`docs/ROADMAP.md`，再查看最新提交和代码。以仓库实际内容为准，不假定之前聊天进度。先总结已完成/待完成/风险，再只处理 `STATUS.md` 里的第一项，修改后跑对应测试并更新 `STATUS.md`。不要改动 B0–B7 研究边界，不得混入图隐藏边或测试扰动响应标签。

## 当前仓库快照（2026-11-01，Week 2 结束时）

- 分支：`feat/week2-data-pipeline`（未合入 `main`）；`main` 上只有 M0 的骨架提交。
- 已实现：STRING 双通道读取、ID 对齐、图构建、边/目标划分、负采样、泄漏防线、特征降维、PIE DE 标签、
  数据集容器与 manifest、B0–B3 训练/评估、两个合成数据脚本、`configs/`、99 个测试、两个 CI job。
- 已验证：`python -m pytest -q` → 99 passed（CPU，torch 2.15.0+cpu）；合成数据端到端训练可跑通且同 seed 完全可复现。
- **未验证**：真实 STRING/PIE 数据、RTX 5090 上的 torch/CUDA、B4–B7、鲁棒性实验、下游 DE。
- 待确认参数：`docs/DECISIONS.md` 的 D-005、D-009～D-018。

接手时的最小操作序列：

```bash
python -m pip install -e '.[dev,data]'      # 需要训练再装 .[train]
python -m pytest -q
python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \
    --processed data/processed/synthetic_v1
python -m cofi_gene.train --config configs/b3.yaml --seed 42 --epochs 4
```

## 已知的坑（都在代码里留了注释）

1. STRING 单通道分数为 0 表示"该类型无证据"，迭代器会跳过；**绝不能**用 `combined_score` 当通道。
2. `combine score` 陷阱、未知蛋白、同符号多蛋白都写在 `cofi_gene.synthetic` 里，并由测试断言。
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
