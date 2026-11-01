# 新对话 / 本地 Agent 交接协议

## 从新对话开始（可复制提示词）

> 请接手 GitHub 仓库 https://github.com/huqian2024/CoFi-Gene 。按顺序读取根目录 `AGENTS.md`、`README.md`、`docs/STATUS.md`、`docs/DECISIONS.md`、`docs/DATA_CONTRACT.md`、`docs/ROADMAP.md`，再查看最新提交和代码。以仓库实际内容为准，不假定之前聊天进度。先总结已完成/待完成/风险，再只处理 `STATUS.md` 里的第一项，修改后跑对应测试并更新 `STATUS.md`。不要改动 B0–B7 研究边界，不得混入图隐藏边或测试扰动响应标签。

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
