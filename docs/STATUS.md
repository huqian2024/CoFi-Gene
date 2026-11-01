# 项目状态（每次合并 PR 后更新）

最后更新：2026-11-01（M0 完成：起步包已合并至仓库根目录、本地单测通过、首次推送）
阶段：**M0 已完成 → 进入 M1（数据契约）**

## 已确认的事实

- 研究设计在根目录 `AGENTS.md`，范围固定为 B0–B7；文件顶部已加入 Agent 工程维护入口。
- 团队两人：coding 与模型实现主要由一人负责；另一人负责上平台训练，并共同确认数据/生物学设置。
- 用户反馈服务器已配置；尚未看到 `torch.cuda` / PyG 检查结果，因此 **GPU 可用性未验证**。
- 远程仓库 https://github.com/huqian2024/CoFi-Gene 公开、默认分支 `main`；本地工作目录 `/home/qian/CoFi-Gene` 已展平为工程根目录并推送。
- 本机（开发机）为 CPU 环境：Python 3.12.2、numpy 1.26.4、pyarrow 16.1.0、**无 torch / 无 PyG**、GPU 为 RTX 4060 Laptop（**不是** 5090 训练机）。
- PIE 发布 DE parquet 和预处理数据；STRING/PIE 的真实文件尚未检查、下载或对齐。

## 当前验收

| 工作项 | 状态 | 验收证据 |
| --- | --- | --- |
| 原始研究范围 B0–B7 | 已定义 | `AGENTS.md` |
| 项目骨架 / 单测 | 已合并、可安装、本地通过 | 起步包展平到仓库根目录；`python -m venv .venv && pip install -e '.[dev]'` 成功；`.venv/bin/python -m pytest -q` → 5 passed（pytest 9.2.0，CPU，Python 3.12.2） |
| CI（GitHub Actions） | 已通过 | run 41281008328（commit `b97c19a`，job `CPU unit tests`: success） |
| 服务器 torch / CUDA / PyG | 未验证 | 环境检查 JSON、GPU 上最小算子 |
| STRING 两图 / ID 对齐 | 未开始 | 版本、节点/边/覆盖率统计、哈希 |
| PIE DE 读取 / 标签定义 | 未开始 | schema 与阳性率报告 |
| B0–B3 真实数据训练 | 未开始 | 日志、验证集 AUPRC、保存的配置 |
| B4–B7 | 未开始 | 一致的对照设置 |
| unseen perturbation 下游 | 未开始 | 分组 split 检查与 AUPRC |

## 正在执行（最多 1–2 项）

1. **M0 已完成**：起步包合并 + 首次 CI 通过（run 41281008328）；`AGENTS.entry.patch` 因上下文不匹配（原文件标题为加粗）改为手工插入后已删除。
2. **M1（下一步）**：训练服务器跑 `python scripts/check_environment.py --strict`；STRING/PIE 原始数据小样本核验，填写 `docs/DATA_CONTRACT.md` 待定项并锁定 D-005～D-008。

## 阻塞项/需两人敲定

- STRING 的发布版本、物种（预期人类 9606，待确定）、证据阈值与 ID 映射规则。
- `gene_text` 源版本/索引、特征压缩到 128 维的方法与统一基线。
- 下游 DE 标签的显著性阈值及是否附加效应量门槛；PIE 单元格/基因覆盖率。
- 训练服务器的软件版本与数据目录、产物保存位置。

## 最近完成

- 2026-11-01：起步包展平合并到仓库根目录（`README.md`、`pyproject.toml`、`src/`、`tests/`、`scripts/`、`docs/`、`.github/`），`AGENTS.md` 顶部加入 Agent 维护入口，`.gitignore` 补充 `.pytest_cache/` 等缓存项。
- 本地验证：`.venv/bin/python -m pytest -q` → 5 passed（Python 3.12.2，pytest 9.2.0，CPU）；`.venv/bin/python -c "import cofi_gene"` 正常，src 布局可安装。
- 开发机需经 `127.0.0.1:7890` 代理访问 GitHub 与 PyPI（本仓库 `.git/config` 已设 http(s).proxy；pip 需临时 `export HTTPS_PROXY=http://127.0.0.1:7890`）。
- 当前检查环境没有 PyG 或 GPU，不能代表队友服务器环境。

## 维护规则

- 状态只写实际验证过的事情；“计划”“实施中”“已完成”严格区分。
- 完成代码任务必须记录：commit/PR、测试命令与结果、数据版本、实验 run ID、下一步。
- 在新对话里先以此文件为准，不从上一次对话推断进度。
