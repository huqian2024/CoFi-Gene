# CoFi-Gene

基于两张 STRING 证据图、独立 GCN、跨图融合和单轮 Graph Feedback 的 Gene Embedding 课程项目。

**先读**：`AGENTS.md`（原仓库已有的研究范围与 B0–B7 设计）→ [`docs/STATUS.md`](docs/STATUS.md)（当前事实与下一步）→ [`docs/DECISIONS.md`](docs/DECISIONS.md)（已锁定及待讨论决定）→ [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)（数据约定）。

> 本初始包仅新增工程结构和数据泄漏防线，**不包含真实数据、已训练模型或验证过的 GPU 训练结果**。不要用它覆盖仓库原有的 `AGENTS.md` / `.gitignore`。

## 研究边界

- 图 A：STRING experimental evidence；图 B：STRING co-expression evidence（同数据库的不同证据通道）。
- B0 原始特征，B1/B2 单图，B3 拼接，B4 Gate，B5 Attention，B6 Attention + Feedback，B7 Gate + Feedback。
- 上游 Link Prediction AUPRC / 缺边鲁棒性；下游按未见扰动目标划分的 DE AUPRC。
- 原则：统一基因顺序、共享数据划分/训练预算，不把隐藏边送入编码/反馈，不让测试扰动的响应标签进入训练。

## 初次使用

```bash
git clone https://github.com/huqian2024/CoFi-Gene.git
cd CoFi-Gene
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_environment.py
```

服务器安装 PyTorch + PyG 时，先按该服务器的驱动/CUDA/显卡实际情况选兼容版本，不要从这里猜一个固定 CUDA wheel。环境检查不等于 GPU 已完成训练验证。

若所在机器必须走代理，先在当前 shell 设置（示例为本地 Clash 端口，按实际改）：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
```

工程起步包（`README.md`、`pyproject.toml`、`src/`、`tests/`、`scripts/`、`docs/`、`.github/`）已合并到仓库根目录；`AGENTS.md` 顶部已加入 Agent 工程维护入口，原 `AGENTS.entry.patch` 因上下文不匹配（标题为加粗）改为手工插入后删除。

## 近期里程碑

1. `python scripts/check_environment.py --strict` 在训练服务器通过，并记录结果（服务器运维）。
2. 下载/抽样检查 PIE 的 DE parquet 与 `gene_text`，STRING 的两个证据通道；锁定基因 ID 和标签规则（数据契约）。
3. 先在小型合成图上实现 B0–B3 的训练/评估端到端，再用真实数据做小规模 smoke run。
4. B4/B5 → B6/B7 → 下游 MLP → 完整多种子实验（见 `docs/ROADMAP.md`）。

## 协作原则

- Coding 同学：数据管线、训练/评估代码、单测、统一 CLI/配置、代码 review。
- 训练同学：数据含义与 ID / 标签确认、训练服务器配置、跑实验/归档产物、共同检查泄漏和结果。
- 每次完成任务后更新 `docs/STATUS.md`；改实验定义时新增一条 `docs/DECISIONS.md`；别把数据、权重、密钥直接 commit。
- 新聊天请附本仓库链接，并让助手按 **AGENTS → STATUS → DECISIONS → DATA_CONTRACT** 读取；仓库中记录的状态比聊天记忆更可靠。

## 数据来源

- https://string-db.org/
- https://huggingface.co/datasets/arcinstitute/PIE_replogle_nadig_essential
- https://huggingface.co/datasets/arcinstitute/PIE_sources (其中 `gene_text`)
- https://github.com/ArcInstitute/pie
