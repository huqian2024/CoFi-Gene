# CoFi-Gene

基于两张 STRING 证据图、独立 GCN、跨图融合和单轮 Graph Feedback 的 Gene Embedding 课程项目。

**先读**：`AGENTS.md`（研究范围与 B0–B7 设计）→ [`docs/STATUS.md`](docs/STATUS.md)（当前事实与下一步）→ [`docs/DECISIONS.md`](docs/DECISIONS.md)（已锁定及待确认决定）→ [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)（数据约定）→ [`docs/TEAM_HANDOFF.md`](docs/TEAM_HANDOFF.md)（队友视角交接：已完成/需确认/需准备/需验证）。

> 当前状态：数据管线与 B0–B3 已实现，并在**合成数据 + CPU** 上验证；**真实 STRING/PIE 数据与 GPU(5090) 均未验证**。详见 `docs/STATUS.md` 的"四层状态区分"。

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

## 快速开始（当前已实现范围：数据管线 + B0–B3）

```bash
python -m pip install -e '.[dev,data]'          # 数据管线 + 测试
python -m pytest -q                              # 99 个用例（缺 torch 时训练用例自动 skip）

# 1) 用合成数据把整条链路跑通（无需下载、无需 GPU）
python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \
    --processed data/processed/synthetic_v1

# 2) 训练上游模型（B0–B3）
python -m cofi_gene.train --config configs/b3.yaml --seed 42
# 产物：runs/<run_name>/{metrics.json, config.json, model.pt}

# 3) 真实数据：先填 configs/data.yaml 的文件路径与已确认参数，再构建
python scripts/build_dataset.py --config configs/data.yaml --out data/processed/real_v1
```

训练机需要单独安装 torch（按该机 CUDA 版本选择 wheel）：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU 机器示例
```

## 代码结构

| 路径 | 作用 |
| --- | --- |
| `src/cofi_gene/string_io.py` | STRING 明细表流式读取，**只按单通道列**过滤（不用 combined score） |
| `src/cofi_gene/gene_ids.py` | 蛋白 ID → 基因符号归一化、冲突计数 |
| `src/cofi_gene/graph_build.py` | 去重/自环/权重策略、图统计、双图重合率 |
| `src/cofi_gene/splits.py` | 边划分、按扰动目标分组、缺边模拟、双视图遮蔽 |
| `src/cofi_gene/negative_sampling.py` | 避开全部已知正边的确定性负采样 |
| `src/cofi_gene/features.py` | `gene_text` 读取与 3072→128（SVD/PCA，固定符号） |
| `src/cofi_gene/pie_io.py` | PIE DE 表读取与**唯一**标签规则 |
| `src/cofi_gene/dataset.py` | 数据集容器（含 `validate()` 泄漏防线）与存取 |
| `src/cofi_gene/pipeline.py` | 原始文件 → 数据集 + 构建报告 |
| `src/cofi_gene/models/` | `torch.sparse` 实现的 GCN、B3 拼接融合、边预测头、B0–B3 注册表 |
| `src/cofi_gene/train.py` | 共同训练器（LP BCE + 固定负样本 + val 早停 + test 只评一次） |
| `src/cofi_gene/synthetic.py` | 合成 STRING/PIE 文件生成器（含若干"陷阱"用于测试） |
| `configs/` | `_base.yaml`、`b0–b3.yaml`、`data.yaml`、`data.synthetic.yaml` |
| `scripts/` | `build_dataset.py`、`make_synthetic_dataset.py`、`inspect_pie_de.py`、`check_environment.py` |

## 当前边界（务必区分"已实现"与"已验证"）

- 已实现并测试：数据读取/对齐/划分/防泄漏/特征压缩/B0–B3 训练与评估，全部在**合成数据 + CPU** 上验证。
- **未做**：真实 STRING/PIE 数据接入、GPU(5090) 验证、B4–B7、缺边鲁棒性、Gene ID 错位对照、下游 DE MLP、多种子汇总。
- 待团队确认的参数见 `docs/DECISIONS.md`（D-005、D-009～D-018）。


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
