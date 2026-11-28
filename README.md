# CoFi-Gene

基于两张 STRING 证据图、独立 GCN、跨图融合和单轮 Graph Feedback 的 Gene Embedding 课程项目。

> 🚀 **负责真实数据与 GPU 训练的同学请直接看 [`TRAINING_HANDOFF.md`](TRAINING_HANDOFF.md)**
> —— 进度差说明、P0–P3 优先任务、可直接复制的环境/数据/训练命令、验收表与"第一波最低交付标准"。

**先读**：`AGENTS.md`（研究范围与 B0–B7 设计）→ [`docs/STATUS.md`](docs/STATUS.md)（当前事实与下一步）→ [`docs/DECISIONS.md`](docs/DECISIONS.md)（已锁定及待确认决定）→ [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)（数据约定）→ [`docs/TEAM_HANDOFF.md`](docs/TEAM_HANDOFF.md)（队友视角交接：已完成/需确认/需准备/需验证）。

> 当前状态：数据管线、B0–B3（Week 2）、**B4/B5（Week 3）**、**B6/B7（Week 4：One-Step Graph Feedback）**、
> **Week 5 / M5（缺边鲁棒性、双视图遮蔽、Gene ID 错位对照）** 与
> **M5.1（Evaluation Audit & Protocol Enhancement）** 的代码均已实现，
> 本机在**合成数据 + CPU** 上验证（338 个用例；B6/B7 已通过防泄漏审计与 `α=0` 等价性验证）；
> **Week 7（三种子聚合与三类图）的工具链已就绪**：`scripts/aggregate_seeds.py`（mean±样本 std、同 seed 配对差、预算饱和检查）、
> `scripts/make_figures.py`（upstream/downstream/robustness/feedback 四类 SVG）、`scripts/export_feedback_case.py`；
> 正式的三 seed 训练与出图仍需服务器按 [`docs/WEEK7_SERVER_TRAINING_GUIDE.md`](docs/WEEK7_SERVER_TRAINING_GUIDE.md) 执行。
> **真实 STRING/PIE 首轮（B0–B7，seed 42）由队友在 RTX 4090 上完成并合并**，结果见
> [`docs/ALL_MODELS_RESULTS_2026-11-19.md`](docs/ALL_MODELS_RESULTS_2026-11-19.md)，但其修复提交与数据产物尚未入库。
> ⚠ **M5/M6 的历史数值属于"原始评估协议下的初步结果"**：负样本可能跨 split 重复、缺边只删编码图、
> B4–B7 服务器实现与 main 语义不同；必须按 [`docs/M5_1_SERVER_TRAINING_GUIDE.md`](docs/M5_1_SERVER_TRAINING_GUIDE.md)
> 用修复后的 `real_v1_1` 复核。详见 [`docs/STATUS.md`](docs/STATUS.md) 的"M5.1"小节与
> [`docs/M5_1_EVALUATION_AUDIT.md`](docs/M5_1_EVALUATION_AUDIT.md)。

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

## 快速开始（当前已实现范围：数据管线 + B0–B7）

```bash
python -m pip install -e '.[dev,data]'          # 数据管线 + 测试
python -m pytest -q                              # 338 个用例（缺 torch 时训练用例自动 skip）

# 1) 用合成数据把整条链路跑通（无需下载、无需 GPU）
python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \
    --processed data/processed/synthetic_v1

# 2) M5.1 数据审计（PASS 才能训练）+ 非神经网络基线
python scripts/audit_dataset.py --dataset data/processed/synthetic_v1 \
    --out runs/m5_1_audit/audit.json
python scripts/run_baselines.py --dataset data/processed/synthetic_v1 \
    --out-dir runs/m5_1_baselines

# 3) 训练上游模型（B0–B7）
python -m cofi_gene.train --config configs/b3.yaml --seed 42   # 拼接基线
python -m cofi_gene.train --config configs/b4.yaml --seed 42   # Gated Fusion
python -m cofi_gene.train --config configs/b5.yaml --seed 42   # Cross-Graph Attention
python -m cofi_gene.train --config configs/b6.yaml --seed 42   # B5 + One-Step Feedback
python -m cofi_gene.train --config configs/b7.yaml --seed 42   # B4 + One-Step Feedback
# 产物：runs/<run_name>/{metrics.json, config.json, model.pt, audit.json}

# 4) M5.1 阶段化实验矩阵（Stage 0–5，支持 --dry-run / --resume）
python scripts/run_experiments.py --stage 0 --dataset data/processed/synthetic_v1
python scripts/run_experiments.py --stage 1 --dataset data/processed/synthetic_v1 \
    --out-dir runs/m5_1/stage1 --resume

# 5) Week 7：三种子聚合 + 三类图（数字全部来自 runs/*/metrics.json）
python scripts/aggregate_seeds.py --runs runs/m5_1/stage5 --out-dir runs/week7/aggregate
python scripts/make_figures.py --kind upstream \
    --input runs/week7/aggregate/seed_aggregate.json --out figures/week7/fig1a_upstream.svg
python scripts/make_figures.py --kind robustness \
    --input runs/m5_1/stage4/stage4_summary.json --channel A \
    --out figures/week7/fig2_robustness_A.svg
python scripts/export_feedback_case.py \
    --checkpoint runs/m5_1/stage1/s1_b7_xfer_drop00_adj_seed42/model.pt \
    --dataset data/processed/synthetic_v1 --channel A --top-k 10 \
    --out runs/week7/case_b7_A.json
python scripts/make_figures.py --kind feedback \
    --input runs/week7/case_b7_A.json --out figures/week7/fig3_feedback_A.svg

# 6) 真实数据：先填 configs/data.yaml 的文件路径与已确认参数，再构建（新版本目录，不覆盖 real_v1）
python scripts/build_dataset.py --config configs/data.yaml --out data/processed/real_v1_1
python scripts/audit_dataset.py --dataset data/processed/real_v1_1
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
| `src/cofi_gene/splits.py` | 边划分、按扰动目标分组、缺边（嵌套）模拟 |
| `src/cofi_gene/negative_sampling.py` | 跨 split 互斥的确定性负采样；`uniform_random` / `degree_matched` / `two_hop` |
| `src/cofi_gene/protocols.py` | **M5.1** 统一可见性协议（`cross_view_transfer` / `dual_view_strict_holdout`）+ 可见性报告 |
| `src/cofi_gene/audit.py` | **M5.1** 结构化数据审计（C01–C10／A01–A07）、`require_pass()`、JSON/Markdown 报告 |
| `src/cofi_gene/baselines.py` | **M5.1** 非神经网络基线（PA / CN / Adamic–Adar / Resource Allocation / 特征余弦） |
| `src/cofi_gene/evaluate.py` | **M5.1** 冻结 checkpoint 评估（Stage 3，按 checkpoint 自带 config 重建模型） |
| `src/cofi_gene/aggregate.py` | **Week 7** 多种子聚合：mean±样本 std、同 seed 配对差、预算饱和与 `integrity` 阻断标记 |
| `src/cofi_gene/figures.py` | **Week 7** 零依赖 SVG 渲染：柱状图（误差线/缺值）、折线图、反馈案例图 |
| `src/cofi_gene/feedback_report.py` | **Week 7** B6/B7 每条边的 before/after/Δ 与局部案例选择 |
| `src/cofi_gene/perturbation.py` | 缺边两模式（`adjacency` / `supervision_aware`）、按通道删边、错位置换；遮蔽委托 `protocols` |
| `src/cofi_gene/features.py` | `gene_text` 读取与 3072→128（SVD/PCA，固定符号） |
| `src/cofi_gene/pie_io.py` | PIE DE 表读取与**唯一**标签规则 |
| `src/cofi_gene/dataset.py` | 数据集容器（含 `validate()` 泄漏防线）与存取 |
| `src/cofi_gene/pipeline.py` | 原始文件 → 数据集 + 构建报告 |
| `src/cofi_gene/models/` | `torch.sparse` 实现的 GCN、B3 拼接融合、B4 门控融合、B5 跨图注意力、**B6/B7 单轮边权反馈**、边预测头、B0–B7 注册表 |
| `src/cofi_gene/train.py` | 共同训练器（LP BCE + 固定负样本 + val 早停 + test 只评一次） |
| `src/cofi_gene/synthetic.py` | 合成 STRING/PIE 文件生成器（含若干"陷阱"用于测试） |
| `configs/` | `_base.yaml`、`b0–b7.yaml`、`data.yaml`、`data.synthetic.yaml` |
| `scripts/` | `build_dataset.py`、`make_synthetic_dataset.py`、`inspect_pie_de.py`、`check_environment.py`、`run_robustness.py`（M5）、**`audit_dataset.py` / `run_baselines.py` / `run_experiments.py`（M5.1）**、**`aggregate_seeds.py` / `make_figures.py` / `export_feedback_case.py`（Week 7）** |

## 当前边界（务必区分"已实现"与"已验证"）

- 已实现并测试：数据读取/对齐/划分/防泄漏/负采样互斥/特征压缩/**B0–B7 训练与评估**，
  **M5.1 数据审计、两套可见性协议、两种删边模式、困难负样本、非神经网络基线、阶段化实验矩阵**，
  **Week 7 的三种子聚合与三类图工具链**，全部在**合成数据 + CPU** 上验证（`338 passed`）。
- **未做**：真实数据重训（需 `real_v1_1`）、GPU 验证、M5.1 Stage 0–5 与 Week 7 Stage 5 的真实执行、
  服务器 `codex/*` 分支入库、下游 M6 用修复后表示重跑、图 1b（下游 DE）的生成。
- M5/M6 历史数值只能标注为"原始评估协议下的初步结果"；见 [`docs/M5_1_EVALUATION_AUDIT.md`](docs/M5_1_EVALUATION_AUDIT.md)。
- 待团队确认的参数见 `docs/DECISIONS.md`（D-005、D-009～D-018、D-019～D-021、D-022/D-023、D-024～D-026、**D-027～D-032**）。
- 训练负责人请直接使用 [`docs/M5_1_SERVER_TRAINING_GUIDE.md`](docs/M5_1_SERVER_TRAINING_GUIDE.md)；
  开发侧交接见 [`docs/M5_1_DEVELOPMENT_HANDOVER.md`](docs/M5_1_DEVELOPMENT_HANDOVER.md)。


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
