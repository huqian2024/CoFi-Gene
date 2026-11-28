# 真实数据与 GPU 训练交接（Training Handoff）

> ⚠ **已被 M5.1 指南取代（2026-11-28）**：真实数据训练请以
> [`docs/M5_1_SERVER_TRAINING_GUIDE.md`](docs/M5_1_SERVER_TRAINING_GUIDE.md) 为准（新数据版本 `real_v1_1`、
> 数据审计闸门、Stage 0–5 命令、结果回传清单）。本文件保留为历史交接记录，其中的数据细节仍有效，
> 但"直接跑 B0–B7"的流程**缺少负样本互斥修复后的审计步骤**，不要照旧执行。


> **面向**：负责真实 STRING / PIE 数据接入与 RTX 5090 训练的同学。
> **仓库**：<https://github.com/huqian2024/CoFi-Gene> ｜ **分支**：`main`（Week 2 – Week 5 已合并，代码基线 `f1c95c1`）
> **本文定位**：只讲"现在做什么、怎么跑、怎么验收"。状态事实以 [docs/STATUS.md](docs/STATUS.md) 为准，
> 待确认决策见 [docs/DECISIONS.md](docs/DECISIONS.md)，数据字段约定见 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)。
> 本文所有命令、参数、路径、产物字段均已在仓库当前提交上逐条核对过；**没有**假设 PyG、CUDA、真实数据已就绪。

## 0. 当前状态与你的任务（2026-11-22 更新，优先于下文全部旧描述）

> ⚠️ 下文 §0-old 是 2026-11-15 的记录，当时"B6/B7 未实现、真实数据未验证、M5 BLOCKED"。**这些都过期了**，以本节为准。

| 维度 | 代码（本仓库 `main`） | 真实数据 / GPU |
| --- | --- | --- |
| 数据管线（STRING 双通道、ID 对齐、划分、负样本、manifest） | ✅ 已实现，`218 passed` | ✅ 训练机已构建 `data/processed/real_v1`：18,002 基因；A 144,071 边 / B 871,260 边，重合 42,035 |
| B0–B7 | ✅ 全部实现（B6/B7 = 单轮边权反馈） | ✅ seed 42 首轮跑完（RTX 4090），见 `docs/ALL_MODELS_RESULTS_2026-11-19.md` |
| **Week 5 / M5**：缺边鲁棒性 10/30/50%、双视图遮蔽、Gene ID 错位对照 | ✅ 代码 + 协议（`scripts/run_robustness.py`，见 §4.5） | ❌ **待你执行 —— 这是你本轮的主任务** |
| 下游 DE MLP | ⛔ 未实现（M6） | — |
| RTX 5090 | — | ❌ 仍未验证（首轮跑在 4090） |

### 0.1 本轮只有三件事，按顺序做

| 顺序 | 任务 | 完成标志 |
| --- | --- | --- |
| S1 | **把服务器代码同步到 `main`**：`main` 现已包含 M5 代码（基线 `f1c95c1`），而你首轮用的 3 个修复提交 `d3c5433a435b`（基因符号数字后缀）、`7386d5ab7e4f`（GCN 邻接 CUDA 设备）、`0df87b474f02`（无参数 B0 评估）**只存在于服务器分支 `codex/real-data-first-round`，`origin` 上没有** | 这三个提交 push 上来（或让 coding 侧挑拣进 `main`），否则真实数据结果无法从仓库复现 |
| S2 | **在真实数据上跑 M5**：主曲线 + Gene ID 错位对照（命令见 §4.5） | `runs/m5_robustness/summary.json` + `.md`，以及按 `docs/M5_ROBUSTNESS_RESULTS_TEMPLATE.md` 填好的结果文档 |
| S3 | **确认两个待定口径**：D-016（是否启用双视图遮蔽）、D-018（预算是否固定为当前占位值） | 在 `docs/DECISIONS.md` 对应条目写明结论 |

### 0.2 同步约定（边界，重要）

- **分支同步**：先把 `main` 合到你的训练分支，再开始跑 M5，避免用旧代码产出曲线：

  ```bash
  git fetch origin
  git checkout codex/real-data-first-round   # 你现有的训练分支
  git merge origin/main                      # 有冲突按你的分支优先，但要保留 main 的 M5 代码
  python -m pytest -q                        # 期望 218 passed
  git push origin codex/real-data-first-round
  ```

- **只提交代码与文档**：`data/`、`runs/`、`model.pt`、`manifest.json` 都不入库（`.gitignore` 已排除）。结果表只贴指标与命令。
- **谁改什么**：`src/cofi_gene/perturbation.py` 与协议默认值由 coding 侧维护；你只通过 **config / CLI** 使用（`--edge-drop-fraction` / `--mask-other-view` / `--misalign-cross-view`）。要改协议口径，先在 `docs/DECISIONS.md` 新增条目（或改 D-024/025/026 的状态），不要直接改代码。
- **文档不撞车**：你的结果写到新文件 `docs/M5_ROBUSTNESS_RESULTS_<YYYY-MM-DD>.md`，并只在 `docs/STATUS.md` 的"下一步"处加一行链接；`STATUS.md` 其余段落由 coding 侧维护。



---

## 0-old. 历史记录（2026-11-15 视角，已被上方 §0 取代）：当前存在明确的"进度差"

> ⚠️ **模型工程已经推进到 Week 3 / M3，但仓库内没有任何一份可核验的真实数据 + GPU 训练结果。**

| 维度 | 代码 | 真实数据 / GPU 验证 |
| --- | --- | --- |
| 数据管线（STRING 双通道、ID 对齐、划分、负采样、manifest） | ✅ 已实现，`147 passed` | ❌ 真实 STRING / PIE 文件未下载、未核验 |
| B0–B5 | ✅ 已实现（B3 拼接 / B4 门控 / B5 跨图注意力） | ❌ 只在**合成数据 + CPU** 上跑通 |
| B6 / B7（Graph Feedback） | ⛔ 未实现，注册表主动抛 `NotImplementedError` | — |
| 训练环境 | ✅ CPU 上端到端可跑 | ❌ RTX 5090 / CUDA / torch / PyG **一次都没验证** |

必须明确的六件事：

1. **代码状态**：Week 3 / M3 已完成，`B0–B5` 已实现并各有配置文件和测试；`B6/B7` 仍被注册表拒绝。
2. **Week 4 计划**：下一步是开发 `B6/B7` 的 One-Step Graph Feedback（`AGENTS.md` 4.3）。
3. **验证状态**：真实数据构建与 GPU 训练**在仓库内没有可核验的完整结果**——没有 `manifest.json`、没有 GPU 上的 `metrics.json`、没有 run 日志。
4. **进度不同步**：模型实现进度**领先于**已记录的实验验证进度（验证停在 M1/M2 的"代码完成 + 合成数据"阶段）。
5. **不预设结论**：上面说的是"仓库里没有证据"，**不代表你没有开展训练**。服务器上可能已有数据、环境和试跑结果，请先按 §1 的 P0 同步实际状态，再往下走。
6. **优先级**：**不能再继续只推进模型实现而长期缺少真实数据验证。第一波真实训练现在就应该启动**，`B6/B7` 不必等。

> 一句话：先把"真实数据 manifest + GPU 环境 + 至少一个真实基线 run"落地，再谈 B6/B7 和消融结论。

---

## 1. 你的优先任务（P0 → P3）

### P0 — 同步现状（今天就做，30 分钟内可完成）

目标：让仓库里的记录与服务器实际状态一致。请提供**可核验的路径或日志**，不要只给结论。

- [ ] 服务器上已有的原始数据（STRING / PIE / gene_text）路径、大小、下载日期、版本号
- [ ] 是否已有 `data/processed/...` 数据集目录；若有，贴 `manifest.json` 的关键字段（见 §3.4-C）
- [ ] 已有环境：`python --version`、`python -c "import torch; print(torch.__version__, torch.version.cuda)"`、`nvidia-smi`
- [ ] 已有训练结果：`runs/` 下的目录名、`metrics.json` 里的 `git_commit` / `dataset.dataset_sha256` / `best_val_mean_ap`
- [ ] 目前阻塞点（缺数据？缺驱动？OOM？ID 对不上？）
- [ ] 服务器使用的代码版本：`git rev-parse HEAD`（若与 `main` 不同，先说明差异，不要直接 merge 别人的分支）

记录方式建议：把命令与输出贴在交接消息里，日志文件放 `runs/logs/`（`runs/` 已在 `.gitignore` 中，不会误入库）。

### P1 — 环境与数据（P0 之后立刻做）

- [ ] `scripts/check_environment.py --strict` 通过并留档（退出码 0），记录 torch / CUDA / GPU 型号与显存
- [ ] 真实 STRING 两份文件、`gene_text` 矩阵 + 基因顺序文件、PIE 各细胞系 DE parquet 到位（§3.1）
- [ ] 核对基因 ID 映射：`n_string_proteins` → `n_mapped_proteins` → `n_unique_symbols` → `n_genes_used`，以及冲突计数
- [ ] 核对规模：两图边数、节点数、重合边比例（`jaccard` / `frac_a_in_b`）、各 split 正边与负边数
- [ ] 确认防泄漏不变量：`CoFiDataset.validate()` 必须直接通过（不是靠人工检查）
- [ ] 产出并核验数据 manifest（`manifest.json`，字段见 §3.4-C），把关键数字填进 `docs/DATA_CONTRACT.md` 第 3 节

### P2 — 第一波真实训练

- [ ] **先跑 B3**（双图 + 拼接投影，最能代表"数据管线是否正常"的基线）
- [ ] 再跑 **B4 / B5**（门控 / 跨图注意力，与 B3 逐字相同的 optim / data / runtime）
- [ ] 条件允许时补齐 **B0 / B1 / B2**（B0 是"无图"特征基线，B1/B2 是单图基线）
- [ ] 首轮统一 `--seed 42`
- [ ] 三个模型共用同一份数据集目录 → 同一划分、同一负样本、同一训练预算（CLI 参数见 §4.2，预算见 §4.3）
- [ ] **不要等 B6/B7**：它们未实现，且不影响 B0–B5 的首轮验证

### P3 — 结果交付（每个 run 都要给全）

- [ ] Git commit（`git rev-parse HEAD`）、完整命令行（原样复制，不要手写简化版）、`config.json`
- [ ] 环境信息（torch / CUDA / GPU 型号 / 驱动版本）与数据信息（`manifest.json` 的 `processed` 段）
- [ ] 训练日志（`run.log`）、`metrics.json` 里的 `val` / `test` 指标（AUPRC = `average_precision`，AUROC = `roc_auc`，以及 `positive_rate`）
- [ ] 实际耗时与显存峰值（命令见 §4.4；注意：当前训练器**不会**自动记录这两项）
- [ ] 报错原文（若失败）：完整 traceback，不要只写"跑不通"
- [ ] 产物目录：`runs/<out_dir>/<run_name>/{metrics.json, config.json, model.pt}`
- [ ] 一份 ≤ 1 页的首轮总结：跑了什么、数据规模、指标、异常、下一步


---

## 2. 环境检查（先跑这一组）

```bash
git clone https://github.com/huqian2024/CoFi-Gene.git
cd CoFi-Gene
git rev-parse HEAD                       # 记录到交接记录里

python -m pip install -e '.[dev,data]'   # numpy / pyyaml / pyarrow / pytest
```

`torch` **不在** core 依赖里，需要按该机 CUDA 选 wheel（CPU 机器示例）：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

5090（sm_120）请装匹配其架构的 CUDA 12.x wheel；装完必须验证，不要凭 wheel 文件名推断。

```bash
python scripts/check_environment.py            # 无论成败都先跑这个，输出是 JSON
python scripts/check_environment.py --strict   # 期望退出码 0
echo "exit=$?"
python -m pytest -q                            # 期望 218 passed（缺 torch 时训练用例自动 skip）
```

判读规则（来自 `scripts/check_environment.py` 的实现）：

| 输出字段 | 含义 |
| --- | --- |
| `torch` / `torch_geometric` | 为对象 `{"error": ...}` 表示**没装成功**；为字符串才是版本号 |
| `torch_cuda_build` | torch 编译时的 CUDA 版本；`None` 表示 CPU 版 wheel |
| `cuda_available` | `false` 时任何 `--device cuda` 都会直接报错 |
| `device` / `compute_capability` | GPU 型号与算力（5090 预期为 `[12, 0]`） |
| `cuda_tensor_smoke` | 真正在 GPU 上做了一次张量运算，`true` 才算 CUDA 可用 |

- `--strict` 会**同时**要求 `torch`、`torch_geometric`、`cuda_tensor_smoke` 三者齐备。
- **PyG 装不上不影响训练**：上游实现是 `torch.sparse`（见 [docs/DECISIONS.md](docs/DECISIONS.md) D-013），B0–B5 不需要 PyG。
  若 sm_120 上 PyG 无兼容轮子，请保留非 strict 的输出并注明"PyG 未安装"，**不要**为过 `--strict` 去装不兼容的轮子。

---

## 3. 数据准备与构建

### 3.1 目录约定（与 `configs/data.yaml` 一致）

```
data/raw/string/9606.protein.links.detailed.v12.0.txt.gz   # 两通道都在这个文件里
data/raw/string/9606.protein.info.v12.0.txt
data/raw/pie_sources/gene_text.npy                         # 3072 维（须核对 meta.json）
data/raw/pie_sources/gene_text.ids.txt                     # 一行一个基因 id，与矩阵行严格对应
data/raw/pie/de/{K562,RPE1,HepG2,Jurkat}.parquet
```

- 路径与参数**只改 `configs/data.yaml`**，不要改脚本或硬编码。
- `data/` 已被 `.gitignore` 排除：原始数据、处理产物、权重都不要提交。
- `gene_text` 必须同时给矩阵文件和行顺序文件；没有 id 顺序，管线会直接报错而不是猜。

### 3.2 抽样检查 PIE 表（可选但推荐，秒级）

```bash
python scripts/inspect_pie_de.py data/raw/pie/de/K562.parquet --max-rows 20000
```

检查必需列是否存在（`target` / `feature` / `ref_mean` / `log2_fold_change` / `p_adj`）、缺失计数、抽样目标/响应基因数。
注意其自带的 `caution` 字段：抽样**不是**全局阳性率估计。

### 3.3 构建数据集（三段式：dry-run → 冒烟 → 正式）

```bash
# 0) 只构建、只打印统计，不写盘（仍然会真正读取原始文件）
python scripts/build_dataset.py --config configs/data.yaml --dry-run

# 1) 冒烟构建：截断行数，只验证管线能跑通；产物不可用于实验结论
python scripts/build_dataset.py --config configs/data.yaml \
    --out data/processed/real_smoke \
    --dataset-name real_smoke_truncated \
    --max-link-rows 2000000

# 2) 正式构建（参数按 docs/DECISIONS.md 已确认的取值填写）
python scripts/build_dataset.py --config configs/data.yaml \
    --out data/processed/real_v1 \
    --dataset-name real_v1_min_a400_b150
```

`--dry-run` 打印的结构（已实测）：`genes` / `features` / `channels`{每通道 `num_edges`、`num_nodes`、`n_rows_mapped`、`n_self_loops_dropped`、`n_duplicate_pairs_merged`、`n_edges_unknown_protein`、`n_rows_below_threshold`} / `overlap` / `splits` / `downstream` / `warnings`。

常用覆盖参数（全部会被写进 manifest 的 `config` 段，可追溯）：

| 参数 | 作用 | 关联决策 |
| --- | --- | --- |
| `--min-score-a` / `--min-score-b` | 两个通道各自的分数阈值（A=experimental，B=coexpression） | D-005（待锁定） |
| `--weight-mode {none,minmax}` | 边权是否归一化 | D-011（待确认） |
| `--n-components` | `gene_text` 降维维度（默认 128，上限为 `min(节点数, 原始维数)`） | D-012 / D-017 |
| `--p-adj-threshold` / `--min-abs-log2fc` | DE 标签规则 | D-014（待锁定） |
| `--train-fraction` / `--val-fraction` / `--split-seed` | 上/下游划分比例与种子 | — |
| `--negatives-ratio` / `--negative-seed` | 每个 split 的负样本比例与种子 | D-015 |
| `--mask-other-view` | 是否启用"双视图均遮蔽"口径 | D-016（M5 前定） |
| `--max-link-rows` / `--max-genes` | **仅冒烟**用截断；`--max-genes` 会在 manifest `warnings` 里留痕 | — |
| `--dataset-name` / `--out` | 数据集名与输出目录 | — |

性能提示（未在真实规模上实测）：构建过程会对 STRING links 文件做 **4 次流式扫描**（每个通道各 1 次选基因 + 1 次建图），
所以先用冒烟构建确认路径/阈值/ID 都对，再跑全量。

### 3.4 数据校验（三条命令，互相独立）

**A. 结构与防泄漏不变量**（加载 + `validate()`，含 val/test 正边不在编码图、负样本不是已知正边、下游目标不跨 split）：

```bash
python - <<'PY'
import json
from pathlib import Path

from cofi_gene.dataset import CoFiDataset

path = Path("data/processed/real_v1")
dataset = CoFiDataset.load(path)
summary = dataset.validate()                      # 违反任一不变量会直接抛 ValueError
print(json.dumps(summary, indent=2, sort_keys=True))

meta = json.loads((path / "dataset.json").read_text(encoding="utf-8"))["meta"]
print(json.dumps({"split": meta["split"], "channel_columns": meta["channel_columns"]},
                 indent=2, sort_keys=True))
PY
```

**B. 训练侧自检**（读配置 + 加载数据集 + `validate()`，不训练、不用 GPU）：

```bash
python -m cofi_gene.train --config configs/b3.yaml \
    --dataset data/processed/real_v1 --dry-run | tail -40
```

**C. manifest 关键字段**（把数字填进 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) 第 3 节）：

```bash
python -c "import json; m=json.load(open('data/processed/real_v1/manifest.json')); \
print('git', m['git_commit']); print(json.dumps(m['processed'], indent=2)); \
print(json.dumps(m['report']['gene_selection'], indent=2)); \
print(json.dumps(m['report']['overlap'], indent=2)); \
print(json.dumps(m['report']['downstream'], indent=2)[:800])"
```

`manifest.json` 顶层字段（已实测）：`dataset_name` / `created_at` / `git_commit` / `versions` / `command` /
`config` / `split_decisions` / `inputs`（每个输入文件的 `sha256` + 字节数）/ `processed`
（`dataset_npz_sha256`、`X_sha256`、`num_nodes`、`num_features`）/ `report`
（`alignment`、`channels`、`overlap`、`splits`、`features`、`downstream`、`warnings`）。

---

## 4. 训练（B0–B7，当前可用；Week 5 / M5 见 §4.5）

### 4.1 首轮三条命令（真实数据 + GPU）

```bash
# B3：双图 GCN + 拼接投影（首轮基线，先跑这个）
python -m cofi_gene.train --config configs/b3.yaml \
    --dataset data/processed/real_v1 --device cuda --seed 42 \
    --out-dir runs/real_v1 --run-name b3_real_v1_seed42

# B4：门控融合
python -m cofi_gene.train --config configs/b4.yaml \
    --dataset data/processed/real_v1 --device cuda --seed 42 \
    --out-dir runs/real_v1 --run-name b4_real_v1_seed42

# B5：跨图注意力
python -m cofi_gene.train --config configs/b5.yaml \
    --dataset data/processed/real_v1 --device cuda --seed 42 \
    --out-dir runs/real_v1 --run-name b5_real_v1_seed42

# B6：B5 + 单轮边权反馈（Attention 路线；参数与 B5 相同，但每步两次编码）
python -m cofi_gene.train --config configs/b6.yaml \
    --dataset data/processed/real_v1 --device cuda --seed 42 \
    --out-dir runs/real_v1 --run-name b6_real_v1_seed42

# B7：B4 + 单轮边权反馈（Gate 路线；参数与 B4 相同）
python -m cofi_gene.train --config configs/b7.yaml \
    --dataset data/processed/real_v1 --device cuda --seed 42 \
    --out-dir runs/real_v1 --run-name b7_real_v1_seed42
```

补齐基线时同理替换 `configs/b0.yaml` / `b1.yaml` / `b2.yaml`（`b1` 只用图 A，`b2` 只用图 B，`b0` 无消息传递）。

### 4.2 CLI 参数（`python -m cofi_gene.train --help`，已核对）

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `--config` | **必填**，模型 YAML（`base:` 继承 `configs/_base.yaml`） | 未知键会直接报错，不会被静默忽略 |
| `--dataset` | 覆盖 `data.processed_dir` | 指向 **processed 目录**（含 `dataset.json`），不是 `data/raw` |
| `--device` | 覆盖 `runtime.device` | 默认是 `cpu`；**GPU 必须显式写 `--device cuda`** |
| `--seed` | 覆盖 `runtime.seed` | 首轮统一 42 |
| `--out-dir` | 覆盖 `runtime.out_dir`（默认 `runs`） | 产物根目录 |
| `--run-name` | 覆盖 `runtime.run_name` | 不给则自动 `<model>_seed<seed>_<config hash12>`，如 `b3_seed42_f2ebde49513f` |
| `--epochs` | 覆盖 `optim.epochs` | 若改，B3/B4/B5 **一起改**并记录（否则消融不可比） |
| `--dry-run` | 只打印 resolved config + 数据集摘要后退出 | 用于校验，不训练 |
| `--skip-test` | 完全不评测 test split（`metrics.json` 中 `test: null`） | 仅冒烟用；正式首轮**不要**用 |

### 4.3 必须保持一致的东西（否则结果不可比）

- 训练预算来自 `configs/_base.yaml`，当前是**占位值**（D-018 未锁定）：
  `Adam`、`lr 0.01`、`weight_decay 0.0`、`epochs 200`、`eval_every 10`、`early_stop_patience 50`、`grad_clip 5.0`、
  `dropout 0.2`、`hidden_dim 128`、`num_layers 2`、`fusion_dim 128`、`edge_head dot`。
- resolved config 哈希（当前提交实测）：`b0 bd7bb75ec0f2`、`b1 f3160ec62e2d`、`b2 0ea7242eb7f9`、
  `b3 f2ebde49513f`、`b4 eb0a3fe0c5a8`、`b5 29ee907107e4`。哈希变化 = 预算或结构被改过，必须在交接记录里写明原因。
- 负样本来自数据集构建阶段（构建期固定），**不要在训练里重采样**；`optim.negatives_ratio` 只被校验、当前不影响训练（§7）。
- 同一份 `--dataset` 目录 → 同一划分、同一负样本；换数据集必须整套模型重跑。
- 参数量以本次 run 的 `metrics.json → model.parameter_count` 为准（合成 16 维特征时为
  B0 2,176 ｜ B1/B2 18,688 ｜ **B3 = B4 = 70,272** ｜ B5 169,344）。

### 4.4 建议的运行方式（记录时间与显存）

训练器**不会**打印逐 epoch 日志，也**不会**记录耗时/显存（§7），所以外部包一层：

```bash
mkdir -p runs/logs
/usr/bin/time -v python -m cofi_gene.train --config configs/b3.yaml \
    --dataset data/processed/real_v1 --device cuda --seed 42 \
    --out-dir runs/real_v1 --run-name b3_real_v1_seed42 \
    2>&1 | tee runs/logs/b3_real_v1_seed42.log

# 另开一个终端记录显存峰值（训练期间）
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv -l 5
```

### 4.5 Week 5 / M5：缺边鲁棒性 + Gene ID 错位对照（本轮主任务）

代码已在 `main` 提供，**默认关闭**，必须显式开启。协议说明见 `docs/WEEK5_IMPLEMENTATION_REPORT.md`。

```bash
# 0) 先确认代码/数据/预算没跑偏
git rev-parse HEAD                      # 应为 f1c95c1 或其后
python -m pytest -q                     # 期望 218 passed
python -m cofi_gene.train --config configs/b3.yaml \
    --dataset data/processed/real_v1 --dry-run | tail -20

# 1) 主曲线：B0–B7 × {0, 0.1, 0.3, 0.5}（每格 = 一次完整训练）
python scripts/run_robustness.py --dataset data/processed/real_v1 \
    --device cuda --seed 42 --out-dir runs/m5_robustness --summary-name seed42

# 1b) 只想先看关键对照（成本低，先出图）——注意 run_name 与上面相同，
#     同一 (model, ratio, seed) 只会保留最后一次；两个 sweep 不必都跑
python scripts/run_robustness.py --dataset data/processed/real_v1 \
    --device cuda --seed 42 --models b1 b2 b4 b5 b6 b7 \
    --ratios 0 0.1 0.3 0.5 --out-dir runs/m5_robustness --summary-name key42

# 2) 实验三：Gene ID 错位对照（只允许 b3–b7；ratio 用 0 即可）
python scripts/run_robustness.py --dataset data/processed/real_v1 \
    --device cuda --seed 42 --models b3 b4 b5 b6 b7 --ratios 0 \
    --misalign-cross-view --out-dir runs/m5_robustness --summary-name misalign42

# 3) 可选：双视图遮蔽口径（D-016 未锁定前只作补充实验，不进主表）
python scripts/run_robustness.py --dataset data/processed/real_v1 \
    --device cuda --seed 42 --models b1 b4 b5 --ratios 0.3 \
    --mask-other-view --out-dir runs/m5_robustness --summary-name maskother42

# 4) 单格排查也可以直接用训练 CLI
python -m cofi_gene.train --config configs/b5.yaml \
    --dataset data/processed/real_v1 --device cuda --seed 42 \
    --out-dir runs/m5_robustness --run-name b5_drop30_seed42 \
    --edge-drop-fraction 0.3
```

协议要点（细节见 `docs/WEEK5_IMPLEMENTATION_REPORT.md`、`docs/DECISIONS.md` D-024/D-025/D-026）：

- 只删每个通道的 **`encode_edges`（训练/编码图）**；`val`/`test` 正边与构建期固定负样本不参与抽样，
  因此 0/10/30/50% 四档共用**同一评估集**（`n_positives`/`n_negatives` 应完全相同）。
- 删边种子只依赖 `(run seed, 通道序号)`、**与比例无关** → 10% ⊂ 30% ⊂ 50%，
  曲线反映信息损失而不是重新抽样；每个比例都是**重训**，不是评测期删边。
- 错位对照是"融合前对单个视图做固定 derangement（无不动点）"，编码器/监督/评估边不变。
- 开启任一扰动时 `metrics.json` 会多一个 `perturbation` 块（每通道 kept/removed 计数、
  错位的 `permutation_sha256`）；`dataset.dataset_sha256` 仍指向未扰动数据集，属预期。

M5 验收（缺一不算完成）：

- [ ] `runs/m5_robustness/seed42.json` 存在，`dataset.sha256` 与首轮一致（`fee691ce…`）
- [ ] 每个 ratio 的 `test.*.n_positives` / `n_negatives` 与 ratio=0 完全相同
- [ ] `perturbation.channels.*.num_encode_edges_removed > 0`；ratio=0 的 run 不应出现 `perturbation` 块
- [ ] 结果文档按 `docs/M5_ROBUSTNESS_RESULTS_TEMPLATE.md` 填好，push 到 `docs/M5_ROBUSTNESS_RESULTS_<日期>.md`
- [ ] 若某格 OOM 或 `best_epoch == 200`，在结果文档"异常与限制"里写原文，不要静默丢弃


---

## 5. 查看训练结果

每个 run 在 `runs/<out_dir>/<run_name>/` 下写三个文件（已实测）：

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | `run_name`、`created_at`、`git_commit`、`versions`、完整 `config`、`model`（`name`/`channels`/`parameter_count`/`output_dim`/`fusion_kind`/`edge_heads`）、`dataset`（`processed_dir`/`dataset_sha256`/`num_nodes`/`num_features`/`meta`）、`best_epoch`、`best_val_mean_ap`、`val`、`test`、`history`（逐 epoch `train_loss` 与 `val_mean_ap`）、`notes` |
| `config.json` | 完全解析后的配置（含 `base:` 展开与 CLI 覆盖） |
| `model.pt` | `{"state_dict", "config", "model_name", "seed"}` |

```bash
ls runs/real_v1/b3_real_v1_seed42

# 概览
python -m json.tool runs/real_v1/b3_real_v1_seed42/metrics.json | head -60

# 训练曲线（逐 epoch）+ 各通道 val/test AUPRC
python -c "import json; m=json.load(open('runs/real_v1/b3_real_v1_seed42/metrics.json')); \
print('run', m['run_name'], 'best_epoch', m['best_epoch'], 'val_mean_ap', round(m['best_val_mean_ap'], 4)); \
[print(r['epoch'], round(r['train_loss'], 4), r.get('val_mean_ap')) for r in m['history']]; \
print('val', {k: round(v['average_precision'], 4) for k, v in (m['val'] or {}).items()}); \
print('test', None if m['test'] is None else {k: round(v['average_precision'], 4) for k, v in m['test'].items()})"
```

指标字段名（`cofi_gene.metrics.binary_summary`）：`average_precision`（= AUPRC，主指标）、`roc_auc`（辅）、
`positive_rate`、`n`，评估时另附 `n_positives` / `n_negatives`。**报数字时必须同时给阳性率**，否则 AUPRC 无法与基线比较。

CLI 结束时也会打印一行摘要：

```
[cofi_gene.train] run=... best_epoch=... val_mean_ap=... test_ap={"A": ..., "B": ...}
```

`test` 用**最佳 val checkpoint 评测一次**；任何"先看 test 再调参"的做法都违规（§8）。

---

## 6. 常见失败与排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'cofi_gene'` | 没装包 / 不在仓库根目录执行 | `python -m pip install -e '.[dev,data]'`，并在仓库根目录运行 |
| `[cofi_gene.train] training needs torch: ...`（退出码 2） | 未装 torch | 按该机 CUDA 装 torch（`.[train]` 或对应 wheel） |
| `RuntimeError: device 'cuda' requested but CUDA is not available` | CPU 版 torch 或驱动不可用 | 先跑 `scripts/check_environment.py`，看 `torch_cuda_build` 是否为 `null` |
| `FileNotFoundError: .../dataset.json` | `--dataset` 指错目录（指向 raw 或父目录） | 指向 `data/processed/<name>/`，须含 `dataset.npz` + `dataset.json` |
| `ValueError: unknown config key(s)`（`build_dataset.py`） | `configs/data.yaml` 有 `PipelineConfig` 不认识的键 | 只用 §3.3 表里的键；键名以 `src/cofi_gene/pipeline.py::PipelineConfig` 为准 |
| `ValueError: unknown key(s) in model/optim/data/runtime`（训练） | 训练 YAML 键名错误 | 允许键见 `src/cofi_gene/config.py::_ALLOWED` |
| `ValueError: n_components=128 exceeds min(N, F)=...` | 节点数太少（冒烟数据）或原始特征维数太小 | 冒烟时调小 `--n-components`，或跑全量数据 |
| `ValueError: fewer than 3 genes carry evidence above threshold` | 阈值过高 / ID 映射对不上 | 检查 D-005 阈值与 D-009 映射；先看 `--dry-run` 的 `gene_selection` |
| `ValueError: gene order has N rows but feature matrix has M` | `gene_text.ids.txt` 与矩阵行数不一致 | 重新导出严格对齐的 id 顺序文件 |
| `ValueError: gene order collapses to duplicate symbols after normalisation` | id 文件里大小写/版本号重复 | 去重或改用稳定基因 ID，并把规则记入 D-009 |
| `ValueError: channel A: N held-out positive edge(s) are present in the encoder graph` | 数据集被手工改过或绕开了构建逻辑 | **重建数据集**；不得删检查、不得改 npz 绕过 |
| `ValueError: channel A neg_train: N negative(s) are known positives` | 同上（负样本与正边冲突） | 重建数据集 |
| `NotImplementedError: model 'b6' is not implemented yet ...` | B6/B7 尚未实现（M4） | 预期行为，先跑 B0–B5 |
| `torch.cuda.OutOfMemoryError` | 全图训练、无 batch 参数，显存随节点/边数增长 | 记录报错与规模；当前**没有**分块/AMP 方案（§7），上报后再定 |
| PyG 在 sm_120 上装不上 | 上游无匹配 wheel | 不影响 B0–B5（D-013）；用非 strict 探针记录并注明 |
| 训练 loss 出现尖峰 | 小图上 `lr=0.01` 的已知现象（周报 W2-3），预算未锁定 | 记录现象，不要据此下模型结论；等 D-018 锁定 |


---

## 7. 已知缺口与限制（BLOCKED / TODO，别当成已具备）

| 项 | 状态 | 影响 | 现在的替代做法 |
| --- | --- | --- | --- |
| 逐 epoch 训练日志 | **TODO**：`runtime.log_every` 在 `config.py` 里被允许但 `train.py` **从未使用**；训练循环不打印 | 看不到实时收敛过程 | stdout 只有最后一行摘要，用 `tee` 保存；曲线事后从 `metrics.json → history` 读 |
| 运行耗时 / 显存记录 | **TODO**：`metrics.json` 只有 `created_at`，无 wall-clock、无显存峰值 | P3 要求的耗时/显存需外部记录 | `/usr/bin/time -v` + `nvidia-smi -l 5`（§4.4） |
| AMP / 混合精度 / batch 或分块 | **TODO**：无 `--amp`、无 `batch_size`，每次前向都是全图 | 显存随节点/边数线性增长，5090 上未验证 | 记录规模与 OOM 现象，上报后再定方案 |
| 断点续训 | **TODO**：`model.pt` 只存最终 best state，无 optimizer/epoch 状态 | 中断需从头跑 | 单次跑完并保留日志 |
| 缺边鲁棒性（10/30/50%） | ✅ 已实现（Week 5 / M5，D-024）：`python scripts/run_robustness.py` 与 `--edge-drop-fraction` | 可按协议出曲线 | 见 §4.5；每个比例 = 一次完整训练，评估集固定 |
| Gene ID 错位对照 | ✅ 已实现（Week 5 / M5，D-026）：`--misalign-cross-view`，仅 b3–b7 | 可做跨图互补性验证 | 见 §4.5；记录 `permutation_sha256` |
| 下游 DE MLP（冻结 embedding） | **BLOCKED**（M6）：未实现；只有库接口 `model.encode()` / `fusion_representations()` | 暂无 AUPRC 下游指标 | 上游可先跑；下游等 M6 |
| B6 / B7（Graph Feedback） | ✅ 已实现（Week 4，D-022） | 每步两次编码，约 2× 前向成本 | 直接用 `configs/b6.yaml` / `configs/b7.yaml` |
| `optim.negatives_ratio` 生效 | **已知缺陷**（W2-1）：只被校验，训练负样本来自数据集构建期 | 改该键**没有任何效果** | 要改负样本比例必须重建数据集（`--negatives-ratio`） |
| 真实规模下的内存 | **未验证**：合成数据仅 60 节点；真实 3–5k 节点的邻接/边张量未实测 | 可能出现内存或显存瓶颈 | 先用冒烟构建估规模，再正式跑 |
| 真实数据 / GPU 结果 | **未验证**：仓库内无任何真实数据 manifest 或 GPU run | 无法引用任何真实指标 | 这就是本文档要解决的问题 |

---

## 8. 实验规范（红线，违反则结果作废）

1. **不得用隐藏测试边构建训练图。** 编码图默认等于 train split（`encode_edges`），
   `CoFiDataset.validate()` 会在加载/构建时抛错。不要手工改 `dataset.npz` 或删检查来"绕过去"。
2. **不得用 test 标签选超参。** 早停只看 val；test 只在最佳 val checkpoint 上评测一次（默认行为）。
   反复重跑 test 挑选结果 = 泄漏。
3. **未冻结的实验决策要先确认**（见 [docs/DECISIONS.md](docs/DECISIONS.md)）：
   P0 = D-005 / D-009 / D-014；P1 = D-011 / D-012 / D-015 / D-017 / D-018；Week 3 新增 D-019 / D-020 / D-021。
   未确认前沿用"当前默认"，且**不得把默认值写成结论**。
4. **不得提交真实数据、模型权重或敏感路径。** `data/`、`runs/` 已在 `.gitignore`。
   归档 `manifest.json` 前确认里面没有绝对路径/账号信息（`versions.executable` 会记录解释器路径）。
5. **不得把 CPU 单元测试通过等同于 GPU 训练成功。** `pytest -q → 218 passed` 只说明代码逻辑与合成数据链路正确。
6. **不得把合成数据结果描述为真实数据结果。** 现有 `runs/week3_smoke/`、`runs/branch_merge_smoke/` 全是合成数据冒烟。
7. **单图模型只评自己的图**：B1 只评通道 A，B2 只评通道 B；不要把单图模型拿去评未使用的另一张图。
8. **报 AUPRC 必须同时给阳性率与负采样口径**（`positive_rate`、`n_positives` / `n_negatives`，以及负样本来自构建期固定抽样的说明），否则数字无法比较。
9. **同一批对比必须共用同一数据集目录与同一训练预算**；改 epochs / 数据集要整套重跑。

---

## 9. 交接验收表

图例：【必须】= 第一波训练完成的必要条件；【可选】= 时间允许再做；【阻塞】= 依赖未实现的功能，先记录不硬做。

### P0 同步现状（【必须】）

- [ ] 【必须】贴回服务器 `git rev-parse HEAD` 与是否与 `main` 一致
- [ ] 【必须】贴回 `check_environment.py` 的 JSON（可先非 strict）
- [ ] 【必须】列出已有原始数据/处理数据路径与版本、大小
- [ ] 【必须】列出已有 `runs/` 目录与对应 `metrics.json` 的 `run_name` / `git_commit` / `best_val_mean_ap`
- [ ] 【必须】写明当前阻塞点（1–3 条，具体到报错或缺失项）
- [ ] 【可选】把上述内容整理成一段交接说明发给 coding 同学，用于更新 `docs/STATUS.md`

### P1 环境与数据（【必须】）

- [ ] 【必须】`python scripts/check_environment.py --strict` 退出码 0，输出留档（PyG 无法安装时：保留非 strict 输出 + 书面说明，见下一条）
- [ ] 【必须】若 PyG 装不上：保留非 strict 输出并注明，不得为此阻塞训练
- [ ] 【必须】`python -m pytest -q` → 218 passed（记录实际数字与机器）
- [ ] 【必须】真实 STRING 两文件 + `gene_text` 矩阵与 id 顺序文件 + 至少一个细胞系 DE parquet 到位
- [ ] 【必须】`build_dataset.py --dry-run` 输出中 `genes` / 两通道 `num_edges` / `overlap.jaccard` / `splits` 合理
- [ ] 【必须】正式构建产出 `data/processed/<name>/{dataset.npz, dataset.json, manifest.json}`
- [ ] 【必须】§3.4-A 的 `validate()` 通过（无泄漏、负样本无冲突）
- [ ] 【必须】把基因数/边数/重合率/覆盖率填入 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) 第 3 节
- [ ] 【可选】四个细胞系 DE 全覆盖；记录各自阳性率与目标覆盖率
- [ ] 【可选】核对 `gene_text` 原始维数确为 3072

### P2 第一波真实训练（【必须】）

- [ ] 【必须】B3 在真实数据 + GPU 上跑完，产出 `metrics.json` / `config.json` / `model.pt` 与日志
- [ ] 【必须】B4、B5 用**同一数据集、同一 seed(42)、同一预算**跑完
- [ ] 【必须】B0–B5 的 `metrics.json` 里 `dataset.dataset_sha256` 一致（证明同一份数据）
- [ ] 【可选】B0 / B1 / B2 补齐
- [ ] 【可选】同配置下重复一次 B3，确认可复现（loss/指标一致）
- [ ] 【阻塞】B6 / B7：未实现（M4），不阻塞首轮
- [ ] 【阻塞】多种子（3 seeds）：等 D-018 预算锁定后再做，否则重跑

### P2b M5 鲁棒性与错位对照（【必须】，2026-11-22 起）

- [ ] 【必须】训练分支已同步 `origin/main`（含 M5 代码），`python -m pytest -q` → 218 passed
- [ ] 【必须】`python scripts/run_robustness.py ... --summary-name seed42` 主曲线跑完（B0–B7 × 0/0.1/0.3/0.5）
- [ ] 【必须】Gene ID 错位对照（`--misalign-cross-view`，b3–b7，ratio 0）
- [ ] 【必须】`summary.json` 的 `dataset.sha256` 与首轮一致（`fee691ce…`）
- [ ] 【必须】每个 ratio 的 `n_positives` / `n_negatives` 与 ratio=0 完全一致（同一评估集）
- [ ] 【必须】按 `docs/M5_ROBUSTNESS_RESULTS_TEMPLATE.md` 写 `docs/M5_ROBUSTNESS_RESULTS_<日期>.md` 并 push
- [ ] 【可选】D-016 双视图遮蔽补充实验（`--mask-other-view`，不进主表）
- [ ] 【必须】写明每格耗时 / 显存、是否命中 200 epoch 上限、任何 OOM 原文

### P3 结果交付（【必须】）

- [ ] 【必须】每个 run：commit + 完整命令 + `config.json` + 环境信息 + 日志路径
- [ ] 【必须】指标表：每通道 `average_precision`（AUPRC）+ `roc_auc` + `positive_rate` + 样本数
- [ ] 【必须】耗时与显存峰值（外部记录）
- [ ] 【必须】失败案例的完整 traceback（如有）
- [ ] 【必须】≤1 页首轮总结（含异常现象与下一步）
- [ ] 【可选】把 `metrics.json` 摘要（不含权重/数据）归档到仓库外或 `docs/` 下的表格（提交前检查无敏感信息）

---

## 10. 第一波训练的最低交付标准（三条齐备才算完成）

1. **真实数据 manifest**：`data/processed/<name>/manifest.json` 存在，且包含
   `inputs[*].sha256`、`processed.num_nodes`/`num_features`/`dataset_npz_sha256`、
   `report.gene_selection`、`report.channels`、`report.overlap`、`report.splits`、`report.downstream`；
   配套 `dataset.npz` + `dataset.json` 能通过 §3.4-A 的 `validate()`。
2. **GPU 环境验证**：`check_environment.py --strict` 退出码 0 的输出（或 PyG 缺失时的非 strict 输出 + 说明），
   含 GPU 型号、`compute_capability`、显存、torch/CUDA 版本。
3. **至少一个真实数据 Baseline 的训练日志与配置**：优先 B3，`runs/<out>/<run>/{metrics.json, config.json, model.pt}` +
   `run.log`，且 `metrics.json` 中 `git_commit`、`dataset.dataset_sha256`、`val`/`test` 的
   `average_precision` 与 `positive_rate` 齐全。

> 三项缺一，都不能在 `docs/STATUS.md` 里写成"真实数据已验证"。

---

## 11. 相关文档

| 文档 | 用途 |
| --- | --- |
| [README.md](README.md) | 快速开始与代码结构 |
| [AGENTS.md](AGENTS.md) | 研究范围、B0–B7 设计、Feedback 规则（唯一基准） |
| [docs/STATUS.md](docs/STATUS.md) | 最新进度、四层状态区分、下一步（事实来源） |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 已锁定 / 待确认决策（P0–P2 优先级表） |
| [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) | 数据字段、划分协议、manifest 格式、数据清单表格 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | M0–M8 里程碑与训练交付合同 |
| [docs/WEEK3_IMPLEMENTATION_REPORT.md](docs/WEEK3_IMPLEMENTATION_REPORT.md) | B4/B5 实现细节、张量形状、参数统计、已知限制 |
| [docs/WEEK5_IMPLEMENTATION_REPORT.md](docs/WEEK5_IMPLEMENTATION_REPORT.md) | **Week 5 / M5**：删边 / 遮蔽 / 错位协议、CLI、验收与风险 |
| [docs/M5_ROBUSTNESS_RESULTS_TEMPLATE.md](docs/M5_ROBUSTNESS_RESULTS_TEMPLATE.md) | **你要填的结果模板**（复制成 `M5_ROBUSTNESS_RESULTS_<日期>.md`） |
| [docs/ALL_MODELS_RESULTS_2026-11-19.md](docs/ALL_MODELS_RESULTS_2026-11-19.md) | 首轮真实数据 B0–B7 上游结果（seed 42，RTX 4090） |
| [docs/TEAM_HANDOFF.md](docs/TEAM_HANDOFF.md) | Week 2 视角的队友交接（数据含义/待确认项） |
| [docs/HANDOFF.md](docs/HANDOFF.md) | 新对话/本地 Agent 的接手协议 |

