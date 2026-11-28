# M5.1 服务器训练指南（Evaluation Audit & Protocol Enhancement）

面向：**真实数据 + GPU 训练负责人**。按本文件从上到下执行即可；不需要阅读或修改源码。
所有命令都写在仓库根目录执行，且已在本机用合成数据验证过参数与入口（真实数据执行待服务器确认）。
**研究诚信**：本指南只增加更严格的协议与检查，不改变 B0–B7 的定义，也不以"跑出更高指标"为目标。

---

## 第一部分：本次更新说明

| 项 | 内容 |
| --- | --- |
| 里程碑 | **M5.1**：修复评估缺陷 + 新增可验证协议（不是新模型、不是新研究目标） |
| 修复的缺陷（P0） | ① 负样本在 train/val/test 之间可能重复；② `CoFiDataset.validate()` 覆盖不足；③ 缺边只删编码图、训练正例未同步；④ 跨视图遮蔽有两份实现；⑤ 训练入口无审计/无完成标志 |
| 新增的科研协议（P1） | 严格双视图遮蔽 `dual_view_strict_holdout`、困难负样本 `degree_matched` / `two_hop`、监督感知删边 `supervision_aware`、非神经网络基线（PA/CN/AA/RA/余弦） |
| 旧结果怎么用 | 历史 M5/M6 数值 = **原始评估协议下的初步结果**，需用修复后的协议复核；旧 checkpoint 只可用于诊断，不能作为最终结论 |
| 训练用代码版本 | `main` = `c58b69c`（M5.1 + Week 7 已合并并推送；`codex/m5.1-evaluation-audit` 同 commit） |
| 训练用数据版本 | **必须新建** `data/processed/real_v1_1`（负样本互斥修复版）；**不要覆盖** `real_v1` |
| 不改变的部分 | B0–B7 模型定义、两图通道语义、节点顺序、128 维特征、下游目标划分规则、历史文件 |

**一句判断标准**：数据集审计 PASS + 训练 run 里有 `audit`/`protocol` 块 + `status=complete`，这条结果才可用。

---

## 第二部分：职责与前置条件

### 你做（训练负责人）

1. 同步指定分支/commit；2. 准备真实原始数据并**新建** `real_v1_1`；3. 跑数据审计并回报结果；
4. 按 Stage 0→5 顺序训练；5. 收集 `audit.json` / `metrics.json` / `model.pt` / 日志 / 汇总表格并回传。

### 你不做（由本地开发负责）

- 不修改 `src/cofi_gene/**`、`scripts/**`、`configs/b*.yaml`；
- 不为某一模型单独调 epoch/lr/负样本比例；
- 不删除或覆盖 `data/processed/real_v1`、`runs/**` 历史目录；
- 不手工编辑 `metrics.json` / `audit.json`。

### 前置检查

| 检查 | 要求 |
| --- | --- |
| Python | 3.10+（仓库 CI 用 3.11） |
| torch | 与服务器 CUDA 匹配的 wheel；上游训练不使用 PyG（D-013），**不需要**安装 PyG |
| 磁盘 | `real_v1_1` 约与 `real_v1` 同量级（`dataset.npz` 数 GB），请预留空间 |
| 原始数据 | STRING v12.0 `9606.protein.links.detailed.v12.0.txt.gz`、`9606.protein.info.v12.0.txt`、PIE `gene_text.npy` + `gene_text.ids.txt`、四个 `de/<cell>.parquet` |
| 已有结果 | `runs/`、`data/processed/real_v1` **只读**，新实验一律写新目录 |

---

## 第三部分：代码同步

代码已推送到 GitHub：`main` = `c58b69c`（M5.1 + Week 7 已合并）。任选一种同步方式：

```bash
# 方式 A（推荐）：本地开发把分支推送到 GitHub 后，服务器直接拉取
git fetch origin
git checkout main && git pull --ff-only
git log --oneline -1                     # 记下 commit sha，回报时需要

# 方式 B：用 patch / git bundle 传输（开发方提供），在服务器上应用
# git bundle verify m5_1.bundle && git fetch m5_1.bundle main
```

同步后必须确认（把输出贴回）：

```bash
git status -sb                 # 必须没有未提交改动
git log --oneline -3
python -m pytest -q            # 期望 300 passed（缺 torch 时部分用例自动 skip）
```

**保护已有结果**：不要 `git reset --hard`、不要 `git clean -fdx`、不要在 `runs/` 或
`data/processed/` 上执行任何删除命令。

**版本一致性核对**：`pip show cofi-gene | grep -E 'Version|Location'` 应指向本仓库的 editable 安装；
`python -c "import cofi_gene, inspect; print(cofi_gene.__file__)"` 必须落在本仓库 `src/` 内。

---

## 第四部分：环境准备

```bash
cd <仓库根目录>
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,data]'                 # 数据管线 + 测试（含 pyarrow）

# torch 按服务器 CUDA 版本选择（示例：CUDA 12.1 wheel）
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

python scripts/check_environment.py                    # 打印 python/torch/CUDA 与 tensor 冒烟
# 注意：--strict 还要求 PyG；本仓库上游不需要 PyG，不要用 --strict 作为门槛
python -m pytest -q                                    # 确认 300 passed
```

环境信息回报清单：`python -V`、`torch.__version__`、`torch.version.cuda`、GPU 型号与显存、
`nvidia-smi` 摘要（**不要**贴 token / ssh key / 账号密码）。

---

## 第五部分：数据构建

### 5.1 确认 `configs/data.yaml` 指向真实文件

```bash
python - <<'PY'
import pathlib, yaml
cfg = yaml.safe_load(open("configs/data.yaml", encoding="utf-8"))
for key in ("string_links", "string_info", "gene_text", "gene_order"):
    print(f"{key:14s}", pathlib.Path(cfg[key]).exists(), cfg[key])
for ctx, path in cfg["de_tables"].items():
    print(f"de {ctx:8s}", pathlib.Path(path).exists(), path)
print("negatives_strategy:", cfg["negatives_strategy"])
PY
```

任何一行 `False` → 先修正路径再继续（绝对路径或相对仓库根路径都可以）。

### 5.2 新建 `real_v1_1`（**不要**覆盖 `real_v1`）

```bash
python scripts/build_dataset.py --config configs/data.yaml \
  --dataset-name real_v1_1 --out data/processed/real_v1_1
```

新旧数据的区别（只有一处）：

| 项 | `real_v1`（历史） | `real_v1_1`（M5.1） |
| --- | --- | --- |
| 正边划分 | 不变 | **不变**（同 `split_seed`、同比例） |
| train 负样本 | 历史抽样 | **不变**（同样的种子派生方式与 forbidden 语义） |
| val/test 负样本 | 可能与 train 重复 | 改为避开 train/val 已用负边，三者两两互斥 |
| 节点/特征/边 | 不变 | 不变（同一 `n_components`、同一 STRING 阈值） |

如果 `dataset.json` 的 `meta.split.negatives_strategy` 不是 `uniform_random`，请回报（说明用了别的模板）。

### 5.3 数据审计（**必须 PASS 才能训练**）

```bash
mkdir -p runs/m5_1/audit
python scripts/audit_dataset.py --dataset data/processed/real_v1_1 \
  --out runs/m5_1/audit/real_v1_1_transfer.json
python scripts/audit_dataset.py --dataset data/processed/real_v1_1 \
  --protocol dual_view_strict_holdout --out runs/m5_1/audit/real_v1_1_strict.json
```

| 退出码 | 含义 | 处理 |
| --- | --- | --- |
| 0 | 全部检查 PASS（可能有 WARN） | 继续 |
| 1 | 至少一项 FAIL | **停止**，回传 `.json` + `.md`，不要训练 |
| 2 | 数据集无法加载 | 检查路径/版本并回报 |

审计会打印每条通道的 `num_val / num_test / visible_in_other_encode`，请把这两行贴回：
Protocol A 下该数字**不应为 0**（跨证据迁移），Protocol B 下必须为 0。

### 5.4 记录数据清单

```bash
python - <<'PY'
import hashlib, json, pathlib
d = pathlib.Path("data/processed/real_v1_1")
manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
actual = hashlib.sha256((d / "dataset.npz").read_bytes()).hexdigest()
print("dataset.npz sha256:", actual)
print("manifest sha256  :", manifest["processed"]["dataset_npz_sha256"])
print("match            :", actual == manifest["processed"]["dataset_npz_sha256"])
print("split decisions  :", meta["meta"]["split"])
print("negative protocol:", {k: v["strategy"]
                             for k, v in meta["meta"]["negative_sampling"].items()})
PY
```

---

## 第六部分：正式训练（Stage 0 → 5）

所有阶段走同一个入口；先 `--dry-run` 看计划，再真正运行。每阶段开始前把 `git log -1` 贴回。

```bash
# 通用：把 --device 换成 cuda；CPU 只用于冒烟
STAGE_ARGS="--device cuda --audit strict"
```

### Stage 0：环境 + 数据审计（不训练）

```bash
python scripts/run_experiments.py --stage 0 --dataset data/processed/real_v1_1 \
  --out-dir runs/m5_1/stage0
```

产物：`runs/m5_1/stage0/audit.json` 与 `audit.md`；退出码非 0 时**不要**进入 Stage 1。

### Stage 1：修复后基础结果（B0–B7，Protocol A，0% 删边，seed 42）

```bash
python scripts/run_experiments.py --stage 1 --dataset data/processed/real_v1_1 \
  --out-dir runs/m5_1/stage1 $STAGE_ARGS --resume
```

- 8 格：`s1_<model>_xfer_drop00_adj_seed42`，每格 = 一次完整训练；
- 用途：修复后的基础表现，并检查历史 M5 结论是否仍成立；
- `--resume` 可中断后继续（只跳过 `status=complete` 的格子）。

### Stage 2：严格遮蔽诊断（B1/B2/B4/B5，Protocol B，0% 删边）

```bash
python scripts/run_experiments.py --stage 2 --dataset data/processed/real_v1_1 \
  --out-dir runs/m5_1/stage2 $STAGE_ARGS --resume

# 同协议的非神经网络基线（同一候选集，可与模型行并列）
python scripts/run_baselines.py --dataset data/processed/real_v1_1 \
  --protocol dual_view_strict_holdout \
  --out-dir runs/m5_1/stage2/baselines --summary-name stage2_baselines
```

- 4 格：`s2_<model>_strict_drop00_adj_seed42`；
- 目的：去掉"另一视图直接可见"的信息后，双图方法还剩多少优势；
- `audit.json` 的 `A02-strict-holdout` 写明每个通道牺牲了多少训练正例，请一并回报。

### Stage 3：困难负样本诊断（冻结 Stage 1 checkpoint，不重训）

```bash
# 3a) 构建困难负样本数据集（与 real_v1_1 只差负样本）
python scripts/build_dataset.py --config configs/data.yaml \
  --dataset-name real_v1_1_deg --out data/processed/real_v1_1_deg \
  --negatives-strategy degree_matched
python scripts/audit_dataset.py --dataset data/processed/real_v1_1_deg \
  --out runs/m5_1/audit/real_v1_1_deg.json

# 3b) 冻结评估 + 基线
python scripts/run_experiments.py --stage 3 --dataset data/processed/real_v1_1_deg \
  --frozen-from runs/m5_1/stage1 --out-dir runs/m5_1/stage3 \
  --device cuda --audit strict --resume
python scripts/run_baselines.py --dataset data/processed/real_v1_1_deg \
  --out-dir runs/m5_1/stage3/baselines --summary-name stage3_baselines
```

- 8 格：`s3_<model>_xfer_drop00_adj_seed42/eval.json`，复用 Stage 1 的 `model.pt`（**不训练**）；
- 缺 checkpoint 时该格记 `missing` 并返回码 1，回传 `stage3_summary.json`。

### Stage 4：监督感知删边（B1/B2/B4/B5/B6/B7，0% 与 50%）

```bash
python scripts/run_experiments.py --stage 4 --dataset data/processed/real_v1_1 \
  --out-dir runs/m5_1/stage4 $STAGE_ARGS --resume
```

- 12 格：`s4_<model>_xfer_drop{00,50}_sup_seed42`；
- 0% 格是严格 no-op（同模型应与 Stage 1 结果一致，可用于验证环境）；
- 50% 格同时删除编码边与训练正例，`metrics.json.protocol.edge_removal_mode=supervision_aware`；
- 只有该模式可称"训练关系缺失"；`adjacency`（M5 旧口径）只能称"邻接信息减少"。

### Stage 5：多随机种子确认（B0–B7 × seed 42/43/44）

```bash
python scripts/run_experiments.py --stage 5 --dataset data/processed/real_v1_1 \
  --out-dir runs/m5_1/stage5 $STAGE_ARGS --resume
```

- 24 格：`s5_<model>_xfer_drop00_adj_seed{42,43,44}`；
- 只改模型初始化/训练洗牌种子，**数据划分固定**；均值 ± 样本标准差见 `stage5_summary.md`；
- 若要研究数据划分敏感性，必须另建数据集并单列，不能与模型随机性混在一起解释。

### 单格重跑 / 过滤 / 续跑

```bash
# 只重跑某个失败格子（去掉 --resume 会覆盖该格）
python scripts/run_experiments.py --stage 1 --dataset data/processed/real_v1_1 \
  --only-models b5 --out-dir runs/m5_1/stage1 --device cuda --audit strict

# 只看计划（不读数据集、不训练）
python scripts/run_experiments.py --stage 4 --dataset data/processed/real_v1_1 --dry-run

# 单条训练命令（调试用）
python -m cofi_gene.train --config configs/b5.yaml --dataset data/processed/real_v1_1 \
  --device cuda --seed 42 --visibility-protocol cross_view_transfer \
  --edge-removal-mode supervision_aware --edge-drop-fraction 0.5 \
  --audit strict --out-dir runs/manual --run-name manual_b5_check
```

---

## 第七部分：新增评估协议说明

| 协议 | 允许看到的信息 | 禁止的信息 | 结论怎么读 |
| --- | --- | --- | --- |
| `cross_view_transfer`（Protocol A，Stage 1/4/5） | 本通道训练边；另一来源中已存在的同一基因对证据 | 本通道 val/test 正边进入编码或训练正例 | 描述为"跨证据迁移/补全"，**不是**完全未知关系的恢复；必须同时报告 `visible_in_other_encode` 比例 |
| `dual_view_strict_holdout`（Protocol B，Stage 2） | 两通道中不在任一通道 held-out 并集里的训练边 | 任一通道 val/test 正边进入**任一**编码图或训练正例 | 才是"两视图都看不到正确关系"的严格口径；`A02` 会告诉你牺牲了多少训练正例 |
| `uniform_random`（历史负样本） | 训练图之外的随机非边 | 任何已知正边；与其他 split 重复的负边 | 与历史结果可比；AP 的随机基线 = 阳性率 |
| `degree_matched`（Stage 3） | 训练图上正例端点的度数 | 用含 val/test 的图算度数 | 消除"靠度数区分正负"的捷径；`fallback_rate` 高说明候选不足 |
| `two_hop`（可选） | 训练图二跳邻居 | 全图枚举 | 更难的局部结构负样本；候选不足会 fallback 并记录 |
| `adjacency` 删边（M5 旧口径） | —— | —— | 只能称"邻接信息减少"，训练监督仍完整 |
| `supervision_aware` 删边（Stage 4） | —— | 被删边继续作为训练正例 | 这才是"训练关系缺失"；报告需写 `num_train_positives_removed` |
| 非神经网络基线 | 训练图拓扑 / 共享特征 / 同一候选集 | checkpoint、测试边、完整图 | 用来判断高 AP 是否来自拓扑或采样特性 |

**指标口径**：主指标是 `average_precision`（AUPRC，`metrics.json.test.<channel>`）；
`roc_auc` 为辅。AP 必须与阳性率一起读（1:1 负样本时随机 AP ≈ 0.5）。不得因为某个协议下数字更低就换指标或换测试集。

---

## 第八部分：结果输出（你需要回传的文件）

每个 run 目录（`runs/m5_1/stage<N>/<run_name>/`）应包含：

| 文件 | 内容 | 是否必须回传 |
| --- | --- | --- |
| `metrics.json` | 完整记录：config、seed、dataset sha256、git commit、`audit`、`protocol`、`status=complete`、history、val/test | ✅ 全量 |
| `audit.json` | 该 run 启动时的数据集审计（44 项检查 + 可见性 + 哈希） | ✅ 全量 |
| `config.json` | 解析后的完整配置 | ✅ |
| `model.pt` | 最优验证 checkpoint（Stage 3 依赖它） | ✅（Stage 1 必须有） |
| `run.log` | 你的调度器日志（stdout/stderr） | ✅ |
| `eval.json`（仅 Stage 3） | 冻结评估结果 | ✅ |
| `stage<N>_summary.json` / `.md` | 阶段汇总（均值 ± 标准差、缺失格子） | ✅ 每阶段 |
| `baselines/*.json` / `.md` | 非神经网络基线 | ✅ 若该阶段要求 |
| `data/processed/real_v1_1/manifest.json` | 数据清单与哈希 | ✅ |
| `runs/m5_1/audit/real_v1_1_*.json` | 数据审计（transfer + strict） | ✅ |

回传时必须附：**代码 commit**、**数据版本目录名**、**每阶段退出码**、**失败格子列表及错误信息**、
GPU 型号与显存峰值（若已有记录机制；没有就写"未记录"，不要估算）。

---

## 第九部分：故障处理

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| `dataset audit FAILED (...)` / 退出码 1 | 数据泄漏或结构问题（`failed_ids` 指明是哪项） | **停止训练**，回传 `audit.json`；不要用 `--audit warn` 绕过 |
| 审计 `A04` FAIL / `C09` FAIL | 用了旧数据集（`real_v1`） | 用本指南 5.2 重建 `real_v1_1` |
| `A06-dataset-hash` FAIL | `--dataset` 路径下没有 `dataset.npz` | 检查路径；`ls data/processed/<name>` |
| `FileNotFoundError: ... dataset.json` | 数据集目录不完整 | 重新构建；不要手工拼目录 |
| 索引错误 / 模块路径不存在 | 代码版本与文档不一致 | `git log -1` 对齐本指南；重新 `pip install -e .` |
| `checkpoint has no config` 或加载权重报 size mismatch | checkpoint 来自旧版本/别的模型 | 用同 commit 重训 Stage 1；不要手工改 `model.pt` |
| CUDA OOM | batch/图规模 + 双图注意力 | 降低同卡并发（一格一卡），或换更大显存卡；**不要**为省显存改模型维度 |
| loss 变成 `nan` | 学习率/边权异常 | 记录该格 config 与日志回报；不要自行调 lr 后继续同一阶段 |
| `could not draw ... negatives` / `lower negatives_ratio` | 该图非边不足（负样本比例过高） | 回报；在 `configs/data.yaml` 里降低 `negatives_ratio` 重新构建（需团队确认） |
| 训练中途被中断 | 抢占/超时 | 用同样的命令加 `--resume` 续跑；已 `complete` 的格子会被跳过 |
| 输出目录冲突 | 重复使用同一 `--out-dir` 且没加 `--resume` | 会覆盖同名 run；请为每个阶段固定一个目录，或先改名备份 |
| 汇总里出现 `missing` 格子 | 该格失败或未跑 | 用 `--only-models <m>` 单格重跑，再看 `stage<N>_summary.json` |
| `--misalign-cross-view` 报错 | 只支持 B3–B7 | 该控制只对跨图模型有意义，B0–B2 不要加 |

---

## 第十部分：实验完成后的交接模板（直接填写回传）

```markdown
### M5.1 服务器执行回报

- 代码 commit：`____________`（`git log -1 --oneline` 输出）
- 分支：`main`（`c58b69c`）
- 环境：Python ___ / torch ___ / CUDA ___ / GPU ___（型号 + 显存）
- 数据版本：`data/processed/real_v1_1`
  - dataset.npz sha256：`____________`
  - 节点数 / 特征维：______ / ______
  - 负采样协议：`uniform_random`（如否，说明）
- 数据审计：`runs/m5_1/audit/real_v1_1_transfer.json` → 状态 ____（failed_ids: ____）
- 可见性（Protocol A）：A: val/total ____、visible_in_other_encode ____；B: ____ / ____
- 可见性（Protocol B）：严格遮蔽后残留可见度 ____（必须为 0）

#### 各阶段
| 阶段 | 计划格数 | 完成 | 失败/缺失格子 | 退出码 | 汇总文件 |
| --- | --- | --- | --- | --- | --- |
| 0 审计 | 1 | | | | `runs/m5_1/stage0/audit.json` |
| 1 基础 | 8 | | | | `stage1_summary.md` |
| 2 严格 | 4 | | | | `stage2_summary.md` |
| 3 困难负样本 | 8 | | | | `stage3_summary.md` |
| 4 监督感知删边 | 12 | | | | `stage4_summary.md` |
| 5 三种子 | 24 | | | | `stage5_summary.md` |

#### 关键数字（填 `stage<N>_summary.md` 的 mean ± std）
- A 通道 AP：B0 ___ / B1 ___ / B2 ___ / B3 ___ / B4 ___ / B5 ___ / B6 ___ / B7 ___
- B 通道 AP：同上
- 基线（同协议）：common_neighbors ___ / adamic_adar ___ / feature_cosine ___
- 阳性率基线：___

#### 异常与观察
- 失败格子与错误：______________________________________
- 最优 epoch 是否落在预算上限：__________________________
- 任何与历史 M5 结论不一致之处：__________________________

#### 回传清单
- [ ] 各阶段 `metrics.json` / `audit.json` / `config.json` / `model.pt` / 日志
- [ ] 各阶段 `stage<N>_summary.json` + `.md`
- [ ] 数据 `manifest.json` + 两个数据审计 JSON
- [ ] 失败实验记录（含完整错误栈）
```

> 未完成的事项请如实标注"未执行"，**不要**用估算值或其它阶段的数字代替。



