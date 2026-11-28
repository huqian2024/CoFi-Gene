# 项目状态（每次合并 PR 后更新）

最后更新：2026-11-28（**M5.1 评估审计与协议增强完成**：本地分支 `codex/m5.1-evaluation-audit`，4 个提交、40 文件、+4481/−175、`300 passed`；同时已合并队友 M5/M6 结果文档 `06c175a..4cabf59`）。
阶段：**M5.1 评估审计与协议增强已在本地分支完成（300 passed）；真实数据需用 `real_v1_1` 重训复核；M6 旧结果仅为初评**
详细报告：[`docs/M6_THREE_SEED_ANALYSIS_2026-11-26.md`](M6_THREE_SEED_ANALYSIS_2026-11-26.md)（M6 三种子 + 逐目标）｜[`docs/M6_DOWNSTREAM_RESULTS_2026-11-26.md`](M6_DOWNSTREAM_RESULTS_2026-11-26.md)（M6 首轮）｜[`docs/M5_ROBUSTNESS_RESULTS_2026-11-25.md`](M5_ROBUSTNESS_RESULTS_2026-11-25.md)（M5）｜[`docs/ALL_MODELS_RESULTS_2026-11-19.md`](ALL_MODELS_RESULTS_2026-11-19.md)（B0–B7 上游全表）｜[`docs/FIRST_ROUND_RESULTS.md`](FIRST_ROUND_RESULTS.md)（B0–B5 首轮）｜[`docs/WEEK5_IMPLEMENTATION_REPORT.md`](WEEK5_IMPLEMENTATION_REPORT.md)（M5 代码侧）｜[`docs/WEEK4_IMPLEMENTATION_REPORT.md`](WEEK4_IMPLEMENTATION_REPORT.md)（Week 3 见 `WEEK3_IMPLEMENTATION_REPORT.md`）

## M5.1 评估审计与协议增强（2026-11-28，本地分支 `codex/m5.1-evaluation-audit`）

| 项 | 值 |
| --- | --- |
| 范围 | Evaluation Audit & Protocol Enhancement；**未新增模型**，B0–B7 数学定义未改（`test_models` / `test_week3_models` 全绿） |
| 修复（P0） | ① 负样本 train/val/test 互斥；② `validate()` 补 6 类检查（`encode⊆train`、self-loop、权重长度等）；③ 缺边区分 `adjacency` / `supervision_aware`；④ 遮蔽统一到 `protocols.py`；⑤ 训练前审计 + `audit.json` + `status=complete` |
| 新增（P1） | `degree_matched` / `two_hop` 负样本、5 个非神经网络基线、两套可见性协议、Stage 0–5 实验矩阵（57 格）、冻结 checkpoint 评估 |
| 新增文件 | `src/cofi_gene/{protocols,audit,baselines,evaluate}.py`、`scripts/{audit_dataset,run_baselines,run_experiments}.py`、`tests/test_m5_1_*.py`（8 个文件）、3 份 M5.1 文档 |
| 测试 | `pytest -q` → **300 passed**（218 → +82，无 skip）；CI 增加审计/基线/Stage 4/Stage 3 冒烟 |
| 数据版本 | 历史 `real_v1` **不覆盖**；训练须用新构建 `real_v1_1`（同划分、负样本互斥）；`real_v1_1_deg` 用于困难负样本评估 |
| 历史结果口径 | M5/M6 结果 = **原始评估协议下的初步结果**，需用修复后协议复核；旧 checkpoint 只作诊断 |
| 未解决 | 服务器 `codex/m5-source-specific`、`codex/m6-downstream` 未入库；真实数据审计与重训待服务器执行（B-11） |
| 交付文档 | [`docs/M5_1_EVALUATION_AUDIT.md`](M5_1_EVALUATION_AUDIT.md)、[`docs/M5_1_SERVER_TRAINING_GUIDE.md`](M5_1_SERVER_TRAINING_GUIDE.md)、[`docs/M5_1_DEVELOPMENT_HANDOVER.md`](M5_1_DEVELOPMENT_HANDOVER.md) |

## M5 / M6 真实数据结果（队友提交于 2026-11-25 / 2026-11-26，2026-11-28 合并）

| 项 | 值 |
| --- | --- |
| 合并方式 | `git fetch` + `git pull --ff-only origin main`：`06c175a..4cabf59`，3 个提交、**仅新增 3 个 markdown，无代码改动**；本地无独有提交，工作区干净 |
| 实现来源 | M5 运行来自服务器分支 `codex/m5-source-specific`（整合提交 `5ec6cbf`，服务器回归 248 passed）；M6 来自 `codex/m6-downstream`（`8ff31f8`，253 passed / 1 skipped）。**两个提交在本仓库 `git cat-file` 均为 not-in-this-repo** |
| 关键口径差异 | M5/M6 使用**来源特异 + 残差**的 B4–B7 实现，与本仓库 main 上「`encode()` 对两通道返回同一融合向量」的定义不同；两边的 B4–B7 指标不可直接互换引用 |
| M5 规模 | real_v1：18,002 基因 / 128 维；A 测试正负例各 14,408、B 各 87,126（阳性率 50%）；主曲线 B0–B7 × {0,10,30,50}% = 32 单元 + 错位 5 单元（B3–B7，ratio=0）+ 双视图遮蔽 3 单元（B1/B4/B5，ratio=0.3），共 40 单元、seed 42、两个 4090 并行 |
| M5 结论 | 四档删边比例下 A 通道最高仍是 B1（0.9617 → 50% 0.9578）、B 通道最高仍是 B2（0.9670 → 0.9636），**双图方法未超各自单图**；B6 在 50% 删边两通道各降约 0.009（幅度大于 B5），**本轮不支持「Attention 反馈提升缺边鲁棒性」**；错位后 B5 两通道 −0.0045 / −0.0022，B7 −0.0102 / −0.0015，方向不一致，机制待独立证据 |
| M5 预算风险 | 36/40 单元跑满 200 epoch，26 个最优验证权重落在 epoch 200 → 预算上限问题依旧存在 |
| M6 数据 | PIE 四细胞系（HepG2/Jurkat/K562/RPE1），`p_adj<0.05` 无 log2FC 门槛；train/val/test = 1283/274/276 目标、24,358,488 / 5,167,932 / 5,252,016 行；按目标分组防泄漏，自然阳性率（test macro 参考 0.093746、micro 0.096429） |
| M6 训练 | 冻结上游表示 + 统一 MLP（261→128→64→1、41,857 参数），Adam lr=1e-3、batch 65,536、≤12 epoch、patience=3，权重按验证集 macro target AP 选择 |
| M6 首轮（seed 42） | **macro target AP 最高是 B0 = 0.2946**；micro AP 最高 B7 = 0.3580（B0 = 0.2925，+0.0655）；micro AUROC 最高 B6 = 0.8017 |
| M6 三种子（下游 seed 42/43/44） | macro target AP：B0 0.294553±0.001138 仍最高，B7 0.289564±0.001619；成对差值 B1−B0 −0.025852、B5−B4 −0.004266、B6−B5 +0.002130、B7−B4 +0.000824；micro AP：B7−B0 +0.059236 |
| M6 逐目标 | 276 个测试目标中 B7 相对 B0 提高 108 / 下降 163 / 持平 5；低阳性率目标组平均退步明显，RPE1 差距最大（B0 0.3942 vs B7 0.3567） |
| M6 可支持判断 | 现有证据**不支持**图预训练提升「平均未知扰动目标」的预测能力；整体（micro）排序有收益但逐目标未提升，不能改主指标或断言机制 |
| 已知上游缺陷 | 上游 train/test 负样本仍有重叠（A=12、B=354），未以修正数据重训 → 本轮 M6 只能标为「当前表示的下游初评」 |
| 复现缺口 | `scripts/run_downstream_de.py`、`docs/SOURCE_SPECIFIC_IMPLEMENTATION.md`、`runs/m5_source_specific_20261125/`、`runs/m6_20261126/`、`data/processed/real_v1/` 均只在训练机（数据/权重/预测不上 GitHub，符合约定） |

M5/M6 之后的本仓库下一步（不新增模型，仍守 B0–B7）：

1. **要求把 `codex/m5-source-specific`、`codex/m6-downstream` 推回 GitHub（或重放为补丁）**，至少补上 `scripts/run_downstream_de.py` 与 `SOURCE_SPECIFIC_IMPLEMENTATION.md`；否则本仓库无法重跑 M5/M6。
2. **修复上游 train/test 负样本重叠（A=12、B=354）后复验上游与下游**，在此之前 M6 结论一律标注「初评」。
3. 统一按验证集确定训练预算（M5 有 26/40 单元顶到 200 epoch，M6 有单元顶到 12 epoch 上限）。
4. 诊断 M6 的 macro/micro 分歧与低阳性率目标、RPE1 退步；**不按测试排名为单组调参**，保留 macro target AP 为主指标。

## Week 5 交付（2026-11-22，M5 代码侧）

| 项 | 值 |
| --- | --- |
| 范围 | `AGENTS.md` §6 实验二（10/30/50% 缺边）、实验三（Gene ID 错位）；**不新增模型**，不改数据管线/划分语义 |
| 新增文件 | `src/cofi_gene/perturbation.py`、`scripts/run_robustness.py`、`tests/test_week5_{perturbation,cross_view,robustness_script}.py`、`docs/WEEK5_IMPLEMENTATION_REPORT.md` |
| 修改文件 | `models/__init__.py`（`CROSS_VIEW_MODELS` + `set_cross_view_permutation`）、`config.py`（3 个可选 `data.*` 键）、`train.py`（扰动接线 + `perturbation` 记录 + 3 个 CLI 开关）、`.github/workflows/ci.yml`、`README.md`、`docs/{DECISIONS,ROADMAP,STATUS}.md` |
| 删边协议 | 只删 `encode_edges`；`val/test` 正边与固定负样本不动；种子 = `SeedSequence([seed, channel_index, 0xC0F1])`、与比例无关 → 10% ⊂ 30% ⊂ 50%；每个单元重训 |
| 遮蔽协议 | 对通道 X 的 val/test 正边 `p`，从另一通道 Y 的 `encode_edges` **和** `train` 正边删除；默认关闭（D-016 未锁定） |
| 错位对照 | 只允许 b3–b7；derangement 由 `(N, seed)` 生成，融合前对单个视图 `index_select`；non-persistent buffer，`state_dict` 不变 |
| 测试 | `pytest -q` → **218 passed**（+32：12 numpy + 14 torch + 6 脚本） |
| CI | `torch-cpu` 增加 drop=0.3 / misalign / drop=0.0 三次冒烟 + `b1` 两格 sweep + JSON 断言 |
| 兼容性 | 不开启扰动时 config/hash/run 名/`metrics.json` 与 Week 4 完全一致（`edge_drop_fraction=0` 与基线逐位相等） |
| 未做（训练方边界内） | 真实数据曲线实跑、`mask_other_view=true` 口径的正式确认、3 seed、下游 DE |

## 真实数据首轮结果（队友提交于 2026-11-16 / 2026-11-19，2026-11-22 合并）

| 项 | 值 |
| --- | --- |
| 合并方式 | `git merge --ff-only origin/main`：`ae0b46b..d6a86bd`，2 个提交、**仅 2 个新增 markdown，无代码改动** |
| 数据 | STRING v12.0 / 人类 9606：图 A experimental≥400 = 144,071 边，图 B coexpression≥150 = 871,260 边，重合 42,035，Jaccard=0.043188；节点 18,002；`gene_text` 3,072→128 PCA |
| 下游数据审计 | PIE 四细胞系 36,258,678 行，1,833/1,857 扰动目标对齐（98.71%），候选标签 `p_adj<0.05`；**DE 实验仍未跑** |
| 上游协议 | train/val/test = 80/10/10；每 split 1:1 固定负样本（阳性率 50%，随机 AP 基线≈0.5）；`mask_other_view=false`（D-016 口径：另一视图可见） |
| 训练预算 | Adam lr=0.01 / wd=0 / ≤200 epoch / eval_every=10 / patience=50 / grad_clip=5，与 `configs/_base.yaml` 一致；B0 无参数，优化步数为 0 |
| 环境 | RTX 4090 24 GB / 驱动 580.178.04 / Python 3.11.15 / torch 2.5.1+cu121 / PyG 2.8.1（**不是 5090**） |
| 测试 | 训练机回归 187 passed、1 skipped（CUDA 探针专用）；合并当时本仓库 `186 passed`（Week 5 起为 `218 passed`） |

测试 AP（`average_precision`，seed 42；B1 只评图 A、B2 只评图 B）：

| 模型 | A 测试 AP | B 测试 AP | 双通道平均 |
| --- | --- | --- | --- |
| B0 特征点积 | 0.8078 | 0.6826 | 0.7452 |
| B1 单图 A | **0.9618** | — | — |
| B2 单图 B | — | **0.9670** | — |
| B3 拼接 | 0.9394 | 0.9355 | 0.9375 |
| B4 Gate | 0.9609 | 0.9587 | 0.9598 |
| B5 Attention | 0.9431 | 0.9408 | 0.9419 |
| B6 Attention+反馈 | 0.9444 | 0.9398 | 0.9421 |
| B7 Gate+反馈 | 0.9592 | 0.9615 | **0.9603** |

可以写进报告、但必须标注“单 seed + 当前口径”的判断：

- **图学习优于固定文本特征**：B1−B0 在 A 上 +0.1540，B2−B0 在 B 上 +0.2844（同时含 GCN 结构与可训练参数，无法拆分贡献）。
- **融合尚未超过对应单图**：A 最高仍是 B1，B 最高仍是 B2；B 通道上双图最好的 B7 比 B2 低约 0.0056。
- **Gate 优于 Attention**：B4 比 B3 高约 +0.022；B5 比 B4 低约 −0.018；带反馈时 B7 > B6。
- **反馈收益很小且方向随通道翻转**：B6−B5 = A +0.0014 / B −0.0010；B7−B4 = A −0.0017 / B +0.0028。
- 代码事实（已在本仓库核对）：B3–B7 的 `encode()` 对两个通道返回**同一份融合向量**，且 `edge_head=dot` 解码无参数，因此同一基因对在 A/B 打分相同；A/B 指标差异只来自各自评估边集合不同。

复现风险（下一步必须先处理）：

1. 结果文档引用的三个修复提交 `d3c5433a435b`（基因符号数字后缀被当版本号）、`7386d5ab7e4f`（GCN 邻接张量 CUDA 设备）、`0df87b474f02`（无参数 B0 直接评估）**在本仓库不存在**（`git cat-file` 均为 not-in-this-repo），只保存在训练机 `codex/real-data-first-round` 分支。
2. `data/processed/real_v1/`、`runs/real_v1/b*_real_v1_seed42/`、`manifest.json` 与 `dataset.npz`（SHA-256 `fee691ce…`）都只在训练机上，本仓库 `data/processed/` 仅有 `ci_synthetic`、`synthetic_v1`。
3. 只有 seed 42；B1/B2/B4/B5/B7 的最佳 epoch 落在 200 上限，训练预算要按验证集统一确定后才能支撑消融结论。

下一步（按 `AGENTS.md` 范围，不新增模型）——**2026-11-22 记录，状态已由开头「M5 / M6 真实数据结果」小节更新**：

1. 把三个服务器修复提交推回仓库或重放为补丁（未完成；M5/M6 的 `codex/*` 分支同样未推回）。
2. ~~下游 DE MLP（M6）仍未实现~~ → **已由训练机实现并跑出结果（`scripts/run_downstream_de.py`，2026-11-26），但实现未进入本仓库**。
3. ~~缺边鲁棒性 10/30/50%（M5）待跑~~ → **已完成 40 单元（2026-11-25，单 seed）**；`mask_other_view=true` 口径作为补充记录；3 seed（M7，上游三 seed）仍待做。

## Week 4 交付（2026-11-15，M4）

| 项 | 值 |
| --- | --- |
| 分支 | `feat/week4-graph-feedback`（基于 `main` = `eddf1d7`）→ 本地 `--ff-only` 合并到 `main`，**未 push** |
| 提交 | `6d2c584`（实现）、`b66c392`（审计/等价性测试）、随后一条 docs/CI 提交 |
| 新增模型 | **B6 = B5 Attention + 单轮边权反馈**、**B7 = B4 Gate + 单轮边权反馈**（反馈模块无参数，参数量分别等于 B5/B4） |
| 新增文件 | `models/feedback.py`、`configs/b6.yaml`、`configs/b7.yaml`、`tests/test_week4_feedback.py`、`tests/test_week4_leakage.py`、`docs/WEEK4_IMPLEMENTATION_REPORT.md` |
| 修改文件 | `models/__init__.py`（注册表 b0–b7、`set_encoding_graphs()`、两段式 forward）、`train.py`（传编码图 + 反馈元数据）、`config.py`（可选键 `feedback_alpha/margin`，不进 DEFAULTS）、`models/fusion.py`（仅 docstring）、`tests/test_{models,config,train,week3_models}.py`、`.github/workflows/ci.yml`、`README.md`、`docs/DECISIONS.md`、`docs/STATUS.md` |
| 反馈公式 | `p = σ(detach(head_c(H¹)))`；`w' = clamp(w·(1+α(2p−1)), (1−m)w, (1+m)w)`，`α=0.1`、`m=0.5`；只重加权既有训练边，不补边 |
| 测试 | `python -m pytest -q` → **186 passed**（4.25 s，CPU；Week 3 为 147，+39） |
| 等价性 | `α=0` 时 B6≡B5、B7≡B4（`eval()` + 同 seed + 同参数，`torch.equal` 逐位相等） |
| 防泄漏 | 反馈邻接 == 该通道 `encode_edges`；交换 val/test 正边后训练 loss 与 state_dict 逐位不变；`mask_other_view=True` 场景亦通过 |
| 冒烟 | 合成数据 60 基因/16 维、seed 42、3 epoch、`--skip-test`：B6 `val_mean_ap=0.635594`、B7 `0.886939`，两次运行逐位一致 |
| 配置 hash | B0–B5 不变（`bd7bb75ec0f2`/`f3160ec62e2d`/`0ea7242eb7f9`/`f2ebde49513f`/`eb0a3fe0c5a8`/`29ee907107e4`）；B6 `a4456156d5fd`、B7 `5df6a41740ac` |
| 尚未做 | 真实 STRING/PIE 数据、RTX 5090、缺边鲁棒性（M5）、下游 DE MLP（M6）、3 seed 汇总（M7）、候选补边（D-022 排除） |

### Week 4 未解决事项（继续保留）

1. **D-016 跨视图遮蔽协议待确认**：默认 `mask_other_view=false`（"另一视图可见"），B6 可间接利用对方视图可见的边。
2. **`optim.negatives_ratio` 既有配置问题**：只被校验、不生效（负样本在构建期固定，W2-1），改键无效果。
3. **真实数据与 GPU 验证未完成**：所有"跑通"均为合成数据 + CPU。 → **已于 2026-11-16/19 由训练机在 RTX 4090 上以真实数据完成（见开头小节）；本仓库侧仍只有合成数据。**
4. **反馈成本与偏乐观打分**：每 step 多一次编码（≈2× 前向）；反馈用共享 head 对训练正边打分，`p` 偏高（受 `m=0.5` 限制）。
5. **D-023 待确认**：当前"每 forward 重算一次反馈"，尚未实现 warmup 后冻结式。


## 分支管理（2026-11-15）

| 项 | 值 |
| --- | --- |
| 结论 | `main` 已**快进合并** Week 2 + Week 3 全部成果，不再是停在 M0 的 `dfa496e` |
| 合并方式 | `git merge --ff-only feat/week2-data-pipeline` → `dfa496e..b875b7e`，4 个提交、57 文件、+6028/−50 |
| 合并后验证 | `.venv/bin/python -m pytest -q` → **147 passed**（4.60 s，CPU，torch 2.15.0+cpu） |
| 远程状态 | `origin/main` = 本合并结果（`06d8f14` 及以后）；CI run **43677502606**：`data-and-splits` ✅、`torch-cpu` ✅ |
| 开发分支 | `feat/week2-data-pipeline` 保留（与 `main` 同提交，名已过期，勿再往上提交）；**Week 4 / M4 从 `main` 切新分支** |
| 未做 | 未 force push、未删除任何分支、未改动历史提交、未提交真实数据/权重（`data/`、`runs/` 仍在 `.gitignore`） |

## 文档交付（2026-11-15）

- 新增根目录 **[`TRAINING_HANDOFF.md`](../TRAINING_HANDOFF.md)**：面向"真实数据 + RTX 5090 训练"同学的交接文档
  （进度差说明、P0–P3 优先任务、可复制命令、实验红线、P0–P3 验收表、第一波最低交付标准、BLOCKED/TODO 清单）。
  `README.md` 顶部已加醒目相对链接。
- 口径：文中所有命令的参数均取自当前提交的真实接口（`scripts/*.py` 的 argparse、`python -m cofi_gene.train --help`、
  `configs/*.yaml`），并在本机用**合成数据**做了等价验证（`147 passed`、`build_dataset --dry-run`、
  `train --dry-run`、`CoFiDataset.validate()`、`metrics.json` 读取）。
  **真实 STRING/PIE 路径与 GPU 命令未在真实数据/GPU 上执行**，仍是待验证项。

## Week 3 交付（2026-11-07）

| 项 | 值 |
| --- | --- |
| 分支 | `feat/week2-data-pipeline`（Week 3 当时未合入 `main`；**已于 2026-11-15 快进合入 `main`**，见开头“分支管理”节） |
| 新增模型 | **B4 Feature-wise Gated Fusion**（单 `Linear`+Sigmoid，单融合向量）、**B5 对称双向 Cross-Graph Attention**（4 头 × 32 维） |
| 新增文件 | `models/attention.py`、`configs/b4.yaml`、`configs/b5.yaml`、`tests/test_week3_models.py`、`docs/WEEK3_IMPLEMENTATION_REPORT.md` |
| 修改文件 | `models/__init__.py`、`models/fusion.py`（+`GatedFusion`）、`train.py`（`--skip-test`）、`config.py`（`attention_heads`/`attention_dropout`/`attention_residual`）、`tests/test_{models,config,train}.py`、`.github/workflows/ci.yml` |
| 测试 | `python -m pytest -q` → **147 passed**（Week 2 为 99，本次 +48） |
| 冒烟训练 | B3/B4/B5 在 `data/processed/synthetic_v1`（60 基因/16 维）上跑通，**用 `--skip-test` 不读 test split**；产物在 `runs/week3_smoke/` |
| 可复现 | 同 seed 下 B3/B4/B5 的 loss 与 val 指标完全一致（参数化确定性用例） |
| 参数统计（feature_dim=16, hidden=128） | B0 2,176 ｜ B1/B2 18,688 ｜ **B3 = B4 = 70,272（参数对齐）** ｜ B5 169,344 |
| 尚未做 | B6/B7、缺边鲁棒性、错位对照、下游 DE MLP、3 seed、真实数据、GPU |

### Week 3 明确未做

- **未实现任何 Feedback**（B6/B7 仍抛错），遵守"Feedback 不是本周目标"。
- 未改数据管线 / 划分 / 负采样 / 泄漏防线 / `metrics.py` / B0–B3 数学定义。
- 未改 `docs/DECISIONS.md` 既有条目（只在末尾**新增** D-019～D-021 待确认项）。
- 未用真实 STRING/PIE 数据，未在任何 GPU 上训练，未 commit、未 push、未合并。

## Week 2 最终交付（2026-11-01）

| 项 | 值 |
| --- | --- |
| 分支 | `feat/week2-data-pipeline`（Week 2 当时未合入 main；**已于 2026-11-15 合入，main 现为 `b875b7e`**） |
| 功能提交 | `59f0dd9` feat(week2): 数据管线 + 防泄漏防线 + B0–B3 上游训练 |
| 文档提交 | `21d389f` docs(status): 记录 CI run 41283635590 两个 job 均通过 |
| CI | run 41283635590（`data-and-splits` ✅、`torch-cpu` ✅，后者含完整训练命令端到端检查） |
| 测试 | `python -m pytest -q` → **99 passed**（CPU，torch 2.15.0+cpu，pyarrow 26.0.0） |
| 确定位 | 同 seed 两次训练的 loss 与 val/test 指标完全一致 |
| 合成数据规模（示例） | 60 基因 / 16 维特征；A=117 边、B=106 边、重合 52（jaccard 0.30） |
| 队友交接 | 见 `docs/TEAM_HANDOFF.md` |

### 四层状态区分（重要，勿混淆）

| 层次 | 内容 |
| --- | --- |
| ① **代码已实现** | STRING 双通道读取、ID 对齐、图构建、边/目标划分、负采样、泄漏防线、特征压缩、PIE 标签、数据集容器与 manifest、**B0–B7** 训练/评估（B6/B7 为 Week 4 单轮反馈）、**Week 5 协议扰动（删边/遮蔽/基因错位）+ 批量曲线脚本**、两个数据 CLI、配置体系、**218 个测试**、两个 CI job |
| ② **合成数据 + CPU 已验证** | 上述全部在 `cofi_gene.synthetic` 生成的假 STRING/PIE 上跑通；B0–B7 各有 val/test 指标，训练可复现；B6/B7 另有防泄漏审计与 `α=0` 等价性验证；Week 5 扰动另有删边/遮蔽的泄漏守卫与错位置换的等价性用例 |
| ③ **真实数据已验证（训练机）** | 2026-11-16 / 11-19 在训练机用真实 STRING v12.0 + PIE 跑完 B0–B7 上游，数值见 `docs/ALL_MODELS_RESULTS_2026-11-19.md`；但原始数据、manifest、`runs/real_v1/` 与三个修复提交**均未入库**，本仓库仍不能独立复现 |
| ④ **GPU 已验证（4090，非 5090）** | 真实规模训练在 RTX 4090 24 GB 完成（torch 2.5.1+cu121 / PyG 2.8.1）；RTX 5090（sm_120）与 5090 上的 B6/B7 双前向显存行为**仍未验证** |

## 已确认的事实


- 研究设计在根目录 `AGENTS.md`，范围固定为 B0–B7；顶部含 Agent 工程维护入口。
- 团队两人：coding/模型实现一人为主；另一人负责上平台训练并共同确认数据与生物学设置。
- 远程仓库 https://github.com/huqian2024/CoFi-Gene 公开，默认分支 `main`（2026-11-15 起包含 Week 2/Week 3 全部成果）；
  `feat/week2-data-pipeline` 已合并冻结，后续 Week 4 / M4 从 `main` 切新分支开发。
- 本机（开发机）：Python 3.12.2、numpy 2.5.4、pyarrow 26.0.0、**torch 2.15.0+cpu（仅 CPU）**；GPU 为 RTX 4060 Laptop，**不是 5090 训练机**。
- 开发机访问 GitHub / PyPI 需经 `127.0.0.1:7890` 代理（仓库 `.git/config` 已设 http(s).proxy；pip 需 `export HTTPS_PROXY=...`）。
- CI（`.github/workflows/ci.yml`）有两个 job：`data-and-splits`（快，numpy/pyarrow）与 `torch-cpu`（装 CPU 版 torch 并跑训练命令）。

## 当前验收

| 工作项 | 状态 | 验收证据 |
| --- | --- | --- |
| 原始研究范围 B0–B7 | 已定义 | `AGENTS.md` |
| 包结构 / 依赖分组 / CI | 已完成 | `pyproject.toml`（core=numpy+pyyaml；extras: data/train/dev）、两个 CI job |
| 单测总体 | **218 passed**（4.52 s，CPU；Week 4 为 186，Week 5 新增 32） | `.venv/bin/python -m pytest -q` |
| STRING 双通道解析（分列、阈值、去重、自环、未知蛋白、combined-score 不当证据） | 已实现，合成数据验证 | `cofi_gene/string_io.py`、`graph_build.py`、`tests/test_string_io.py` |
| 基因 ID 对齐（符号归一化、多蛋白冲突计数） | 已实现，单测覆盖 | `cofi_gene/gene_ids.py`、`tests/test_gene_ids.py` |
| 边划分 / 分组划分 / 负样本 | 已实现，含确定性 | `cofi_gene/splits.py`、`negative_sampling.py`、`tests/test_splits.py` |
| 防泄漏防线 | 已实现并在构建期强制 | `CoFiDataset.validate()`、`tests/test_leakage.py`（7 个对抗用例） |
| 特征 3072→128 | 已实现（固定符号 SVD/PCA） | `cofi_gene/features.py`、`tests/test_features.py` |
| PIE DE 读取与标签规则 | 已实现（pyarrow） | `cofi_gene/pie_io.py`、`tests/test_pie_io.py` |
| 数据构建 CLI + manifest | 已实现（sha256/git/版本/命令/报告） | `scripts/build_dataset.py`、`tests/test_scripts.py` |
| B0–B3 训练/评估 | 已实现，合成数据端到端跑通 | `cofi_gene/train.py`、`models/`、`tests/test_train.py`（4 模型 × 真实前向反向） |
| 训练确定性 | 已确认（同 seed 两次 loss 与指标完全一致） | `tests/test_train.py::test_training_is_deterministic_for_a_fixed_seed` |
| B4 Gated Fusion | ✅ 已实现，合成数据跑通 | `models/fusion.py::GatedFusion`、`configs/b4.yaml`、`tests/test_week3_models.py` |
| B5 Cross-Graph Attention | ✅ 已实现，合成数据跑通 | `models/attention.py`、`configs/b5.yaml`、`tests/test_week3_models.py` |
| B6 One-Step Feedback（B5 + 反馈） | ✅ 已实现，CPU synthetic 已验证 | `models/feedback.py`、`configs/b6.yaml`、`tests/test_week4_{feedback,leakage}.py` |
| B7 One-Step Feedback（B4 + 反馈） | ✅ 已实现，CPU synthetic 已验证 | 同上（`configs/b7.yaml`） |
| Week 5 缺边鲁棒性 / 双视图遮蔽 / 基因错位对照 | ✅ 代码 + 协议完成，合成数据验证（真实数据曲线待训练方） | `cofi_gene/perturbation.py`、`scripts/run_robustness.py`、`tests/test_week5_*.py` |
| 下游 DE MLP | **未实现**（M6） | — |
| 真实 STRING / PIE 数据 | 训练机已验证（哈希与发布记录一致），原始数据未入库 | `docs/ALL_MODELS_RESULTS_2026-11-19.md`；`DATA_CONTRACT.md` 第 3 节已按该文补录 |
| 服务器 torch/CUDA（4090 已验证 / 5090 未验证） | 4090 上严格 CUDA 探针通过并完成 B0–B7 | 同上（5090 仍需队友跑 `python scripts/check_environment.py --strict`） |

## 正在执行（Week 5 已交付，下一步起点）

0. **M5 完成（代码 + CPU，2026-11-22）**：删边鲁棒性 / 双视图遮蔽 / 基因错位对照已实现，
   `python scripts/run_robustness.py` 可批量产出曲线；**真实数据的曲线与 3 seed 需训练方执行**。
1. **训练方本轮任务（S1–S3）**：①把 3 个服务器修复提交 push 上来（否则首轮真实结果无法从仓库复现）；
   ②在真实数据上跑 M5 曲线与错位对照，并按 `docs/M5_ROBUSTNESS_RESULTS_TEMPLATE.md` 交结果文档；
   ③锁定 D-016（遮蔽口径）与 D-018（预算）。命令与验收见 `TRAINING_HANDOFF.md` §0 / §4.5，
   同步约定见 `docs/TEAM_HANDOFF.md` §〇。

1. **B0–B7 全部已实现**：`configs/b0–b7.yaml` 可用（B4/B5 = Week 3，B6/B7 = Week 4）；下游 MLP 仍未实现。
2. **M1 收尾（数据，需队友）**：确认 STRING 版本/物种/阈值（D-005）、gene_text 来源与压缩方式（D-012/D-017）、DE 标签规则（D-014），然后跑真实数据构建。
3. **M0 收尾（服务器，需队友）**：5090 上 `python scripts/check_environment.py --strict` 通过并留档。
4. **M3 收尾（需两人）**：锁定 D-018（训练预算）以及新增的 D-019/D-020/D-021（门控/注意力/监督形式的工程默认），
   然后在真实数据上跑 B0–B5 单 seed 冒烟 → 才允许跑多 seed 消融。
5. **M4 完成（代码 + CPU，2026-11-15）**：B6/B7 单轮边权反馈已实现并验证（`AGENTS.md` 4.3、**D-022**：只调既有边权重，不补边）。
   （此前此处误引 D-009；D-009 实为"基因符号归一化与冲突策略"。细节见 `docs/WEEK4_IMPLEMENTATION_REPORT.md`。）
6. **真实数据首轮训练（P2）已完成（2026-11-19，训练机）**：B0–B7 均在同一数据集 / seed 42 / 预算下跑完，
   数值与风险见本文开头「真实数据首轮结果」；前置的修复提交仍未回仓，且只有单 seed。
7. **D-016 / D-018 / D-023 待锁定**：跨视图遮蔽口径、训练预算、反馈重算方式，均会影响实验可比性。

## Week 2 明确未做 / 不做的部分

- 未实现 **B4–B7**（注册表主动抛 `NotImplementedError`）；未实现下游 DE MLP。
- 未做缺边鲁棒性（10/30/50%）、Gene ID 错位对照、多种子汇总、三类图。
- 未下载/核验任何真实数据，未在任何 GPU 上训练过。
- 未合并到 `main`，未改动 `main`，未 force push。
- 按交付要求，**今日不进入 Week 3 / M3**。

## 下一阶段启动条件（缺一不可）

1. `docs/DECISIONS.md` 的 **P0 三项**（D-005、D-009、D-014）已锁定。
2. 队友在 5090 上跑通 `scripts/check_environment.py --strict` 并记录版本信息。
3. `configs/data.yaml` 填入真实路径后 `scripts/build_dataset.py` 成功产出 `manifest.json`（含基因数/边数/重合率/覆盖率）。
4. **P1 五项**（D-012/D-017/D-011/D-015/D-018）在开跑 B4/B5 之前锁定，否则消融不可比。

## 最近完成（Week 3 提交，2026-11-07）

- 新增 `models/attention.py`（`CrossGraphAttention`：**4 头 × 32 维**、每基因 2 个候选 token、来源特异 Q/K/V、
  双向对称、残差可配置且默认关闭）。
- `models/fusion.py` 新增 `GatedFusion`（**单层** `Linear(2d→d)` + Sigmoid，`H = g⊙H_A + (1-g)⊙H_B`，单融合向量）。
- `models/__init__.py`：`IMPLEMENTED_MODELS=(b0..b5)`、`PLANNED_MODELS=(b6,b7)`；B4/B5 使用 per-channel edge head；
  新增 `encode()`（唯一 `[N,128]` 下游嵌入）与 `fusion_representations()`（B5 的 `Z_A/Z_B`，仅分析用）。
- `train.py`：打分传 `channel=`；B5 超参接线（`attention_heads`/`attention_dropout`/`attention_residual`）；
  `metrics.json` 的 `model` 段新增 `fusion_kind` / `edge_heads`；新增 `evaluate_test` 参数与 `--skip-test` 开关。
- `config.py`：允许并严格校验可选键 `attention_heads` / `attention_dropout` / `attention_residual`（**不写入 `DEFAULTS`**，
  因此 B0–B3 的 resolved config 与 run 名 hash 完全不变，实测 `configs/b3.yaml` 仍是 `f2ebde49513f`）。
- 新增 `configs/b4.yaml`、`configs/b5.yaml`；CI 的 `torch-cpu` job 增加 B4/B5 训练命令检查。
- 验证命令与输出：
  - `.venv/bin/python -m pytest -q` → **147 passed**（新增 48 个用例）
  - `python -m cofi_gene.train --config configs/b4.yaml --dataset data/processed/synthetic_v1 --out-dir runs/week3_smoke --epochs 40 --run-name week3_smoke_b4 --seed 7 --skip-test` → `metrics.json`（`fusion_kind=gate`, `edge_heads=per_channel`, `test=null`）
  - 同上把 `b4` 换成 `b5` → `fusion_kind=attention`；换成 `b3` 作回归对照
  - 参数统计（feature_dim=16）：B0 2,176 ｜ B1/B2 18,688 ｜ B3 = B4 = 70,272 ｜ B5 169,344
- 详细说明、公式、张量形状、待确认事项见 **`docs/WEEK3_IMPLEMENTATION_REPORT.md`**。

## 最近完成（Week 2 提交）

- 新增数据管线：`string_io`、`gene_ids`、`graph_build`、`features`、`pie_io`、`splits`（扩展）、`negative_sampling`、`dataset`、`pipeline`、`manifest`。
- 新增上游训练：`config`（严格校验 + `base:` 继承）、`metrics`（自实现 AUPRC/AUROC，含 tie 处理）、`models/{gcn,fusion,heads}`、`train`（CLI + 运行记录 + checkpoint）。
- 新增脚本：`scripts/build_dataset.py`、`scripts/make_synthetic_dataset.py`；新增 `configs/`（`_base.yaml`、`b0–b3.yaml`、`data.yaml`、`data.synthetic.yaml`）。
- 新增测试 14 个模块（含 conftest）、99 个用例；CI 拆成 `data-and-splits` 与 `torch-cpu` 两个 job。
- 验证命令与输出：
  - `.venv/bin/python -m pytest -q` → 99 passed
  - `python scripts/make_synthetic_dataset.py --raw data/raw/ci_synthetic --processed data/processed/ci_synthetic` → 60 节点 / 16 维特征
  - `python -m cofi_gene.train --config configs/b3.yaml --dataset data/processed/ci_synthetic --out-dir runs/ci --epochs 4 --run-name ci_b3` → 生成 `metrics.json`
- CI（run 41283635590，commit `59f0dd9`，分支 `feat/week2-data-pipeline`）：`data-and-splits` ✅、`torch-cpu` ✅（后者包含完整训练命令的端到端检查）。

## 明确的限制（不得当成已完成）

1. 本仓库（开发机）内的自动化验证仍基于**合成数据**（`cofi_gene.synthetic`）；真实 STRING/PIE 首轮结果由训练机产出（见开头小节），但原始数据、权重与 run 目录未入库，**不能在本仓库重现**。
2. 训练在开发机 **CPU**（合成数据）与训练机 **RTX 4090**（真实数据）上验证过；RTX 5090 的 torch/CUDA（sm_120）与显存行为**仍未验证**。
3. `configs/_base.yaml` 的 lr/epochs/early-stop 是**占位值**；合成小图上 loss 曾出现尖峰（2.4→128→收敛），说明超参仍需在 M3 固定。
4. 未做：**B6–B7（Feedback）**、缺边鲁棒性（10/30/50%）、Gene ID 错位对照、下游 DE MLP、跨种子汇总、三类图。
   （B4/B5 已在 Week 3 补齐；B0–B5 仍未在真实数据或 GPU 上跑过。）

## 维护规则

- 状态只写实际验证过的事情；"计划""实施中""已完成"严格区分，合成/真实数据必须标注。
- 完成代码任务必须记录：commit、测试命令与结果、数据版本、实验 run ID、下一步。
- 在新对话里先以此文件为准，不从上一次对话推断进度。
