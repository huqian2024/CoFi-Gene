# 队友交接说明（Week 2 结束 → 下一阶段开始前）

> 面向：负责数据与生物学解释、以及 5090 服务器训练的同学。
> 仓库：https://github.com/huqian2024/CoFi-Gene ｜ 分支：`feat/week2-data-pipeline`（**未合入 main**）
> 本文件只讲"现在能用的东西 + 需要你确认/准备的事"。详细状态见 `docs/STATUS.md`。

---

## 一、Week 2 已经完成什么（可以当黑盒用的部分）

一句话：**数据从原始文件到可训练数据集、再到 B0–B3 上游训练评估的整条链路已经写完，并在合成假数据上跑通。**

| 能力 | 状态 | 你可以怎么用 |
| --- | --- | --- |
| 读 STRING 明细表，拆出 experimental / coexpression 两张图 | ✅ 代码完成 | `scripts/build_dataset.py` 一条命令 |
| 蛋白 ID → 基因符号对齐（含冲突/丢失计数） | ✅ 代码完成 | 报告写在 `manifest.json` 的 `alignment` / `gene_selection` |
| 基因节点集统一、特征 3072→128 压缩 | ✅ 代码完成 | 压缩方式默认 PCA，**待你确认**（D-012） |
| 边划分（train/val/test）+ 按扰动目标分组划分 | ✅ 代码完成 | 种子固定 42，写在 `manifest.json → split_decisions` |
| 防泄漏检查（隐藏边不进编码图、负样本避开全部正边、目标不跨 split） | ✅ 代码完成且强制 | 数据集加载/构建时会直接报错，不是"靠自觉" |
| PIE DE 读取 + 标签规则 | ✅ 代码完成 | 规则默认 `p_adj<0.05`，**待你确认**（D-014） |
| B0–B3 训练与评估（AUPRC/AUROC） | ✅ 代码完成 | `python -m cofi_gene.train --config configs/b3.yaml` |
| **B4 门控融合 / B5 跨图注意力（Week 3 新增）** | ✅ 代码完成 | `configs/b4.yaml`（单层 Linear 门控）/ `configs/b5.yaml`（4 头 × 32 维、残差默认关）；详见 `docs/WEEK3_IMPLEMENTATION_REPORT.md` |
| 可复现性 | ✅ 已验证 | 同 seed 两次运行，loss 与指标完全一致（B3/B4/B5 均有参数化用例） |

**已验证的范围（不要夸大）**：以上全部只在 **合成假数据 + CPU** 上验证过。
**没有验证**：任何真实 STRING/PIE 文件、任何 GPU（5090 一行代码都没跑过）、B6/B7（Feedback）、下游 DE 预测。

Week 3 的规矩：**B4/B5 已实现，但门控/注意力/双图监督的具体形式是"工程默认"（D-019/D-020/D-021），
需要你我确认后才会写进报告方法节；B6/B7 本周不做。**

---

## 二、需要你确认的研究选择（按优先级）

完整说明见 `docs/DECISIONS.md`。这里只列必须你我一起定、且不定就返工的部分。

### P0：不定就跑不了真实数据

| 编号 | 要决定什么 | 现状默认 | 建议怎么定 |
| --- | --- | --- | --- |
| D-005 | STRING 版本（v12.0？）、物种（human 9606？）、**两个通道各自的分数阈值** | 模板里写 A=400 / B=150，仅参考值 | 先给"能接受的边数量级"，再看留出的评估边是否够评估 |
| D-009 | 一个基因符号对应多个 STRING 蛋白时保留哪个 | 取字典序最小 protein id | 确认这样不会丢掉重要蛋白，或指定别的规则 |
| D-014 | DE 标签规则：只 `p_adj<0.05`，还是再加 `|log2FC|` 门槛 | 只用 `p_adj<0.05` | 定了要记录全局阳性率，并写进报告的基线 |

### P1：不定就会导致 B0–B7 之间不可比（开跑 B4/B5 前必须定）

| 编号 | 要决定什么 | 现状默认 |
| --- | --- | --- |
| D-012 | `gene_text` 3072→128 用什么方法 | PCA（符号固定），不做 z-score |
| D-017 | PCA 在哪个基因集上拟合 | 在最终节点集上拟合 |
| D-011 | 边权是否归一化 | 不归一化（保留 STRING 0–1000） |
| D-015 | 正负样本比例、是否每个 epoch 重采样 | 每 split 1:1，构建期固定、所有模型共用 |
| D-018 | 训练预算（lr / epoch 数 / 早停 / 参数量上限） | 占位值：lr 0.01、200 epoch、patience 50 |

### P2：M5（鲁棒性实验）之前定即可

- D-016：是否启用"双视图均遮蔽"口径（当前关闭）。
- D-010：同一基因对重复出现时分数取最大还是均值（当前取最大）。

> 这些默认值都写在代码里、可一条配置改掉；**在你确认前，任何数字都不能写进报告当结论。**

---

## 三、需要你准备的真实数据（放进 `data/`，已被 .gitignore 排除，不会进仓库）

目标目录建议：

```
data/raw/string/9606.protein.links.detailed.v12.0.txt.gz   # 两通道都在这个文件里
data/raw/string/9606.protein.info.v12.0.txt
data/raw/pie_sources/gene_text.npy                         # 3072 维
data/raw/pie_sources/gene_text.ids.txt                     # 一行一个基因 id，与矩阵行对齐
data/raw/pie/de/{K562,RPE1,HepG2,Jurkat}.parquet
```

准备时请顺手记下（要填进 `docs/DATA_CONTRACT.md` 第 3 节表格）：

- [ ] STRING 版本号与下载日期；物种 ID（预期 9606）
- [ ] 每个文件的 sha256（`sha256sum <file>`）
- [ ] `gene_text` 的原始维数是否真的是 3072（看它的 `meta.json`）
- [ ] **基因 id 的写法**：是 symbol（TP53）还是 Ensembl（ENSG00000141510）？带不带版本号？
- [ ] PIE 四个细胞系的 DE 表是否都齐全；`target` / `feature` 列各是什么写法
- [ ] 四个细胞系之间的目标基因是否重叠（决定下游分组怎么分）

> 关键提醒：`gene_text.ids.txt` 这一行**必须有**，而且要和你拿到的矩阵行顺序严格对应。
> 没有 id 顺序，特征无法对齐到基因，管线会直接报错而不是"猜"。

---

## 四、5090 服务器需要执行的验证（按顺序）

```bash
git clone https://github.com/huqian2024/CoFi-Gene.git && cd CoFi-Gene
git checkout feat/week2-data-pipeline
python -m pip install -e '.[dev,data]'
python -m pytest -q                       # 期望 99 passed
python scripts/check_environment.py --strict   # 必须退出码 0
```

要点：

- [ ] `check_environment.py --strict` 通过。
      `--strict` 会同时要求 torch、PyG 与 CUDA 张量算子都可用。
      **若 PyG 在 sm_120 上装不上**：先跑 `python scripts/check_environment.py`（不加 `--strict`），
      把 `torch` / `cuda_available` / `compute_capability` 三项贴回来，并注明 "PyG 未安装"。
      上游训练只用 torch，**没有 PyG 也能跑 B0–B3**；这点已记录在 D-013，不要为了过 `--strict` 硬装不兼容的轮子。
- [ ] 记录并贴回：Python / torch / CUDA 版本、GPU 型号与显存（不要贴 token、账号、ssh key）
- [ ] 安装与 sm_120（5090 架构）匹配的 torch wheel；若官方轮子暂不支持，记录报错原文，不要自己造数据
- [ ] 跑一次最小的真实训练链路确认显存与速度：

```bash
python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \
    --processed data/processed/synthetic_v1 --n-genes 200 --n-raw-features 256
python -m cofi_gene.train --config configs/b3.yaml --dataset data/processed/synthetic_v1 \
    --device cuda --epochs 10 --run-name smoke_cuda_b3
```

- [ ] 把 `runs/smoke_cuda_b3/metrics.json` 里的 `versions` 字段贴回来（这是"GPU 已验证"的唯一证据）

---

## 五、下一阶段的前置条件（四条全满足才开工）

1. **P0 三项锁定**：D-005（STRING 版本/阈值）、D-009（符号冲突）、D-014（DE 标签）。
2. **服务器验证通过**：5090 上 `--strict` 成功 + GPU 版 `smoke_cuda_b3` 跑出 `metrics.json`。
3. **真实数据构建成功**：`configs/data.yaml` 填好后跑

   ```bash
   python scripts/build_dataset.py --config configs/data.yaml --out data/processed/real_v1
   ```

   产出 `manifest.json`，其中能读到：节点数、两图边数、重合边比例、下游目标覆盖率。
4. **P1 五项锁定**：D-012 / D-017 / D-011 / D-015 / D-018。

满足后即可用**同一预算、同一划分**跑 B0–B5 首轮对比（B4/B5 代码已就绪）；这一步产出的才是能写进报告的
上游 Link Prediction AUPRC 表。Week 3 新增的 D-019/D-020/D-021 需同时确认（否则 B4/B5 的方法描述口径不定）。

Week 3 已完成（代码侧）：**B4 Feature-wise Gated Fusion（单层 Linear 门控）+ B5 对称双向 4 头 Cross-Graph Attention**，
含 per-channel edge head 与统一 `[N,128]` 下游嵌入；合成数据上 B3/B4/B5 已用 `--skip-test` 端到端跑通（`runs/week3_smoke/`）。
详见 `docs/WEEK3_IMPLEMENTATION_REPORT.md`。

---

## 六、请避免的坑（Week 2 里踩过的）

1. STRING 单通道分数为 0 表示"该类型没有证据"，**不能**用 `combined_score` 当通道分数，否则两张图会重合（已修）。
2. 训练用的负样本在数据构建时固定，所有模型共用；不要在训练里重新抽，否则模型之间不可比。
3. 隐藏的验证/测试边绝不能出现在编码图里；这类错误现在会让数据集构建直接失败，属于预期行为。
4. 合成数据上 lr=0.01 曾出现 loss 尖峰（2.4→128→收敛），说明预算还没定；**不要**用合成结果下结论。

