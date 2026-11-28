# 数据接口合同（v0，待真实文件验收）

## 0. 统一定义

- `gene_ids`: 长度 N 的唯一且**顺序固定**的基因标识数组，不能仅凭字典迭代顺序隐式对齐。
- `X`: `[N, F]` 同一份公开先验基因特征，B0–B7 共享；来源首选 PIE_sources 的 `gene_text`，其已发布原始维数为 3072（须先核对 `meta.json`）。
- `edge_index_A`, `edge_index_B`: `[2, E]` 的整数索引、节点范围 `[0,N)`；图 A/B 应独立存储通道得分，排序、去重、无自环规则统一。
- `edge_weight_A`, `edge_weight_B`: `[E]` 数值、可选，经统一阈值/归一化处理。STRING 的 combined score **不得**代替两种单独证据。
- `id_map`: 记录 STRING protein ID ↔基因符号/稳定基因 ID 映射、冲突处理、丢失计数。

## 1. 上游 Link Prediction 拆分

对 A/B 各自确定 train/val/test 正边并锁定 seed。编码和反馈只能使用本实验允许看到的训练图；验证/测试正边不得加入任何训练图或候选标注。负样本避开该目标图中全部已知正边（含隐藏正边），评估固定负例与阳性率。重合边需要两个实验口径：

- **另一视图可见**：目标图隐藏边可在另一图出现，描述为跨证据迁移，而非完全未知边恢复。
- **双视图均遮蔽**：两图同时移除相关同一基因对，检验真正未知关系。

模拟 10/30/50% 缺边时只额外删**可见训练边**，保持评估边集合固定。记录是否有反向边、重复边、负采样随机种子、各图训练边数。

## 2. 下游 PIE

公开数据链接：`https://huggingface.co/datasets/arcinstitute/PIE_replogle_nadig_essential`。

PIE `de/<cell_line>.parquet` 公开的列包括：`target`（扰动目标）、`feature`（响应基因）、`ref_mean`（对照平均表达）、`log2_fold_change`、`p_adj`；四个细胞系 K562/RPE1/HepG2/Jurkat。其 `preprocessed/` 包含 `ctrl_means.npy`、`fdr.npy`、`tested.npy`、`meta.json` 等。注意 `de` parquet 与 `preprocessed/` 是**两种表示形式**，不要直接假定行顺序与 NPY 的轴对齐。

初版建议从单个 DE parquet 小样本检查：数据列、缺失值、基因 ID、p_adj 分布、目标/响应基因覆盖。标签规则必须由两人锁定后固化（候选：`p_adj < 0.05`；有无效应量门槛待定），并保存标签阳性率。

拆分必须按 `target` 的唯一基因 ID **分组**：同一目标跨多个细胞系只落在 train/val/test 的一个集合。先拆目标，再生成 `(target, feature, context)` 样本，绝不能用随机记录行划分。公开基因先验可用于所有目标，但测试目标响应标签不得进入任何训练逻辑。B0–B7 使用统一测试行及冻结 embedding/MLP 配置。

## 3. 数据清单（下载后填真实值）

> 2026-11-22 按 `docs/ALL_MODELS_RESULTS_2026-11-19.md` / `docs/FIRST_ROUND_RESULTS.md` 补录训练机实测值。
> 原始文件的 sha256 与完整构建配置记录在训练机 `data/processed/real_v1/manifest.json`，**未随仓库提交**；
> 数据集产物 `dataset.npz` 的 SHA-256 为 `fee691ce1da49aa055c12003635a527bb3acabe9f087c99c239dc4e665f8685d`（八个模型共用）。

| 来源 | 版本/commit | sha256 | 基因/边或行数 | 下载地址/相对路径 |
| --- | --- | --- | --- | --- |
| STRING A experimental | STRING v12.0 / 人类 9606 / `experimental≥400` | 见训练机 manifest | 144,071 条边（18,002 节点） | [STRING download](https://string-db.org/cgi/download?species_text=Homo+sapiens)（detailed 通道文件） |
| STRING B coexpression | STRING v12.0 / 人类 9606 / `coexpression≥150` | 见训练机 manifest | 871,260 条边；与 A 重合 42,035（Jaccard=0.043188） | 同上 |
| PIE gene_text | `ae4ce640b7edff7ac4acfd1abed74ca4138e5931` | 与发布记录一致（校验通过） | 3,072 维 → PCA 128 维 | [PIE_sources](https://huggingface.co/datasets/arcinstitute/PIE_sources/tree/ae4ce640b7edff7ac4acfd1abed74ca4138e5931) |
| PIE DE / preprocessed | `5119439699a8a02b0f6ebe593cf0dd373891bd37` | 四份 parquet 与发布记录一致 | 四细胞系 36,258,678 行；1,833/1,857 扰动目标对齐（98.71%） | [PIE_replogle_nadig_essential](https://huggingface.co/datasets/arcinstitute/PIE_replogle_nadig_essential/tree/5119439699a8a02b0f6ebe593cf0dd373891bd37) |

标签规则（候选，D-014 仍未正式锁定）：`p_adj < 0.05`，`min_abs_log2fc` 默认关闭。

不得将大型原始数据和授权密钥放入 Git 仓库；保存 manifest、可重现下载命令和转换脚本。

## 4. 已实现的产物格式（v1，代码：`cofi_gene.pipeline` / `cofi_gene.dataset`）

处理后的数据集写入一个目录（例如 `data/processed/<dataset>/`），包含三个文件：

| 文件 | 内容 |
|---|---|
| `dataset.npz` | `X`；每个通道的 `<ch>_train` / `<ch>_val` / `<ch>_test`（`[2,E]` int64）、对应 `<ch>_*_w` 权重、`<ch>_neg_train/val/test`、`<ch>_encode` 与 `<ch>_encode_w`（编码器实际看到的图） |
| `dataset.json` | `dataset_version`、`gene_ids`（唯一且顺序固定的基因序）、各通道构建统计、下游 `target_splits`、`meta`（通道列名、降维方式、split 决策） |
| `manifest.json` | 输入文件的 sha256/字节数、`git_commit`、运行时版本、`command`、完整构建报告（基因选择、通道统计、重合率、split 摘要、下游覆盖率） |

要点：

- **节点顺序**只由 `gene_ids` 决定；特征、边、划分都按位置索引，任何环节不得自行重排。
- **编码图与评估边分离**：`<ch>_encode` 默认等于 `<ch>_train`；验证/测试正边不会出现在其中（`CoFiDataset.validate()` 会直接报错）。
- **负样本**在构建期一次性抽样并落盘，避开该通道全部已知正边（含隐藏的 val/test 正边），所有 B0–B7 共用。
- **特征降维**只使用公开基因先验，绝不接触 PIE 的 DE 标签；`explained_variance_ratio` 记录在 manifest。

## 5. 当前验证状态（必须区分真实与合成）

| 项 | 状态 | 证据 |
|---|---|---|
| STRING 双通道解析、阈值、去重、自环、未知蛋白、combined-score 陷阱 | 已在**合成数据**上验证 | `tests/test_string_io.py`、`tests/test_pipeline_synthetic.py` |
| 基因符号归一化与冲突计数 | 已在合成数据 + 单元测试验证 | `tests/test_gene_ids.py` |
| 边划分 / 分组划分 / 负样本防泄漏 | 已实现并单测 | `tests/test_splits.py`、`tests/test_leakage.py` |
| 3072→128 降维 | 已实现（SVD/PCA）并在合成矩阵上验证 | `tests/test_features.py` |
| PIE DE 读取与标签规则 | 已实现，pyarrow 环境可跑 | `tests/test_pie_io.py` |
| **真实 STRING/PIE 文件** | **未下载、未核验** | 第 3 节表格仍为空，需团队补齐 |


## 6. M5.1 评估协议与数据版本（2026-11-28 新增）

### 6.1 负样本定义（必须区分，不得混称）

- **默认（目标通道语义）**：负样本 = 该通道内**未被观察到**的基因对。某对在 A 中无边、但在 B 中是正边，
  在 A 的 Link Prediction 任务里仍可作为负样本——这正是"目标通道"的含义。
- **严格双图语义（可选，需单独建数据集）**：负样本 = 两图均未观察到的基因对；用于"完全未知关系"任务，
  必须另建数据集并在报告中标注为不同任务，不能与默认口径混在一张表里。
- 负样本一律是 **sampled non-edge / presumed negative**，不是生物学证实的"无关系"。

### 6.2 负采样规则（`cofi_gene.negative_sampling`）

| 规则 | 内容 |
| --- | --- |
| 互斥 | 通道内 train/val/test 负样本两两不相交；采样按 `train → val → test` 顺序累积 forbidden |
| 正边 | 任何 split 的负样本都不得命中该通道全部已知正边（含隐藏的 val/test 正边） |
| 形式 | 统一 canonical `(min, max)`；无 self-loop；split 内无重复 |
| 策略 | `uniform_random`（默认，历史口径）/ `degree_matched` / `two_hop` |
| 度数来源 | 只允许用**协议可见的训练边**（`split.train`，即 `encode_edges` 的上界）计算 |
| fallback | `uniform` / `relaxed_degree` / `error`；每次 fallback 计数并写入 `meta.negative_sampling.*.splits.*.fallback_rate` |
| 记录 | `dataset.json → meta.split`（strategy/fallback/degree_buckets/互斥说明）与 `meta.negative_sampling`（每通道策略、seed、attempts、fallback、`cross_split_overlap`） |
| 比例 | `PipelineConfig.negatives_ratio` 在**构建期**生效；`optim.negatives_ratio` 不参与训练（改它无效果，要改比例必须重建数据集） |

### 6.3 跨视图可见性协议（`cofi_gene.protocols`）

| 协议 | 编码图 | 训练正例 | 适用结论 |
| --- | --- | --- | --- |
| `cross_view_transfer`（A，默认） | 本通道 train 子集；另一视图可见同名对 | 本通道 train | "跨证据迁移/补全"；必须报告每通道 held-out 在另一视图的可见数量与比例 |
| `dual_view_strict_holdout`（B） | 两通道 train 子集 **减去** 两通道 val/test 并集 | 同样减去该并集 | "两视图均无直接监督"的严格口径；残留可见度必须为 0 |

两协议都保持：节点、特征、val/test 正边、负样本集合不变；执行后必须重新通过 `CoFiDataset.validate()`。

### 6.4 缺边模式（`cofi_gene.perturbation`）

| 模式 | 删除对象 | 可支持的表述 |
| --- | --- | --- |
| `adjacency`（M5 历史口径） | 只删 `encode_edges`（及其权重） | "邻接信息减少" |
| `supervision_aware` | 同一次嵌套抽样删除 `encode_edges` **和** `train` 正例 | "训练关系缺失"；需报告 `num_train_positives_removed` |

比例 0/0.1/0.3/0.5，删边种子 = `SeedSequence([run_seed, channel_index, 0xC0F1])`，与比例无关 → 嵌套；
`data.drop_channels` 可只损伤 A、只损伤 B 或两者（报告中必须写明删了哪些通道）。

### 6.5 数据版本

| 目录 | 内容 | 用途 |
| --- | --- | --- |
| `data/processed/real_v1` | 历史构建（负样本可能跨 split 重复） | 保留只读；审计会 FAIL，不可再用于正式训练 |
| `data/processed/real_v1_1` | 同划分/同特征，负样本互斥 | M5.1 之后的所有正式训练 |
| `data/processed/real_v1_1_deg` | 同上，`negatives_strategy=degree_matched` | Stage 3 困难负样本评估（冻结 checkpoint） |
| `data/processed/synthetic_v1`、`ci_synthetic` | 合成数据 | 单元/CI 冒烟，不得作为科学结果 |

### 6.6 审计门槛（每次训练前自动执行）

`CoFiDataset.validate()` 与 `cofi_gene.audit.audit_dataset()` 必须 PASS 才能训练；检查项、ID 与失败后果见
`docs/M5_1_EVALUATION_AUDIT.md` §3。任何 WARN（如特征溯源不可核验）会写入 `metrics.json.audit.warning_ids`，
必须在结果报告中说明，不能当作 PASS 忽略。

