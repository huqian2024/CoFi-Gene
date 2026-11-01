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

| 来源 | 版本/commit | sha256 | 基因/边或行数 | 下载地址/相对路径 |
| --- | --- | --- | --- | --- |
| STRING A experimental | 待填写 | 待填写 | 待填写 | 待填写 |
| STRING B coexpression | 待填写 | 待填写 | 待填写 | 待填写 |
| PIE gene_text | 待填写 | 待填写 | 待填写 | 待填写 |
| PIE DE / preprocessed | 待填写 | 待填写 | 待填写 | 待填写 |

不得将大型原始数据和授权密钥放入 Git 仓库；保存 manifest、可重现下载命令和转换脚本。
