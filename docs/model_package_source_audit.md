# `pilkwang/rogii-model-package` 源码与 Guard640-v4b 独立审计

日期：2026-07-24（Asia/Shanghai）

## 结论

本次审计把“是否泄漏”和“是否值得强融”分开判断：

| 问题 | 结论 |
|---|---|
| 隐藏推理是否读取 horizontal query 的未知 `TVT` | **否；PASS** |
| `include_target=False` 是否沿 feature-builder 调用链生效 | **是；PASS** |
| visible same-ID 三井的 formation/row imputer 是否排除 query well | **是；PASS** |
| 源码/特征表是否硬编码 train TVT、井 ID、14,151/4,301 行数或目标哈希 | **未发现；PASS** |
| package 的 `10.6702` OOF 是否证明 `0.49 / 192` 外层强融有效 | **否** |
| v4b 是否应做 hidden submission | **NO-GO** |

所以，v4b 的问题不是已经发现 query-target 泄漏，而是**缺少与现有 anchor
同折、同样本的融合验证，并且现有可执行诊断强烈为负**。即使 v4b 普通 run
最终技术审计通过，也不应据此提交 hidden。

机器可复现审计：

```powershell
uv run python scripts/audit_model_package_source.py
```

输出为 `artifacts/model_package_source_audit.json`。全量 773 井 imputer 构建、
三口 visible same-ID 井邻居检查、带真实 `TVT` 的 train-schema false-gate
检查均通过。

为排除审计副本漂移，又从 Kaggle dataset 重新下载两个源码文件并逐字节比较：

- `build_features.py` SHA-256：
  `0DA2B08D900E1707B9136DF1CD2B72B6DEFE43EC1DFBA4BF67F87BA9A8F01E7C`；
- `rogii_feature_core.py` SHA-256：
  `AB877AAD4537AF30C1E1E8B1BACE1C3D513C2A2228323BB57295C45BFF636F0D`。

二者都与 `data/external/model_package_audit` 中被审计的副本一致。

## 1. v4 到 package builder 的实际调用链

Guard640 notebook 的 model-package cell 46：

1. 在 cell 内第 96–114 行只从已挂载 package 的
   `feature_builders/build_features.py` 动态导入 builder；
2. 第 116–144 行仅把 competition root、动态 sample、package root 和 manifest
   传入 builder；
3. 第 372–402 行调用 builder、按 manifest 的 feature columns 加载五个模型并预测；
4. 第 421–436 行只把 package 输出的 residual delta 加到
   `last_known_TVT`，再按动态 sample ID 对齐。

这里没有 notebook 侧的 hidden target 参数；真正的安全边界在 package builder。

下载源码 `build_features.py` 的证据：

- 第 69–73 行加载 feature core 并解析实际需要的 480 个特征；
- 第 76–83 行只用 competition **train** 井建立 formation/row imputers；
- 第 85–99 行对 `core.TEST_DIR` 构建 test frame，并冻结
  `include_target=False`（第 89 行）和
  `exclude_query_well_from_formation=True`（第 98 行）；
- 第 100–107 行仅按运行时 sample ID 对齐，不使用固定 ID 或固定行数。

`rogii_feature_core.py` 的继续传递：

- `build_tail_feature_frame`（第 2590–2646 行）在第 2624 行把
  `include_target` 传给 `make_tail_features_for_well`，并在第 2639 行传给
  `compact_tail_feature_frame`；
- `make_tail_features_for_well`（第 1076–1959 行）用
  `TVT_input.isna()` 定义 hidden suffix（第 1142–1149 行）；
- horizontal query 的真实 `tail['TVT']` 只在第 1956–1958 行出现，并受
  `if include_target and 'TVT' in df.columns` 支配；
- compact frame 也只在 `include_target=True` 时加入 target 列
  （第 2553–2578 行）。

AST 自检只发现一个 horizontal query `TVT` 访问点，即第 1957 行，且确认它受
第 1956 行的 `include_target` 条件支配。进一步把含有真实 `TVT` 列的
`train/000d7d20__horizontal_well.csv` 当作输入、显式设置
`include_target=False` 后，输出 3,836 行且 target 列为零，证明 false gate
不是仅因为 test 文件本来没有 `TVT` 才“碰巧通过”。

### `TVT` 字样不都代表 query target

源码中的下列 `TVT` 使用是合法输入，不能误报为 hidden label：

- `TVT_input`：horizontal well 的已知前缀；
- 每井 `__typewell.csv` 的 `TVT`：比赛提供的参考 typewell 轨迹；
- train horizontal `TVT`：训练模型、OOF 打分和训练统计使用的已知标签。

实际三口 test CSV 只有 `MD,X,Y,Z,GR,TVT_input`，没有 horizontal `TVT`。
v4 的 Kaggle run 又成功执行完整 package builder 并产出 14,151 行；如果调用链
无条件读取 test `TVT`，该运行会直接缺列失败。

## 2. formation 和 row imputer 的 same-ID 排除

源码行为：

- `FormationPlaneKNN` 在第 918–957 行从 train formation tops 建索引；
  `impute` 在第 972–975 行把匹配 `self_wid` 的邻居距离改成无穷；
- `RowANCCKNN` 在第 996–1029 行从每个 train well 最多采样 400 行；
  `impute` 在第 1047–1048 行排除所有 `self_wid` 行；
- `make_tail_features_for_well` 在第 1451 行把当前 `well_id` 赋给
  `self_wid`，并在第 1458–1459、1505–1506 行分别传给 prefix/tail 的
  plane 和 row imputer。

审计脚本用完整 773 井训练索引，在三口 visible same-ID test 井的第一个 hidden
坐标上重放真实邻居选择，同时用 spy 检查调用实参：

| Well | plane 未排除时 self/10 | plane 排除后 | row 未排除时 self/20 | row 排除后 | prefix/tail 实参 |
|---|---:|---:|---:|---:|---|
| `000d7d20` | 1 | **0** | 20 | **0** | 两次均为本井 ID |
| `00bbac68` | 1 | **0** | 20 | **0** | 两次均为本井 ID |
| `00e12e8b` | 1 | **0** | 20 | **0** | 两次均为本井 ID |

尤其 row imputer 在不排除时 20 个邻居全部来自同井；当前代码实际排除后为零。
这否定了“v4 package formation 路径仍直接吃 visible same-ID 训练 formation
行”的假设。

## 3. 静态泄漏扫描与边界

扫描对象为实际 Kaggle dataset 中的：

- `feature_builders/build_features.py`；
- `feature_builders/rogii_feature_core.py`；
- `feature_builders/feature_columns.json`；
- `metadata/model_package_manifest.json`；
- `stacking/blend_config.json`。

结果：

- 480 个模型特征中没有 `id`、`well_id`、`row_index`；
- 没有 `TVT`、`target_tvt`、`target_delta` 或
  `target_delta_from_last_known` 作为模型特征；
- 源码/配置中没有 8 位固定井 ID；
- 没有固定 `14151`、`4301`；
- 没有 `sha256`、MD5、prediction fingerprint、expected hash/rows/IDs
  或 canary。

特征表确实包含 `prefix_len`（第 12 行）、`tail_len`、`tail_row_number`
（第 117–118 行）、`n_rows`、`row_frac`（第 127–128 行）。这些值来自当前
query 文件，是合法几何特征，但可能成为训练井的**准标识符**并引起过拟合。
由于源码不存在“行数/井 ID -> TVT”静态映射，这属于泛化风险，不是目标泄漏。

TCN 的 `well_id` 只在 v4 cell 46 第 243–258 行用于分组和排序；输入张量仍严格是
480 个 feature columns，没有井 ID embedding。

### 一个合法但应记录的 train-target 读取

feature core 在导入时会执行：

- `estimate_abs_tvt_slope_quantiles`（第 2768–2784 行）；
- 第 2772 行从 train CSV 读取 `MD,TVT`；
- 第 2816–2820 行对所有 train 井计算 slope quantile。

这是**推理时读取已知 train label**，不是读取 query label，按竞赛监督学习语义合法。
而且 `MAX_ABS_TVT_SLOPE*` 在该源码中除赋值外没有 scoring-path 使用，因此当前只是
无效导入副作用。它应被视为 P2 代码卫生问题，而不是 hidden leakage。

同理，serialized models 和 blend weights 是用 train TVT 拟合的；这是正常训练。
manifest 第 27–85 行声明最终推理加载的是 `alltrain` 模型。由于 visible 三口井
也在这 773 个训练井中，package 的 visible-demo 预测并非完全独立测试：
alltrain 模型已见过这些井的 feature/target rows，虽然没有直接使用井 ID。
因此 visible demo 只能提供诊断，不能当 hidden 分数估计。

## 4. 为什么 `OOF=10.6702` 不支持 `0.49 / 192`

`blend_config.json` 的报告含义是 package **内部五模型 blend**：

- 第 18 行 `anchor_weight=0.0`；
- 第 19 行权重来源是 `groupkfold_oof_tree_plus_sequence_tcn`；
- 第 22 行 imputer mode 为
  `global_train_self_exclude_query_well`；
- 第 24–25 行 package raw/postprocessed OOF 分别为
  `10.7106 / 10.6702`；
- 第 28 行是 5 folds。

它没有现有 SP45/learned anchor 的 OOF 预测，也没有 v4 外层

```text
gate(d) = 0.49 / (1 + (|package - anchor| / 192)^2)
```

的同折分数。单模 RMSE 不能推出融合 RMSE。若先忽略逐行可变 gate，令
`e = anchor - y`、`d = package - anchor`、固定权重为 `g`，则：

```text
MSE(anchor + g*d) - MSE(anchor)
  = 2*g*E[e*d] + g^2*E[d^2]
```

package 自身的 RMSE 只约束部分边际量，不能给出关键的联合项 `E[e*d]`。
必须在相同 held-out rows 上同时保存 anchor、package、truth 才能证明融合收益。

此外，manifest 第 17–18 行明确承认 OOF formation/KNN imputer 是全局 train
构建，只对 query well 做 self-exclusion，并没有每折重建。这个方案对 hidden
query 仍是 target-free，但不是最严格的 fold isolation，使 `10.6702` 的误差条
应比普通 GroupKFold 更保守。

## 5. v4 的可执行反证

`artifacts/kaggle_runs/guard640_v4_v1` 是完整普通 run，不是源码推测：

- package/base mean absolute disagreement：`16.3946 ft`；
- disagreement p95：`27.5867 ft`；
- `0.49 / 192` 的平均实际 gate：`0.48585`；
- 候选平均绝对移动：`7.9447 ft`；
- 候选 p95/max 移动：`13.2441 / 14.3012 ft`。

这不是“小修正”，而是近似半权融合。v4 的 25-ft 全局 guard 正确拒绝了它。

冻结 truth 后的三井诊断：

| 轨迹 | Pooled RMSE |
|---|---:|
| clean 96-seed anchor | `3.8246` |
| VP 后、package 前 | `3.8254` |
| 强制 `0.49 / 192` package candidate | **`7.5237`** |
| package-only | **`16.2652`** |

强融候选逐井 RMSE 为 `6.3600 / 9.2083 / 5.6215`，相对 package 前
`3.0368 / 3.5829 / 4.6874` 全部变差。

### v4b 普通 run 已直接复现失败

审计进行期间，`heralces/rogii-guard640-v4b` version 1 普通 run 完成。它没有
触发 25-ft guard，确实选择了 package：

- disagreement p95：`26.3492 ft`；
- `selected_for_submission_csv=True`；
- mean/p95 package move：`7.7712 / 12.6725 ft`；
- 独立 submission audit：14,151 行、结构合法、0 issues。

同一次 v4b run 内的冻结 stage 分析消除了跨 run 随机性的混淆：

| v4b stage | Pooled RMSE |
|---|---:|
| 本次 clean anchor | `3.5523` |
| VP 后/package 前 | `3.5689` |
| package 选中后 | **`7.7242`** |
| branch hedge 后 | `7.6295` |
| 最终 U-continuity 后 | **`7.5659`** |

package 层单独使 pooled RMSE 增加 `4.1553`；完整实际栈相对本次 clean anchor
增加 `4.0137`。三井 final 也全部劣化：

- `000d7d20`：`2.9602 -> 6.3748`；
- `00bbac68`：`3.1686 -> 9.4984`；
- `00e12e8b`：`4.4343 -> 5.0933`。

v4 与 v4b 的 upstream PF/anchor 数值存在正常 run-to-run 差异，所以不应直接拿两次
最终文件互减来归因；上表全部使用 v4b 自己保存的 stage 文件，package 前后比较仍是
严格的同 run 对照。结构审计通过而评分诊断失败，正好说明“技术安全”与“值得提交”
是两个独立门槛。

模型输出分布还有一个更具体的告警：

| Family | OOF prediction mean | 当前三井 prediction mean | 外层 package 权重 |
|---|---:|---:|---:|
| XGB | `+1.603` | `-4.172` | 约 0 |
| CatBoost | `+1.659` | `-3.500` | `0.4496` |
| HGB | `+1.772` | `-4.194` | `0.1178` |
| LGB | `+1.596` | `-4.998` | `0.0151` |
| TCN | `+1.634` | **`-35.584`** | **`0.4175`** |

高权重 TCN 的当前输出均值相对 OOF 发生约 `-37.2 ft` 偏移，直接解释 package
整体的负向漂移。三口井真实 suffix delta 均值只有约 `-5.45 / -2.52 / -0.76 ft`，
而 package 预测为 `-20.05 / -20.99 / -11.49 ft`。

由于 alltrain 模型见过这三井，以上不是严格外部验证，不能证明 hidden 必然同样差；
但它提供了明确的负证据。当前又没有任何同折 anchor+package 融合 CV 来抵消这条
证据，因此不能用“源码 target-free”替代“模型值得提交”的证明。

## 6. v4b hidden go/no-go

**最终判定：NO-GO hidden submission。**

把 `MODEL_PACKAGE_DIFF_P95_DISABLE` 从 `25` 改为 `1e9` 已经在普通 run 中强制
选中上述近半权 package candidate；它没有修复 TCN 分布漂移，也没有补上 joint
OOF，且把本次同 run pooled RMSE 从 `3.5689` 直接恶化到 `7.7242`。v4b 普通
run 可作为技术复现实验保留，但“完成、finite、ID 对齐”只能证明管线可执行，
不能证明评分层有效。

只有同时满足以下条件，才应重新打开 hidden gate：

1. 在所有 773 井的冻结 grouped suffix folds 上，为**同一行**保存 clean anchor
   与 package OOF；
2. formation/row imputer 至少做“每 query self-exclude”的重放，并报告按 fold
   重建 imputer 的敏感性；
3. 直接评估 `0.49 / 192` 的 pooled、macro、p90/p95 和 worst-well 增量；
4. 在预先冻结的至少三个 disjoint well partitions 上都不劣于 anchor；
5. 单独解释或移除 TCN 当前 `-35.6 ft` 输出均值漂移；
6. 证明收益不是由 visible same-ID 三井、固定行长或 alltrain memorization 驱动。

在这些证据出现前，优先提交已独立验证的 clean anchor / target-free continuity
候选，而不是 v4b 强融。
