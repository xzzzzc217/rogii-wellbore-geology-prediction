# PF 公共复现 notebook 静态审计

审计对象：

- `kaggle_notebooks/pf_public_reproduction/rogii-pf-public-reproduction.ipynb`
- `kaggle_notebooks/pf_public_reproduction/kernel-metadata.json`

审计日期：2026-07-23。本文只审计当前源码、元数据和已挂载公开资产的接口，不把 notebook 标题中的公开分数 `6.491` 当作私榜或隐藏重跑保证。

## 结论

当前 notebook 适合做“公开方案能否复现并生成合法提交”的 v1 实验，但不建议原样作为唯一正式方案。最重要的风险不是明显的隐藏目标泄漏，而是：

1. 可见的 3 个测试井与训练井同 ID，源码中有多层 same-ID 直接查表/formation 路径；它们会在本地 demo 上触发，却通常不会在约 200 个新 ID 的隐藏测试井上触发。
2. 当前默认 `vp_balanced_modelpkg_005`、model-package 修正和最后的 PF midpoint hedge 没有对应的已执行输出证据；公开标题分数不能证明这些最新源码层有效。
3. 历史执行元数据在 3 个可见井上已耗时约 11.7 分钟（整次 papermill 约 12.9 分钟）。按井数线性外推，已执行部分约 6.4 小时，再加未计时的 model-package 特征和模型推理，存在逼近或超过 Kaggle 9 小时限制的实质风险。
4. 两套特征管线里的单 PF 使用 Numba 全局随机数但没有逐井显式设种，且并行执行，因此“固定配置”不等于逐次 bitwise 可复现。

最终 schema 审计较好：最后会验证 `id,tvt` 两列、行数、ID 顺序和有限值。因此输出格式风险低于模型与运行时风险。

### `6.491` 是否来自当前源码/当前 profile

不能把 `6.491` 归因于当前 exact source + `vp_balanced_modelpkg_005`：

- notebook 的 papermill 记录显示整次执行发生在 2026-07-07；
- model-package cell 的历史时长只有 `0.093 s`，这只符合当时走 “correction skipped” 分支，不符合加载 feature builder、5 类模型并生成预测；
- 最后的 PF midpoint hedge cell 没有 execution count，也没有 papermill duration，说明它不在这份嵌入输出对应的执行链中；
- 当前 control cell 选择了 `vp_balanced_modelpkg_005`，但 markdown 的 “Default contact-gated settings” 仍写 `contact_gated_anchor`，也是源码后改而输出未同步的信号。

作者可能在另一个 Kaggle version、variation notebook 或后续 code submission 上得到过 6.491；也可能只是保留了早期公开分数标题。静态文件无法否定作者确实获得该分数，但有充分证据说明**当前 notebook 内嵌执行记录并未验证当前默认 profile 和最终 hedge**。因此 v1 的实际分数要以我们自己的 code submission 为准。

## 风险分级

| 级别 | 风险 | 隐藏重跑是否实际触发 | 判断 |
|---|---|---:|---|
| P0 | 约 200 井时超过 9 小时 | 是 | 3 井历史耗时已约 11.7 分钟；model-package 当前默认开启但历史计时显示它当时未运行 |
| P0 | 最新源码层没有公开分数对应证据 | 是 | `vp_balanced_modelpkg_005` 与 PF midpoint hedge 的 code cell 未执行/无时长，不能把标题 6.491 直接归因于当前最终输出 |
| P1 | same-ID 训练真值/formation 覆盖 | 隐藏新 ID 通常否；可见 3 井是 | 本地 demo 会被显著“美化”，不得用其结果估计隐藏泛化 |
| P1 | 未显式播种的 Numba PF + 多线程 | 是 | 影响 ridge/learned 特征与最终融合的稳定性 |
| P1 | 外部 pickle/joblib 与超大训练特征资产 | 是 | 兼容性、加载时间、内存和供应链风险；不是目标泄漏本身 |
| P1 | 无验证的最终 PF midpoint hedge | 是 | 无配置总开关，最多可整体移动单井 2 ft，且位于 model-package 之后 |
| P2 | 精细常数可能含 public-LB 选择偏差 | 是 | 没有发现 notebook 内部读排行榜或自动 probe，但 `0.00425` 等常数和标题只提供弱公共证据 |
| P2 | 静态预计算 submission 搜索 | 当前通常否 | fleongg 模型存在时走模型推理；若模型缺失，CSV 也必须与活动 sample ID 完全一致才会使用 |
| P3 | 输出 schema/ID 顺序错误 | 是 | 最终 audit 会 fail-fast，现有防护充分 |

## same-ID 与可见 demo 泄漏

### 会在可见 3 井触发的路径

当前可见测试井都能在 `train/` 找到同 ID 训练副本，以下逻辑因此会触发：

1. selector 主路径先判断 `wid in train_wells`，然后读取同 ID 的训练水平井，将训练 `TVT_input` 复制给测试对象，并用训练井 `TVT`、formation 列和 typewell 直接生成 `tvt_phys`。输出时只要 `tvt_phys` 存在，就优先于 PF selector。
2. 两套 learned/ridge 特征构造器对测试井设置 `self_wid=None`。formation KNN 和 row-level ANCC KNN 因而不会排除同 ID 训练井；在可见 demo 中，同井通常成为极近邻。
3. guarded overlap override 会读取同 ID 训练井的 `TVT` 与 formation 列，先用可见前缀验证，再按 MD 插值覆盖预测区。
4. visible-prefix candidate pool 会：
   - 从同 ID 训练井构造 `contact_md_lookup*`；
   - 用没有排除同 ID 的 formation/ANCC imputer 生成 surface 候选；
   - 在最后又重新应用 guarded contact override。

这些路径不是“无条件抄答案”：guard 使用了可见前缀验证，且只在训练 MD 范围内覆盖；但训练井的隐藏区 `TVT` 和 formation 仍参与了候选或最终值，因此不能作为未知井泛化证据。

### 隐藏约 200 个新 ID 时不会触发的路径

若 Kaggle 隐藏井 ID 不与训练集重叠，则：

- `wid in train_wells` 为假，直接 `tvt_phys` 不生成；
- overlap dry-run 只报告 0 个候选，不改提交；
- guarded override 和 visible-prefix contact lookup 都跳过；
- 静态 public CSV 因 ID 不匹配而跳过；
- formation KNN 仍使用训练井 formation，但只是正常的跨井空间先验，不会包含“当前同井”。

因此，same-ID 逻辑主要污染本地可见 demo 的解释，不等于隐藏重跑一定泄漏。正式验证必须使用按井后缀回测，并对 query well 做 self-exclusion。

## 训练目标与 formation 使用

### 正常、应保留的使用

- 训练 `TVT` 用于训练 LightGBM/CatBoost/Ridge/外部模型包，是监督学习的正常目标使用。
- 测试 typewell 自带的 `TVT, GR, Geology` 是比赛输入，可用于 GR 对齐和地层接触点。
- 训练井 formation 列用于建立跨井空间表面/KNN 先验，只要验证时排除 query well，就是合理的训练特征。
- visible prefix 的 `TVT_input` 是比赛明确提供的已知前缀，可用于 heel calibration、prefix backtest 和轨迹锚定。

### 需要隔离的使用

- 同 ID 训练井的完整 `TVT` 与 formation 不应进入测试井最终值，即使可见前缀验证通过。
- learned/ridge 特征的测试分支也应传 `self_wid=wid`；对隐藏新 ID 没有行为变化，却能保证可见 demo 与回测不走特殊捷径。
- `train_df['target'].mean()` 只应作为最后的缺失 fallback；正式输出应要求 fallback 行数为 0，并在 audit 中显式记录。

未发现默认隐藏路径直接读取测试水平井的 `TVT` 列；隐藏测试预计也不提供该列。读取训练真值的 CV/oracle/full-stack 诊断均由 `RUN_CV_REPORT=False`、`RUN_FULL_STACK_CV_ABLATION=False` 等开关关闭。

## public-LB bias 与 probe

未发现 notebook 内部调用 Kaggle API、读取排行榜、自动提交、按返回分数调参或把已知公开标签写入 submission。

`RUN_OVERLAP_DRY_RUN_PROBE=True` 名称里虽有 probe，但实际只统计 train/test 同 ID 和前缀行数，写 `overlap_layer_probe.csv`；它不是 leaderboard probe，也不改预测。

仍有选择偏差风险：

- 标题的 `6.491` 只是公开榜证据，公开榜仅覆盖部分隐藏井。
- selector 阈值、profile 权重、`SP45_BLEND_WEIGHT=0.60`、visible-prefix 门槛和特别精细的 `MODEL_PACKAGE_GATED_MAX_WEIGHT=0.00425` 均可能经过多次公共实验选择。
- `PP.w_sub1` 注释明确提到 “LB transfer”。
- 当前默认 profile 与 markdown 中描述的默认 `contact_gated_anchor` 不一致。
- 最后的 PF midpoint hedge cell 无历史执行计时；不能确认它包含在 `6.491` 对应版本中。

所以 public 分数只能用来排序实验优先级，不能替代 group-wise backtest、最坏井分位数和灾难井审计。

## 外部资产

### 当前会使用

| 数据集 | 作用 | 风险 |
|---|---|---|
| `phongnguyn23021656/koolbox-offline` | 离线安装/导入 `koolbox.Trainer` | wheel 兼容失败时有本地 fallback；低 |
| `ravaghi/wellbore-geology-prediction-artifacts` | 约 7.39 GB `data/train.csv` 与 5 个 trainer pickle | 大加载时间/内存；pickle 可执行代码；环境版本兼容 |
| `fleongg/rogii-claude-models-pub` | `features.json` 与 3 个 LGBM pickle | pickle 兼容/供应链；模型训练过程只能由发布者说明 |
| `pilkwang/rogii-model-package` | feature builder、4 类 tree 模型、TCN、blend 配置 | 当前 profile 开启；额外特征构建和 CPU TCN 推理可能推高运行时 |

model package manifest 声明：

- 训练目标为 delta；
- 3,783,989 个 OOF 行、5 fold；
- formation imputer 对 query well 做 self-exclusion；
- 隐藏推理受支持。

下载检查的 feature builder 确实以 `include_target=False` 构造测试特征，并设置 `exclude_query_well_from_formation=True`。这比 notebook 自带的 learned/ridge 测试特征更安全。不过这只证明接口设计，不证明外部模型文件的训练全过程和分数声明。

### 已挂载但当前源码未使用

- `nina2025/rogii-03`
- `thbdh5765/rogii-v10-fresh-artifacts`
- `needless090/rogii-tabicl-mirror`

建议 v2 从 metadata 移除未使用数据集，减少挂载失败、审计面和误用静态 public 预测的机会。

## 确定性

### 已确定播种

- selector 的 Python PF 使用 `np.random.default_rng(seed)`，按 `0..n_seeds-1` 固定遍历。
- learned 分支的 likelihood PF 在 Numba 内执行 `np.random.seed(seed_base + s)`。
- GroupKFold 本身确定性；GBDT/CatBoost 配置带 seed。
- model-package 的 row ANCC sampler 使用 manifest seed。

### 未确定播种

- 两套自带特征管线中的 `_pf_ancc`、`_pf_z` 直接使用 Numba/NumPy 全局随机数。
- JIT warm-up 也调用随机数。
- 这些函数在 `joblib.Parallel(..., prefer='threads')` 中运行，调用顺序和 RNG 状态可能变化。
- GPU LightGBM/CatBoost 即使设 seed，也不保证所有环境下 bitwise 一致。

v2 应把 seed 作为 Numba kernel 参数，并用稳定的 `sha256(well_id)` 派生逐井 seed；不要使用 Python 内置 `hash()`。如果继续线程并行，kernel 内部必须自行 `np.random.seed(well_seed)`。

## 9 小时运行限制

历史 cell 元数据合计约 702 秒，整次 papermill 为 776 秒。主要耗时：

| 阶段 | 3 个可见井历史耗时 |
|---|---:|
| 读取 7.39 GB 训练特征 + 构造测试特征 | 269 s |
| 128-seed selector + beam | 177 s |
| learned 分支测试特征/模型 | 61 s |
| visible-prefix calibration | 108 s |

只把后三项按 200/3 放大，约为 6.4 小时；还未包含：

- cell 16 中随隐藏井数增长的部分；
- 当前默认开启但历史时长仅 `0.093 s`、显然当时未实际运行的 model-package；
- TCN 强制 CPU 推理；
- 更多输出 CSV、I/O、隐藏数据规模差异和 Kaggle 资源抖动。

因此原样隐藏重跑有 P0 超时风险。正式版本应以 8 小时为硬预算，保留至少 1 小时安全余量。

### 200 井场景估算

下表是基于 3 井 cell 时长的工程估算，不是 benchmark。7.39 GB CSV 加载有较大固定成本，而 PF/beam/visible-prefix 基本随井数增长，因此给区间而不是单点：

| 方案 | 约 200 井估算 | 结论 |
|---|---:|---|
| 当前 exact source：128 seeds + visible-prefix + model package + hedge | 7.5–10.5+ h | 不安全；model package 无历史实测，可能更慢 |
| 关闭 model package，保留 128-seed visible-prefix | 6.5–8.5 h | 边缘；资源抖动即可超时 |
| 再关闭 visible-prefix calibration | 4.8–6.8 h | 大概率低于 8 h |
| 再把两处 128-seed PF 降到 64 | 3.5–5.5 h | 建议的首个隐藏规模 v2 |
| 64 seeds，保留降配 visible-prefix（12/24 seeds、250 particles） | 4.5–6.5 h | 可作为第二个 v2，仍需完整 dry-run |

开关的主要节省量：

- `RUN_MODEL_PACKAGE_CORRECTION=False`：未知，保守估计节省 1–3 小时；这是当前最大不确定项。
- `RUN_VISIBLE_PREFIX_CALIBRATION=False`：按历史约节省 2 小时。
- `SP45_SELECTOR_N_SEEDS: 128 → 64`：selector 阶段约节省 1.5–1.7 小时。
- learned 分支 `CFG.PF_SEEDS: 128 → 64`：约节省 0.3–0.7 小时。
- overlap probe、guarded override、midpoint hedge 本身耗时很小；关闭它们主要是防泄漏和防未验证改动，而不是加速。

任何运行时估算都必须用一次接近隐藏规模的完整 code run 校准。v2 应在日志中输出每阶段耗时和滚动秒/井。

## 输出格式与失败模式

优点：

- 最终强制 `['id', 'tvt']`；
- 行数必须等于 sample；
- ID 顺序必须逐行相同；
- TVT 必须全部有限；
- 写出 SHA-256 和 named copy。

尚缺：

- 没有拒绝异常 TVT 范围或异常单步跳变；
- 没有记录 fallback 行数/比例；
- 没有检查每个 well 的 sample 覆盖和行索引是否连续；
- model-package 根目录存在但文件/模型版本不兼容时会让整个 kernel 失败，而不是自动退回已审计 base submission；
- 最后的 PF midpoint hedge 无总开关，任何异常都发生在 model-package 之后、最终 audit 之前。

v2 应将每个可选层包装为“先生成 candidate → 完整 audit → 原子替换 final”；发生异常、超时预警或 coverage 不一致时保留上一个合法 base。

## 必须保留/禁用的配置

### 必须保留

```python
RUN_CV_REPORT = False
RUN_FULL_STACK_CV_ABLATION = False
RUN_EXACT_MATCH_RECOVERY = False
MODEL_PACKAGE_ALLOW_AUTO_SEARCH = False
MODEL_PACKAGE_REQUIRE = False
```

同时保留：

- 最终 `submission_audit.json`；
- sample ID 精确对齐；
- finite 检查；
- GroupKFold/self-exclusion 回测；
- visible-prefix 的 target-free 校准思想；
- 失败时回退到最近一个已审计提交。

### v2 默认禁用

```python
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_MODEL_PACKAGE_CORRECTION = False
RUN_PF_BRANCH_HEDGE = False  # 新增总开关，包住当前 cell 53
ALLOW_SAME_ID_TRAIN_LOOKUP = False  # 新增，并同时约束所有 contact/physical 路径
```

另外建议：

- 所有 test feature builder 均传 `self_wid=wid`；
- metadata 移除 3 个未使用 dataset；
- 在未完成 200 井规模 dry-run 前，不启用 model package；
- 未做独立 prefix backtest 前，不使用 precise `0.00425` 修正和最后的 midpoint hedge。

## v2 安全修改建议

按优先级：

1. **彻底隔离 same-ID 路径。** 新增一个默认 `False` 的总开关，覆盖 cell 27、formation KNN、guarded override、visible-prefix contact candidate 和 contact reapply；测试特征始终 self-exclude。
2. **先保运行时。** 默认改为 `vp_balanced_final` 或单纯 anchor，关闭 model package 和最终 hedge；将 optional 层设为可跳过。对约 200 井做一次不提交的完整 code run，要求 `< 8h`。
3. **删除重复计算。** 当前 ridge/selector/learned/visible-prefix 多次计算 PF 和 beam。按 `well_id + 参数 + prefix_cut` 缓存；至少复用 full-prefix PF。
4. **显式逐井播种。** 为所有 Numba PF 加稳定 seed；将使用全局 RNG 的 warm-up 改为固定输入或独立 seed。
5. **增加预算守卫。** 每完成若干井记录滚动秒/井，若预计超过 8 小时，则跳过 model-package/visible-prefix optional 候选并输出最近已审计 base。
6. **若必须保留 visible-prefix PF，先降成本做 v2 smoke。** 可从 `CAL_SEEDS=12`、`FINAL_SEEDS=24`、`PARTICLES=250` 起步；任何降配都必须用相同的 group-wise prefix backtest 比较灾难井，而不是只看均值。
7. **增强 audit。** 写出 fallback_rows、per-well coverage、TVT min/max、最大相邻跳变、每层改动的 RMSE/p95/max，以及最终实际启用的 profile/config 哈希。
8. **保留公共复现与正式方案两条线。** 不修改当前 attributed v1；另建 v2 notebook，避免把公开作者的精确复现和我们的安全改动混成一个不可追溯版本。

### v2 最小改动清单

为了先得到一个预计 3.5–5.5 小时、可安全隐藏重跑的版本，最少只改这些位置：

```python
# control/profile 解析之后强制覆盖
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_VISIBLE_PREFIX_CALIBRATION = False
RUN_MODEL_PACKAGE_CORRECTION = False

SP45_SELECTOR_N_SEEDS = 64
SELECTOR_PF_SEEDS = 64

# 第二个 learned-trajectory CFG
CFG.PF_SEEDS = 64

# 新增并包住最后 midpoint hedge cell
RUN_PF_BRANCH_HEDGE = False

# 新增并同时约束 cell 27 / guarded contact / gold contact
ALLOW_SAME_ID_TRAIN_LOOKUP = False
```

再做两处一行级行为修正：

```python
# 两套 build_well 测试特征都排除同 ID 训练井
swid = wid

# 最终 audit 新增
assert fallback_rows == 0
```

注意：只把 `SUBMISSION_PROFILE` 改成 `contact_gated_anchor` 并不能省掉 visible-prefix 计算，因为该 preset 仍把 `run_visible_prefix_calibration=True`，只是最后选择 anchor。要真正节省约 2 小时，必须显式把 `RUN_VISIBLE_PREFIX_CALIBRATION=False`。

若第一个 v2 完整运行 `< 6.5h`，第二个 v2 可只加回降配 visible-prefix：

```python
RUN_VISIBLE_PREFIX_CALIBRATION = True
VISIBLE_PREFIX_CAL_SEEDS = 12
VISIBLE_PREFIX_FINAL_SEEDS = 24
VISIBLE_PREFIX_PARTICLES = 250
```

model package 和 midpoint hedge应最后才逐层加回；它们当前都没有与 `6.491` 绑定的可验证执行证据。

建议的正式提交流程是：

1. v1 只验证“能完成、能生成合法 submission、外部资产能加载”；
2. v2 关闭 same-ID、model-package 和 unvalidated hedge，完成隐藏规模 dry-run；
3. 以 v2 为主提交；
4. 只有在运行时、按井 prefix backtest 和输出审计都通过后，再逐层加回 model package 或 hedge。
