# ROGII 井眼地质预测：Kaggle 奖牌冲刺

这是 [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
竞赛的完整研究与复现仓库。项目目标是在很短的赛程内建立一个可审计、可复现、能在
Kaggle 无网络 Notebook 环境中稳定运行的地质轨迹预测系统，并尽可能取得奖牌。

> 仓库设为 **Private**。竞赛原始数据、模型权重、提交文件、排行榜快照、Kaggle
> 凭据和运行产物不会提交到 Git。

## 1. 当前状态

截至 2026-07-27：

| 项目 | 状态 |
|---|---:|
| 已加入竞赛并接受规则 | 完成 |
| 训练井 / 可见测试井 | 773 / 3 |
| 当前最佳 Public RMSE | **6.477** |
| 当前公开榜排名 | **701 / 5,785，前 12.12%** |
| 公开榜前 10% 边界 | 6.467 |
| 公开榜前 5% 边界 | 6.432 |
| 当前奖牌判断 | 暂处公开榜铜牌区外，最终奖牌未确定 |
| A28 PF-1.3 w0.62 | 隐藏重跑超时，无分数 |
| A28/Q0522 50/50 融合 | 隐藏重跑超时，无分数 |
| Frontier II DYNQ0196 | **7.507**，较最佳差 1.030，NO-GO |
| GPU TCN 残差小试 | pooled 改善但逐井覆盖不足，门槛 FAIL |
| Harshini 空间曲面 + PF | 严格空间核心 17.944，差于 flat 14.969，NO-GO |

当前 6.477 比公开榜前 10% 边界差约 0.010。公开榜只覆盖约 26% 测试数据，最终排名
由约 74% 的 Private 数据决定，所以项目不会把公开榜位置当作最终结论，也不会为了
小幅 Public 提升牺牲跨井稳定性。

详细时间线见 [进展日志](docs/progress_log.md)，最终选择依据见
[候选选择记录](docs/selection.md)。

## 2. 竞赛信息

| 项目 | 内容 |
|---|---|
| 任务 | 沿水平井轨迹预测连续目标 `tvt` |
| 指标 | RMSE，越低越好 |
| 类型 | Featured Code Competition，计 Kaggle 积分与奖牌 |
| 奖金 | 50,000 USD |
| 最终截止 | 2026-08-05 23:59 UTC |
| 北京时间截止 | 2026-08-06 07:59 CST |
| 提交形式 | Kaggle Notebook 生成 `submission.csv` |
| 运行限制 | CPU/GPU 最长 9 小时，提交运行时关闭互联网 |
| Public / Private | 约 26% / 74% |

最终文件必须严格满足：

- 两列且顺序固定为 `id,tvt`；
- ID 与官方 `sample_submission.csv` 完全同序；
- 每个 ID 唯一；
- 所有预测值均为有限浮点数；
- Notebook 在隐藏测试规模变化时仍可运行，不能写死公开测试井或行数。

RMSE 为：

```text
RMSE = sqrt(mean((tvt_pred - tvt_true) ** 2))
```

## 3. 数据理解

每口井主要包含两类有序数据：

- `horizontal_well.csv`：沿实钻轨迹按测深排列的测井、空间坐标、地质接触面以及已知
  前缀 `TVT_input`；
- `typewell.csv`：邻近参考井的 TVT—测井曲线，可作为地层模板。

预测对象不是独立表格行，而是具有连续性、局部倾角和地层重复模式的空间序列。项目
因此把问题建模为“沿井轨迹追踪地层位置”，而不是普通的逐行回归。

完整数据审计结果：

- 773 口训练井；
- 3 口当前可见测试井；
- 训练后缀合计约 378 万评分行；
- 官方压缩包共 2,327 个文件，已完成 CRC 校验；
- 已检查文件覆盖、列类型、缺失值、行顺序、ID 映射和样例提交覆盖；
- 可见测试井与训练数据存在同井关系，因此随机行切分和同 ID 真值恢复会严重泄漏。

详见 [数据审计](docs/data_audit.md)。

## 4. 总体思路

系统由五个层次组成：

```text
Typewell / Horizontal Well
          |
          v
GR 似然匹配 + 几何/倾角特征
          |
          v
粒子滤波 / PF 轨迹候选
          |
          v
Ridge/树模型锚点 + 鲁棒 U=TVT+Z 投影
          |
          v
学习型 PF-1.3 轨迹与保守融合
          |
          v
可见前缀校准、模型包守门、分支不确定性处理
          |
          v
严格 submission 契约审计
```

核心原则：

1. **物理与序列优先**：利用测深、垂深、Gamma Ray 与参考井曲线追踪连续地层。
2. **按井验证**：所有主要验证按井、类型井指纹或空间簇划分，拒绝随机行 CV。
3. **分阶段守门**：每个后处理层必须单独证明稳定收益，否则保留为关闭状态。
4. **Public 榜只做探针**：不根据少量榜单结果反复调阈值。
5. **可复现优先**：随机 PF 使用稳定的“井 ID + 分支”种子，Notebook 输出经过哈希审计。

## 5. 核心算法

### 5.1 粒子滤波与 Typewell GR 匹配

粒子表示未知后缀上可能的 TVT 轨迹。每一步根据井轨迹几何传播，再用水平井 Gamma
Ray 与 Typewell Gamma Ray 的匹配误差更新权重：

```text
particle propagation:
    tvt_t = tvt_(t-1) + structural_dip * delta_md - delta_z + noise

likelihood:
    weight ~ exp(-0.5 * ((GR_hw - GR_typewell(tvt_t)) / sigma_gr) ** 2)
```

通过多随机种子和多个似然尺度形成轨迹分布，降低单次 PF 抽样的不稳定性。项目分别
研究了 64、96、128 和 256 种子配置，并为 Numba 随机内核加入确定性种子。

### 5.2 Ridge/PF 锚点与鲁棒投影

基础锚点融合 Ridge/树模型输出和 PF selector：

```text
T_anchor = 0.30 * T_ridge + 0.70 * T_selector
```

随后在归一化测深上拟合隐含地层标高：

```text
U = TVT + Z
T_projected = 0.75 * robust_poly(U) - Z + 0.25 * T_anchor
```

使用迭代鲁棒权重抑制局部异常点，使轨迹满足更合理的地层连续性。

### 5.3 学习型 PF-1.3 轨迹

另一条分支使用 LightGBM、CatBoost 等树模型学习相对末端已知 TVT 的轨迹残差，并
结合多尺度 PF、局部几何、测井相似度和沿井位置特征。主候选把投影锚点和学习型轨迹
进行保守融合：

```text
T_final_base = w_sp45 * T_projected + (1 - w_sp45) * T_learned
```

已测试的核心权重包括 0.60 和 A28 的 0.62。

### 5.4 可见前缀校准与接触面守门

利用测试井自身已知前缀进行“伪后缀”回放，只允许能在前缀遮挡实验中自证的校准层
移动最终预测。接触面重建同样需要先通过前缀 RMSE 门槛。所有同井真值直查、固定
公开井 ID 和固定行数逻辑都被视为高风险路径并单独审计。

### 5.5 分支不确定性与 +2 ft Hedge

PF 种子可能形成两个地层分支。当前 6.477 全栈在一个高分离度井上选择了 +2 ft
分支修正。删除该层的严格消融得到 6.604，变差 0.127，因此在当前主候选中保留。

Q0522 根据运行时 PF 分支报告确定方向，再把分支延伸 0.522 ft；A28/Q0522 融合用于
降低单一分支判断的风险。A28 和 Pair 的可见数据普通运行都能完成，但竞赛隐藏重跑
均超过允许时间，因此当前封装是运行时 **NO-GO**，没有获得榜单分数。

### 5.6 残差 GBDT 与竞争性 OOF

我们额外训练空间不变的残差模型，并冻结 0.20 shrink：

- 类型井指纹五折；
- 两套空间聚类协议；
- 额外七空间簇协议；
- fold 一致性、最坏 fold、well-level p95 和 bootstrap 区间。

残差 v3 在四套独立 OOF 协议中均改善，但叠加到真正有竞争力的 Ridge/PF OOF 后未
通过预注册门槛：类型井协议只改善约 0.0115，七空间簇反而变差约 0.0464。因此该路线
是 **NO-GO**，没有消耗提交次数。

### 5.7 GPU TCN 序列残差小试

树模型残差在简单基线上有效、但在强 Ridge/PF OOF 上没有稳定增益，因此下一条路线
不是替换主轨迹，而是学习与强基线误差低相关的局部序列修正。当前私有 TCN 小试使用：

- 378.4 万个训练后缀行和 773 口训练井；
- 51 个不含绝对 X/Y/Z、井 ID 和标签信息的目标无关特征；
- 水平井相对几何、沿井进度、GR 局部统计、Typewell GR 匹配和边界斜率；
- 六层膨胀 1D 卷积，dilation 为 `1,2,4,8,16,32`；
- `tanh` 限幅到 ±30 ft，主候选固定只叠加 0.20 倍预测残差；
- 七空间簇中的完整 fold 0 作为验证集，并从训练中排除所有相同 Typewell 指纹。

这一步是成本受控的资格赛：主 scale 必须相对竞争性 OOF 至少改善 0.02 RMSE，同时
well-level p95 不恶化超过 0.10、最坏井不恶化超过 0.50，且至少一半验证井不变差。
未通过则停止；通过也只授权继续多 fold 验证，**不会直接授权 Kaggle 提交**。特征缓存
只上传到账号自己的私有 Kaggle Dataset，Git 仓库只保留构建和训练代码。

T4 v2 在 81.3 秒内完成 4,000 steps。固定 0.20 主 scale 在 899,519 个验证行上把
竞争性基线从 9.63958 改善到 9.58436，delta 为 -0.05522；well-level p95 和最坏井
也分别改善 0.075 和 1.150。但只有 **44.71%** 的 170 口验证井不变差，未达到冻结的
50% 门槛。最终判定为 **FAIL / NO-GO**：不继续多 fold，不构建竞赛提交。这表明 TCN
收益集中于少数大井，尚不具备足够广泛的跨井稳定性。

### 5.8 Frontier II 运行时快速探针

Frontier II 的 48 个评分代码单元中，44 个与现有精确全栈逐字一致。它关闭了两个在
公开运行中不改变最终字节、但会增加隐藏重跑时间的校准层，并根据运行时 PF 分支报告
追加有符号的 0.196 ft 修正。审计确认：

- 修正前文件 SHA-256 与 6.477 主候选完全一致；
- 评分事务不写死公开测试井 ID、行数或输出哈希；
- 公开运行约 643 秒；
- 作者在可视化清单中自报的 6.39 不作为真实成绩证据。

该版本的受控 Code Submission 最终得到 **7.507**。它比 6.477 主候选差 1.030，比
前 10% 边界 6.469 差 1.038，也只比 Safe PF v2 的 7.514 改善 0.007。作者清单中的
6.39 因而没有被 Kaggle 成绩验证。Frontier II 是明确的 **NO-GO**，不会替换主候选，
也不会围绕 DYNQ0196 继续调参。

### 5.9 Harshini 空间曲面 + PF 审计

公开 Notebook `luffyh04/harshini-submission-f` 是本轮公开扫描中唯一具有新结构的候选：
它从 765 口可用训练井重建六个地层曲面，融合多尺度 PF、LightGBM、CatBoost 和非负
Ridge。作者的按井随机五折报告为 flat 15.589、固定核心 9.048、最终 Ridge 8.741。
代码审计确认 `EXACT datum` 只用于上限诊断，没有进入最终 `P/PR` 特征；单井 LOO
也确实排除了目标井。

问题在于随机按井五折仍允许同一验证区域内的其他井向目标井提供带真值的地层曲面。
为此新增 `scripts/audit_harshini_spatial_core.py`，使用七空间簇并在每折中从曲面点云
整体删除验证簇。最大的 fold 0 覆盖 173 口井、910,183 个后缀行，结果为：

| 轨迹 | 严格空间 fold 0 RMSE |
|---|---:|
| Flat | 14.969 |
| PF | 8.598 |
| 原作者单井 LOO 固定核心 | 8.008 |
| 删除完整验证区域后的曲面分支 | 30.278 |
| 删除完整验证区域后的曲面 + PF 核心 | **17.944** |

验证区域删除后，最近训练曲面的平均距离升至 17,389 ft；固定核心相对 flat 变差
2.975，相对原 LOO 核心变差 9.937。该候选可见运行还用了约 12,464 秒（3.46 小时），
隐藏规模同时存在九小时超时风险。因此不继续昂贵的 boosted 多折，也不创建 Kaggle
提交。完整方法和限制见
[Harshini 空间核心审计](docs/harshini_spatial_core_audit.md)。

## 6. 防泄漏验证

随机行切分会让同一口井的相邻行同时进入训练和验证，得到虚假的高分。项目采用：

- **按井 GroupKFold**：整口井只属于一个 fold；
- **Typewell fingerprint 分组**：相似或相同参考井模板不会跨 fold；
- **空间聚类留出**：按井位置聚类，模拟区域外推；
- **可见前缀 backtest**：在单井内部人为截断已知前缀，模拟真实预测后缀；
- **逐井稳定性门槛**：不仅看 pooled RMSE，还看 fold delta、最坏井和 p95；
- **竞争性基线门槛**：候选必须改善当前强 PF/Ridge OOF，而非只改善 hold-last。

所有实验先写明假设和通过门槛，再读取验证标签，避免在同一份标签上反复调参。

## 7. 已完成实验

| 候选 | Public RMSE | 结论 |
|---|---:|---|
| Safe PF v2，64 seeds | 7.514 | 干净基线，非奖牌候选 |
| Exact public full stack | **6.477** | 当前主候选 |
| Clean128 deterministic | 7.515 | 可复现但无收益，NO-GO |
| Exact full stack 去掉 +2 ft hedge | 6.604 | 变差 0.127，NO-GO |
| Guard640 model-package | 未提交 | 本地守门失败，NO-GO |
| U-continuity | 未提交 | 改善仅 0.00038，低于门槛 |
| 自适应分支 selector | 未提交 | 空间 OOF 不稳定，NO-GO |
| Residual GBDT v1/v2 | 未提交 | 区域最坏 fold 明显变差 |
| Residual GBDT v3 overlay | 未提交 | 竞争性 OOF 未通过 |
| A28 PF-1.3 w0.62 | 无分数 | 隐藏重跑超时，运行时 NO-GO |
| A28/Q0522 50/50 | 无分数 | 隐藏重跑超时，运行时 NO-GO |
| Frontier II DYNQ0196 | 7.507 | 较 6.477 差 1.030，作者 6.39 未验证，NO-GO |
| Sequence TCN v1/v2 | 不提交 | pooled -0.0552，但仅 44.71% 井改善，NO-GO |
| Harshini 空间曲面 + PF | 不提交 | 严格空间核心 17.944，差于 flat 14.969，NO-GO |

Clean128 的两次 Kaggle 普通运行在四个阶段和最终 CSV 上逐字节一致，证明确定性修复
有效；但其榜单成绩与 Safe v2 基本相同，所以只保留为可靠控制组。

更多分析见：

- [公开候选研究](docs/public_research.md)
- [PF 公开复现审计](docs/pf_public_reproduction_audit.md)
- [Exact128 分阶段分析](docs/exact128_v6d_v1_stage_analysis.md)
- [残差 GBDT OOF](docs/residual_gbdt_oof.md)
- [空间不变残差 OOF](docs/residual_gbdt_invariant_oof.md)
- [Shrink 残差 OOF](docs/residual_gbdt_shrink_oof.md)
- [目标无关 selector CV](docs/targetfree_selector_cv.md)
- [GPU TCN 小试设计](docs/sequence_tcn_pilot.md)
- [Harshini 空间核心审计](docs/harshini_spatial_core_audit.md)

## 8. 仓库结构

```text
.
├── src/rogii/              # 可复用的数据审计、轨迹、残差与提交守门模块
├── scripts/                # 构建 Notebook、运行 OOF、分析阶段输出的命令行脚本
├── tests/                  # 单元测试与契约测试
├── kaggle_notebooks/       # 可推送到 Kaggle 的 Notebook 源与元数据
├── experiments/            # 单次实验的假设、配置、结果和决策
├── docs/                   # 数据审计、策略、研究和进展文档
├── pyproject.toml          # Python 3.12 依赖与工具配置
└── uv.lock                 # 锁定依赖
```

以下目录被 Git 忽略：

```text
data/ artifacts/ models/ outputs/ submissions/ leaderboards/
.venv/ .kaggle/ catboost_info/
```

## 9. 本地复现

环境要求：

- Python 3.12；
- `uv`；
- 已接受竞赛规则的 Kaggle 账号；
- 本地 Kaggle 凭据，但凭据不得放入仓库。

安装依赖：

```powershell
uv sync --dev
```

运行完整测试和静态检查：

```powershell
uv run pytest -q
uv run ruff check src scripts tests
git diff --check
```

构建示例 Notebook：

```powershell
uv run python scripts/build_pf_exact128_v6d.py
uv run python kaggle_notebooks/pf_exact128_v6d/validate_notebook.py
```

构建当前 A28 私有验证 Notebook：

```powershell
uv run python scripts/build_frontier_a28_v1.py
```

生成 TCN 私有训练缓存并构建 GPU Notebook：

```powershell
uv run python scripts/build_sequence_pilot_cache.py
uv run python scripts/build_sequence_tcn_pilot_notebook.py
```

复现 Harshini 固定核心的严格空间 fold 0 审计：

```powershell
uv run python scripts/audit_harshini_spatial_core.py `
  --fold 0 `
  --output artifacts/harshini_spatial_core_fold0.json
```

缓存约 397 MB，写入 `artifacts/sequence_pilot_cache_v1/`，受 `.gitignore` 保护。GPU
Notebook 的源码保存在 `kaggle_notebooks/sequence_tcn_pilot_v1/`；Kaggle 运行 ID 为
`heralces/rogii-sequence-tcn-pilot-v2`，它只写 OOF 报告和 checkpoint，不写
`submission.csv`。

由于竞赛已采用 Code Submission，不能直接上传本地 CSV；最终候选必须先由 Kaggle
Notebook 运行生成 `submission.csv`，再提交对应 Notebook 版本。

## 10. 安全与合规

- 不提交 `kaggle.json`、访问令牌、`.env` 或任何 GitHub/OpenAI 凭据；
- 不提交竞赛原始数据、外部模型权重、排行榜完整快照和预测 CSV；
- 对公开 Notebook 保留来源说明，标题中的分数不自动视为真实成绩；
- 不把公开演示井的同 ID 真值诊断当作隐藏测试证据；
- 候选必须通过动态样本长度、有限值、ID 同序和唯一性检查；
- 不声称已经获得奖牌，直到 Private Leaderboard 正式结算。

## 11. 下一步

1. 保留 6.477 为当前主候选，不重复提交已超时的 A28/Pair 版本；
2. 只有在完成隐藏规模预算分析和明确的运行时缩减后，才考虑重建 A28；
3. Dip-aware HMM + GBM 已取消且无产物，不根据未经验证的 5.14 声称提交；
4. Frontier II 已以 7.507 判定 NO-GO；不根据作者自报的 6.39 重复调参；
5. TCN v2 已按预注册门槛停止；后续序列路线必须先解决逐井覆盖不足，不直接调 scale；
6. Harshini 曲面路线已按严格空间门槛停止；不运行 3.46 小时 boosted 多折或隐藏提交；
7. 8 月 2 日前冻结至少两个可复现且结构不同的最终 Notebook；
8. 8 月 5 日 20:00 UTC 前完成最终选择，保留四小时故障恢复窗口。
