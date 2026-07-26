# ROGII 井眼地质预测：Kaggle 奖牌冲刺

这是 [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
竞赛的完整研究与复现仓库。项目目标是在很短的赛程内建立一个可审计、可复现、能在
Kaggle 无网络 Notebook 环境中稳定运行的地质轨迹预测系统，并尽可能取得奖牌。

> 仓库设为 **Private**。竞赛原始数据、模型权重、提交文件、排行榜快照、Kaggle
> 凭据和运行产物不会提交到 Git。

## 1. 当前状态

截至 2026-07-26：

| 项目 | 状态 |
|---|---:|
| 已加入竞赛并接受规则 | 完成 |
| 训练井 / 可见测试井 | 773 / 3 |
| 当前最佳 Public RMSE | **6.477** |
| 当前公开榜排名 | **557 / 5,716，前 9.75%** |
| 公开榜前 10% 边界 | 6.478 |
| 公开榜前 5% 边界 | 6.443 |
| 当前奖牌判断 | 暂处公开榜铜牌区，最终奖牌未确定 |
| A28 PF-1.3 w0.62 | 已提交，评分中 |
| A28/Q0522 50/50 融合 | 已提交，评分中 |

当前 6.477 只领先公开榜前 10% 边界约 0.001。公开榜只覆盖约 26% 测试数据，最终
排名由约 74% 的 Private 数据决定，所以项目不会把公开榜位置当作最终结论，也不会
为了小幅 Public 提升牺牲跨井稳定性。

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

正在评分的 Q0522 根据运行时 PF 分支报告确定方向，再把分支延伸 0.522 ft；A28/Q0522
融合用于降低单一分支判断的风险。

### 5.6 残差 GBDT 与竞争性 OOF

我们额外训练空间不变的残差模型，并冻结 0.20 shrink：

- 类型井指纹五折；
- 两套空间聚类协议；
- 额外七空间簇协议；
- fold 一致性、最坏 fold、well-level p95 和 bootstrap 区间。

残差 v3 在四套独立 OOF 协议中均改善，但叠加到真正有竞争力的 Ridge/PF OOF 后未
通过预注册门槛：类型井协议只改善约 0.0115，七空间簇反而变差约 0.0464。因此该路线
是 **NO-GO**，没有消耗提交次数。

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
| A28 PF-1.3 w0.62 | 评分中 | 受控独立探针 |
| A28/Q0522 50/50 | 评分中 | 分支风险对冲探针 |

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

1. 等待 A28 和 A28/Q0522 两个 Notebook 提交完成评分；
2. 若不优于 6.477，不替换当前主候选；
3. 审计 Dip-aware HMM + GBM 的真实运行产物，必须存在可复核 OOF 才进入候选；
4. 继续寻找与 PF/Ridge 误差低相关、空间 OOF 稳定的序列模型；
5. 8 月 2 日前冻结至少两个可复现且结构不同的最终 Notebook；
6. 8 月 5 日 20:00 UTC 前完成最终选择，保留四小时故障恢复窗口。
