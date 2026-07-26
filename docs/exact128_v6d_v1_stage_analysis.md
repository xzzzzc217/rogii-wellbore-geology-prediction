# Exact128 v6d v1 普通运行与阶段诊断

日期：2026-07-24（Asia/Shanghai）

## 结论

`heralces/rogii-exact128-v6d` version 1 的普通 visible-data Kaggle
运行于 23:53 CST 正常完成。完整运行、动态样本契约和独立严格审计均通过；
这证明 notebook 可以按冻结参数安全执行，但不构成 hidden 分数或奖牌证据。

- Kaggle 状态：`COMPLETE`
- 版本：仅 version 1；未创建 version 2
- 竞赛提交：无
- Kaggle 下载产物：40 个
- 本地冻结目录：`artifacts/kaggle_runs/pf_exact128_v6d_v1`
- 最终 `submission.csv` SHA-256：
  `9b4915c7c46d10568da5a7a8f2e4de86a0bcefc15dce3965bcab23d0a49e9f30`
- 独立严格审计：14,151 行，0 error，0 warning
- 阶段分析：`artifacts/pf_exact128_v6d_v1_stage_analysis.json`

当前结论是：**技术执行 PASS；hidden submission 仍为 NO-GO / 未授权。**
本次没有修改 notebook 源码、推送 v2 或提交竞赛。

## 实际冻结配置

运行日志和 `exact128_v6d_scoring_path_audit.json` 共同确认：

| 项目 | 实际值 |
|---|---:|
| Selector PF | 128 seeds / 500 particles |
| Learned PF | 128 seeds |
| Projection | degree 3 / weight 0.75 |
| Clean blend | SP45 0.60 / learned 0.40 |
| Visible-prefix profile | balanced |
| VP calibration / final seeds | 24 / 48 |
| VP particles | 350 |
| VP cut fractions | 0.50 / 0.65 / 0.75 |
| Model-package max gate / scale | 0.00425 / 6.0 |
| Model-package raw-diff p95 guard | 25.0 ft |
| Generalized branch hedge | enabled |
| U-continuity | disabled |
| Hard budget / final reserve | 27,000 / 1,800 s |
| Deterministic namespace | `rogii-exact128-v6d` |

主 scoring audit 和 VP audit 都记录
`query_well_self_exclusion=true`。以下 unsafe 控制全部为 `false`：

- same-ID train lookup；
- guarded same-well overlap override；
- overlap dry-run probe；
- exact-match recovery；
- public overlap calibration；
- global LB bias；
- model-package auto-search；
- U-continuity。

## 运行时与预算回退

Kaggle JSON log 的实际事件时间：

| 阶段 | 秒 |
|---|---:|
| Kernel 启动至 nbconvert 完成 | 916.376 |
| Profile control 至最终 audit | 894.519 |
| 初始化、artifact/模型加载及 selector 前准备 | 394.062 |
| Selector + projection | 206.686 |
| Learned trajectory | 76.247 |
| Clean blend 写出 | 0.356 |
| Visible-prefix | 170.414 |
| Model package | 45.993 |
| Branch hedge | 0.157 |
| 最终 contract + audit | 0.308 |

总 wall time 约 15.3 分钟，落在构建前 12–20 分钟估计内。

没有预算回退：

- 三口井均完成 128-seed selector；
- learned stage 正常加载 mounted boosters 并写出 14,151 行；
- VP `budget_guard_stopped=false`；
- model package 实际完成推理；它被 p95 disagreement guard 拒绝，而不是被预算守卫跳过；
- 最终 contract 没有 missing/extra ID。

## Visible-prefix 决策

Balanced VP 完成三口井、148 个候选的前缀校准，但冻结 balanced
规则对三口井的 `alpha` 均为 0：

| Well | Best candidate | Gain | Rank margin | Consistency | Balanced alpha |
|---|---|---:|---:|---:|---:|
| `000d7d20` | `surface_weighted_prefix` | 1.1613 | 0.0147 | 0.6667 | 0 |
| `00bbac68` | `pf\|pf_scale_12_hold_0` | 3.4099 | 0.0202 | 1.0000 | 0 |
| `00e12e8b` | `surface_EGFDU_late` | 2.3480 | 0.0005 | 1.0000 | 0 |

因此 balanced VP 实际移动 0 行，`after_visible_prefix` 与 clean anchor
完全相同。Contact override 为 false；上表中的 surface 候选来自
query-self-excluded spatial imputer，不是 same-ID contact reconstruction。

## Model-package 决策

Model package 正常运行并产生 active signal，但与当前 anchor 的原始差异为：

- mean absolute disagreement：17.0401 ft；
- p95 disagreement：**28.0500 ft**；
- maximum disagreement：31.0968 ft。

由于 `28.0500 > 25.0`，全局 disagreement guard 正确禁用了整个 package
层。实际 selected composition 移动 0 行。

作为冻结后的三井诊断，未选择的精确 `0.00425 / 6` 候选平均移动仅
0.00820 ft、p95 0.01248 ft，并使 pooled RMSE 从 4.701843 变为
4.699126（-0.002717）。这个极小差异只属于三井 post-hoc diagnostic；
不能用于取消 target-free p95 guard，也不是 hidden 证据。

## Branch-hedge 决策

| Well | Decision | Separation | Shift | Moved rows |
|---|---|---:|---:|---:|
| `000d7d20` | skip separation | 0.2802 | 0 | 0 |
| `00bbac68` | skip separation | 3.5940 | 0 | 0 |
| `00e12e8b` | applied | 29.4391 | +2.0 ft | 4,301 |

因此最终 composition 只有 hedge 发生实际移动。

## 三井阶段 RMSE：仅诊断

所有预测和运行 audit 已先下载、加载并计算 SHA-256，随后才读取本地
train TVT。以下结果明确属于 regression diagnostic，不是 hidden
分数估计：

| Frozen stage | Pooled RMSE |
|---|---:|
| SP45 projection | 4.267307 |
| Learned trajectory | 5.926245 |
| Clean 0.60 / 0.40 anchor | 4.701843 |
| Balanced VP 后 | 4.701843 |
| 实际 selected model package 后 | 4.701843 |
| 最终 branch hedge 后 | **5.167507** |
| 未选择的 exact package candidate | 4.699126 |

关键诊断：

- 0.60/0.40 blend 相对 SP45 projection 的 pooled RMSE 增加 0.434536；
  它改善 `000d7d20`，但恶化另两口井。
- Balanced VP 和实际 selected package 都移动 0 行。
- Hedge 将 `00e12e8b` 的 RMSE 从 6.284110 增至 7.389924，
  增加 1.105814；pooled RMSE 增加 0.465664。
- 这为该精确 composition 的 hedge 提供负向三井诊断，但三口
  visible same-ID 井不足以单独预测 hidden 表现，也不能据此调参。

## 完整性审计

动态 contract 与独立审计确认：

- 列严格为 `id,tvt`；
- ID 集合、顺序、逐井覆盖和行数完全匹配 mounted sample；
- 无重复、missing、extra、NaN 或 infinity；
- 三口轨迹均非常数；
- 最大相邻跳变为 0.1401 ft，远低于 25-ft 守卫；
- CSV 使用 LF 且有末尾换行；
- contract 后没有再次写 submission。

## 决策边界

本结果可用于普通运行回归检查、安全控制验证和冻结后的阶段归因；
不得用于 hidden 分数估计、围绕三口井调参、leaderboard 或奖牌结论。

在获得独立的全井、target-free 正证据前，不应提交这一 v1 的最终
hedge composition。保留 v1 及全部产物作为冻结普通运行证据。
