# Target-free 灾难井诊断与 fallback 可行性

数据源：`artifacts/target_free_eval_50.json`，50 口井、227,059 个隐藏区间行。这里的 TVT 真值只用于离线打分和外层验证；下述候选规则在推理时只读取模型自产生的 target-free 诊断量，不按真实误差逐井选模型。

## 结论

存在一个**有希望、可行动但尚不足以直接定版**的二变量信号：

```text
effective_seed_count <= 1.5
AND beam_average_cost <= 2.0
```

其中 `effective_seed_count = 1 / sum(seed_weight**2)`。它表示 PF 权重高度集中在极少数 seed，同时 beam 仍能给出不过高的路径代价。触发时切换到 hold-last（更保守）或 beam（本批回测略强）能挡住几口主要灾难井。

固定阈值在这 50 口井上触发 6 井，pooled RMSE 从 `12.842` 降到：

- fallback 到 hold-last：`10.005`
- fallback 到 beam：`10.138`

但这仍是从同一批 50 井发现的规则，不能当作独立证据。建议先在全部 773 井的确定性回测或一个预先冻结的独立井集上复验；复验前可把它作为实验分支，不建议立刻覆盖主提交。

没有发现稳定的单一诊断量，也不建议使用“隐藏段 GR 拟合误差最小的轨迹”。后者与 TVT 收益几乎不相关，容易被少数井制造的表面提升误导。

## 方法级输赢与 oracle 上限

| 方法 | pooled RMSE | macro-well RMSE | 与 final 的逐井输赢 |
|---|---:|---:|---:|
| final | 12.842 | 9.690 | — |
| hold-last | 16.205 | 12.284 | 19 胜 / 31 负 |
| particle filter | 14.090 | 10.530 | 13 胜 / 37 负 |
| beam Viterbi | 16.066 | 12.375 | 17 胜 / 33 负 |

在 final/hold/PF/beam 四者中逐井最优的次数为：final 17、hold 13、PF 13、beam 7。整体均值会掩盖明显的“不同井需要不同轨迹族”现象。

| 逐井 oracle 候选集 | pooled RMSE | 相对 final 的理论改善 |
|---|---:|---:|
| final + hold | 8.620 | 4.222 |
| final + PF | 12.737 | 0.104 |
| final + beam | 8.818 | 4.023 |
| final + hold + PF + beam | 8.362 | 4.479 |

oracle 使用真实误差，只表示选择器的上限，不能用于提交。它说明主要潜力来自“PF/final 与保守 hold/beam 轨迹族之间的切换”，而不是 final 与 PF 之间的切换。

最差 15 口 final 井如下（粗体逻辑未用于选择；这里只展示事后审计）：

| well | rows | final | hold | PF | beam | 最优 |
|---|---:|---:|---:|---:|---:|---|
| df71d1ff | 6,228 | 38.13 | 12.74 | 41.31 | 15.00 | hold |
| 353e5502 | 6,062 | 27.43 | 5.64 | 30.42 | 6.86 | hold |
| f6d009f4 | 6,715 | 24.32 | 47.04 | 25.86 | 44.90 | final |
| 1fee2b62 | 4,641 | 22.65 | 8.27 | 24.78 | 10.94 | hold |
| 727a3a10 | 4,397 | 20.80 | 12.36 | 23.85 | 12.97 | hold |
| 9af1ffd0 | 3,525 | 19.86 | 18.13 | 28.24 | 18.15 | hold |
| 8a9c8932 | 4,663 | 16.48 | 2.40 | 18.26 | 3.30 | hold |
| abf460d2 | 3,217 | 16.40 | 8.35 | 17.35 | 8.13 | beam |
| 1d3fbf02 | 2,694 | 16.38 | 5.91 | 18.07 | 5.24 | beam |
| 9d4272e3 | 4,030 | 16.22 | 17.77 | 16.19 | 17.40 | PF |
| 12f0cd90 | 5,947 | 13.72 | 7.90 | 14.77 | 8.51 | hold |
| 200d233d | 3,926 | 13.56 | 12.78 | 17.54 | 12.60 | beam |
| 5beded95 | 3,057 | 12.92 | 16.64 | 12.50 | 15.33 | PF |
| 7b38844c | 3,749 | 12.85 | 7.39 | 14.11 | 7.52 | hold |
| ba48188d | 4,166 | 11.82 | 50.94 | 7.56 | 50.12 | PF |

`f6d009f4` 和 `ba48188d` 是重要反例：仅凭 final 轨迹偏移大就硬切 hold/beam，会把原本可接受的结果变成灾难。

## 诊断量相关性

Spearman 相关性中，对 final 绝对误差相对较强的是：

| 可观测诊断 | 与 final MSE 的 Spearman ρ |
|---|---:|
| final 最大锚点偏移 | +0.453 |
| PF posterior std median | +0.420 |
| final 预测跨度 | +0.414 |
| PF posterior std p90 | +0.347 |
| 校准相关系数 | -0.265 |

这些量可以用来识别“不可信”，但不能可靠回答“应切到哪条轨迹”。以 `final MSE - hold MSE` 为收益时，绝对相关性最大的几个量仍很弱：PF std ratio `-0.276`、PF std median `+0.268`、beam terminal margin `-0.250`、校准相关系数 `-0.206`、beam cost `-0.173`。

GR 诊断尤其不可靠：

- 各组件 hidden-GR RMSE 与自身 TVT RMSE 的 Spearman ρ：final `0.019`、hold `-0.047`、PF `-0.014`、beam `0.304`。
- `GR_error(final) - GR_error(hold)` 与 `MSE(final) - MSE(hold)` 的 Spearman ρ 仅 `0.002`。
- 直接在 final/hold 中选 hidden-GR 更小者，虽然本批 pooled RMSE 为 `12.056`，但 17 次切换只有 7 次正确，提升主要由一口大灾难井贡献，不应采用。

## 留一与交叉验证

所有外层验证都只用训练折真值拟合阈值/树，再用验证折的 target-free 诊断做选择。

### 无约束单阈值搜索

让训练折在所有诊断量、方向和阈值中自由选一个规则，出现明显过拟合：

| fallback | LOO pooled RMSE | 30 次随机 5-fold 平均 | 5-fold 优于 baseline 的比例 |
|---|---:|---:|---:|
| hold | 16.570 | 14.986 | 6.7% |
| beam | 16.299 | 14.908 | 10.0% |

因此不能从相关性表中事后挑一个阈值直接上线。

### 预设机制的二变量 AND 规则

只允许搜索 `effective_seed_count <= t1 AND beam_cost <= t2`，每个训练折至少触发 3 井：

| fallback | nested LOO | 20 次随机 5-fold 平均 | 中位数 | 10–90% | 优于 baseline |
|---|---:|---:|---:|---:|---:|
| hold | 11.642 | 11.605 | 11.798 | 10.005–12.457 | 20/20 |
| beam | 10.172 | 10.983 | 10.328 | 10.172–11.878 | 20/20 |

训练折学到的阈值中位数约为：

- hold：`effective_seed_count <= 1.88` 且 `beam_cost <= 1.92`
- beam：`effective_seed_count <= 1.65` 且 `beam_cost <= 1.92`

这与便于解释的冻结阈值 `1.5 / 2.0` 接近，是目前唯一显示出较稳定外推迹象的信号。固定 `1.5 / 2.0` 规则做删除一井敏感性分析时，任意删去一井后仍优于对应的 final baseline；不过这不是独立 LOO 训练，不能替代扩大样本验证。

### 小树

只输入 `effective_seed_count` 和 `beam_cost`、预测 fallback 收益的回归树：

| 树设置 | hold LOO | beam LOO |
|---|---:|---:|
| depth 1, min leaf 3 | 16.697 | 16.209 |
| depth 2, min leaf 3 | 12.168 | 12.039 |
| depth 2, min leaf 5 | 13.697 | 13.519 |

树对容量非常敏感，证据不如预设 AND 规则清晰，不建议作为最终 selector。

## 建议

1. 在全 773 井上冻结并复验 `effective_seed_count + beam_cost` 规则，先看 pooled RMSE、p95/p99 和最差井，而不再调阈值。
2. 同时保留两条提交支线：保守版触发后用 hold，进取版触发后用 beam；只有独立回测仍稳定时才提交。
3. `final_departure` 和 posterior std 可作为软降权/告警量，但不要单独触发 hold/beam。
4. 不使用真实 TVT、井 ID、同井查表、LB 偏置或“逐井看误差后选模型”；提交时规则只能从当井可观测诊断计算。
