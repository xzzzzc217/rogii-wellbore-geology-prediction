# Guard640 v4 / v4b 的 model-package p95 独立审计

## 结论

`MODEL_PACKAGE_DIFF_P95_DISABLE=25.0` 对 Guard640 的 `0.49 / 192` 门控存在
实质性的过度跳过风险。它不是逐行削弱异常预测，而是当全体行的 p95 略高于 25 ft 时，
把整个 model-package 层对所有行关闭。现有可执行证据已经出现这种情况。

建议：

- v4b：**GO 进入普通可见数据 dry-run**；
- v4：若运行报告显示 `disabled_by_diff_guard=True`，则 **NO-GO 作为 6.40
  composition 候选提交**，因为它实际已经退化为无 model-package 的另一套模型；
- v4b：在 dry-run 的运行时、独立 submission audit 和最终层选择未通过前，
  **NO-GO 直接做隐藏提交**。

## 证据链

| 源/产物 | max weight | scale | p95 disable |
|---|---:|---:|---:|
| 公开 6.451 源 | 0.00425 | 6 | 25 |
| 公开 6.40 源 | 0.49 | 192 | 1e9 |
| 当前 v4 | 0.49 | 192 | 25 |
| 本地 v4b | 0.49 | 192 | 1e9 |

v4 把公开 6.40 的高权重、长尺度门控与 6.451 的 25-ft 全局开关组合在一起；这一组合
没有对应的公开执行证据。

完成的 6.451 精确复现在相同三口可见井上测得：

- mean absolute package/base disagreement：`15.0684998 ft`
- p95 absolute disagreement：`26.7006595 ft`
- max absolute disagreement：`29.3006617 ft`
- 结果：`26.701 > 25.000`，整层被关闭

在 6.451 的小门控下，即使保留该候选，p95 最终移动也只有 `0.012384 ft`，因此开关
是否触发几乎不改变轨迹。Guard640 不同：在同一个 `26.7006595 ft` 差值处，

```text
gate = 0.49 / (1 + (26.7006595 / 192)^2) = 0.4807035
move = gate * 26.7006595 = 12.8351008 ft
```

所以 25-ft 开关会因为 1.70 ft 的边缘越界，把一个约 12.84-ft 的主干修正突然变成
0。这说明它不是 Guard640 语义下的平滑灾难保护，而是高影响的二元架构开关。

隐藏环境的真实 p95 仍未知，因此不能声称必然超过 25；结论是“有直接证据支持的实质
风险”，不是对隐藏分布的确定断言。

## 为什么 v4b 不会令移动无界

v4/v4b 的逐行门控完全相同：

```text
move(d) = 0.49 * d / (1 + (d / 192)^2)
```

它在 `d=192 ft` 达到最大值：

```text
max move = 0.49 * 192 / 2 = 47.04 ft
```

差值继续增大时，最终移动反而下降。因此把全局阈值恢复为公开 6.40 的 `1e9` 不等于
允许无界外推。预算回退、有限值检查、动态 sample 合同、3-ft branch hedge cap 和
8-ft U-continuity cap 均未改变。

## 单一差异与静态审计

`kaggle_notebooks/pf_guard640_v4b` 由不可变 v4 notebook 构建。唯一评分行为差异是：

```python
MODEL_PACKAGE_DIFF_P95_DISABLE = 1.0e9  # v4 为 25.0
```

kernel ID、标题、审计文件名只为区分未来运行产物。专用 validator 会把这些身份字符串
和阈值反向归一化，再逐 cell 与 v4 比较；任何其他源码或 cell metadata 差异都会失败。

验证结果：

- 56 cells，42 个 code cells 全部编译；
- 禁用的危险路径为 0；
- same-ID/contact、exact-overlap train-TVT、overlap probe、global LB bias、固定 ID、
  固定行数、预测指纹和 canary 均被拒绝；
- 动态 sample 合同与预算保护保持不变；
- 34 个仓库测试通过；
- 新 builder/validator 的 Ruff lint 与 format 检查通过。

本地 notebook SHA-256：
`79D7115593A83D1BF73FB8E2495F0CC2CC5F6C443CEB4D773D6B6E82D4F4D188`。

本次审计没有 push、run 或 submit v4b。
