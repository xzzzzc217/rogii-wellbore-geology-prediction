# U-continuity 全 773 井冻结复验

## 结论

**NO-GO：不把 `cap=8.0, tau=240.0` 的 U-continuity 修正加入当前 final，
也不根据本次结果调整 cap、tau、模型来源或 gate。**

该修正的 pooled RMSE 从 `12.518489` 降至 `12.518109`，改善只有
`-0.000380`，未达到结果可见前冻结的部署门槛 `<= -0.020`。其余稳定性条件均
通过，但部署 gate 要求所有条件同时成立，因此最终判定为 NO-GO。

## 冻结部署 gate

下列条件在全量结果生成前冻结：

1. pooled RMSE delta `<= -0.020`；
2. well-macro delta `< 0`；
3. 20,000 次 paired-well bootstrap 的 pooled delta 95% CI 上界 `< 0`；
4. 5 个固定空间 cluster 中至少 4 个 pooled delta `<= 0`，且没有 cluster
   劣于 `+0.020`。

| 条件 | 观测结果 | 判定 |
|---|---:|---|
| pooled delta `<= -0.020` | **-0.000380** | **FAIL** |
| macro delta `< 0` | -0.000945 | PASS |
| paired bootstrap CI95 上界 `< 0` | -0.000156 | PASS |
| 至少 4/5 空间 cluster 非正 | 5/5 | PASS |
| 最差空间 cluster `<= +0.020` | -0.000249 | PASS |

最终 gate：**NO-GO**。失败原因仅是实际收益幅度远小于最低部署要求；不能因为其余
检查通过而放宽事前门槛。

## 全量指标

| 方法 | pooled | macro | p90 | p95 | p99 | worst |
|---|---:|---:|---:|---:|---:|---:|
| final | 12.518489 | 9.104812 | 19.575424 | 27.272138 | 40.961761 | 69.795973 |
| final + U-continuity | 12.518109 | 9.103866 | 19.574091 | 27.271437 | 40.960985 | 69.795864 |
| delta | -0.000380 | -0.000945 | -0.001333 | -0.000701 | -0.000776 | -0.000109 |

逐井比较为 424 胜、349 负、0 平。最坏井在修正前后都是 `173ed841`。

## Paired-well bootstrap

bootstrap 在结果可见前固定为：

- 将井作为配对单位有放回抽样；
- 每次用抽中的逐井行数和 RMSE 重算两侧 pooled RMSE；
- 20,000 次，随机种子 `20260724`；
- percentile 95% CI。

结果：

| 统计量 | pooled delta |
|---|---:|
| mean | -0.000380 |
| median | -0.000379 |
| CI95 lower | -0.000610 |
| CI95 upper | -0.000156 |

区间整体小于 0，说明极小改善在 paired-well 重采样下方向稳定；但它不能弥补改善
幅度比 `0.020` 门槛小约 53 倍的问题。

## 固定空间 cluster holdout

每井取完整水平轨迹的 median X/Y，经过 `StandardScaler` 后执行固定
`KMeans(n_clusters=5, random_state=20260724, n_init=50)`。cluster 标签只用于
分区，不表示地质顺序。

| cluster | wells | rows | pooled delta | wins | losses |
|---:|---:|---:|---:|---:|---:|
| 0 | 79 | 340,190 | -0.000249 | 42 | 37 |
| 1 | 164 | 790,418 | -0.000253 | 99 | 65 |
| 2 | 215 | 1,030,819 | -0.000333 | 113 | 102 |
| 3 | 107 | 540,934 | -0.000551 | 62 | 45 |
| 4 | 208 | 1,081,628 | -0.000508 | 108 | 100 |

5/5 cluster 均为非正，最差 delta 为 `-0.000249`，空间稳定性条件通过。

## Boundary gap 与移动诊断

公开 Guard640 v4 的冻结公式被逐行转录为：

```text
last_visible_U = TVT_input[last_visible] + Z[last_visible]
gap_U = prediction[first_hidden] + Z[first_hidden] - last_visible_U
move = -clip(gap_U, -8, 8)
       * exp(-(MD[hidden] - MD[last_visible]) / 240)
refined[hidden] = prediction[hidden] + move
```

仅修改官方 `TVT_input` 连续缺失后缀；transform 接口不接受真实 TVT。

| 诊断 | mean abs | median abs | p90 abs | p95 abs | p99 abs | max abs |
|---|---:|---:|---:|---:|---:|---:|
| boundary gap U before | 0.283833 | 0.213158 | 0.628438 | 0.748215 | 0.931209 | 1.349684 |
| boundary gap U after | 0.001180 | 0.000886 | 0.002613 | 0.003111 | 0.003872 | 0.005612 |
| first move | 0.282653 | 0.212272 | 0.625825 | 0.745104 | 0.927337 | 1.344072 |

没有任何井达到 8 ft cap：cap-saturated 子组为 0 井、0 行。逐井
`mean_abs_move` 的平均值为 `0.014753` ft；按 3,783,989 个打分行加权后的
mean absolute move 为 `0.013884` ft。实际修正非常小，解释了指标方向稳定但收益
幅度不足。

## 验证合同与运行

- 数据：全部 773 口训练井。
- 掩码：官方 `TVT_input` 缺失后缀。
- 打分行：3,783,989。
- PF seeds：固定 `0..7`。
- U-continuity：固定 `cap=8.0, tau=240.0`。
- 真实 `TVT` 与模型和 correction 完全隔离，仅在两者返回后用于打分。
- 不使用 same-ID、formation、训练目标特征、LB bias 或公开预计算预测。
- 成功 773 井，失败 0 井；stderr 为空。
- 主 evaluator 运行时：5,265.4 秒（87.8 分钟）。

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_u_continuity.py `
  --max-wells 0 `
  --fail-fast `
  --json-output artifacts\target_free_eval_773_ucontinuity.json `
  --progress-every 10

.\.venv\Scripts\python.exe scripts\analyze_u_continuity.py `
  artifacts\target_free_eval_773_ucontinuity.json
```

## 测试与复核

- 公式、仅 hidden 后缀、首点、指数衰减、正负 cap、输入不变和无真值接口均有
  单元测试。
- paired bootstrap 的确定性和 frozen gate 的全条件语义有单元测试。
- 新增测试共 7 项通过，Ruff 检查通过。
- evaluator 的 baseline final 与第一轮冻结 JSON 精确一致：
  pooled `12.518488798577081`。
- JSON 中恰有 773 个唯一井，失败列表为空，PF seeds 精确为 `0..7`。
- 结果 JSON SHA-256：
  `c62b2b718a395ca7d99b9977f054b13bb696ba168bb249ba854148c05d83732e`。
