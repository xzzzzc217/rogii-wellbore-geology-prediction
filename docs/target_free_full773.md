# Target-free 全 773 井冻结验证

## 结论

**NO-GO：不部署该 disaster fallback 规则，也不在这 773 井结果上重新调整阈值。**

预先冻结的规则为：

```text
effective_seed_count <= 1.5
AND beam_average_cost <= 2.0
```

其中 `effective_seed_count = 1 / sum(seed_weight ** 2)`。规则触发时分别测试切换到
`hold_last` 和 `beam_viterbi`，未触发时保持 `final`。

在全部 773 口训练井的官方 `TVT_input` 缺失后缀上，两个固定 selector 的 pooled
RMSE 都差于 `final`：

- 切换到 hold：`12.6708`，比 final 恶化 `+0.1523`。
- 切换到 beam：`12.7573`，比 final 恶化 `+0.2389`。

虽然 hold selector 降低了 p90/p95/p99，beam selector 降低了 p95/p99，但两者都
提高了 macro RMSE 和最坏井误差；这不是可以接受的稳定改进。规则在 50 井探索集
上的收益没有外推到全量冻结验证，因此不能上线。

## 验证合同

- 数据：`data/competition/train` 下全部 773 口井。
- 掩码：每口井官方 `TVT_input` 缺失后缀。
- 打分行：3,783,989。
- PF seeds：固定为 `0..7`，共 8 个。
- 模型输入：水平井 `MD/X/Y/Z/GR/TVT_input`，typewell `TVT/GR`。
- 不使用同井 ID 查表、formation、训练目标特征、榜单偏置或公开预计算预测。
- 运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_target_free.py `
  --max-wells 0 `
  --pf-seeds 8 `
  --json-output artifacts/target_free_eval_773.json `
  --progress-every 10
```

单组进程运行时为 3,684.3 秒（61.4 分钟）。成功 773 井，失败 0 井；stderr 为空。

## 固定规则结果

下表的 oracle 按逐井真实 RMSE 选择，仅用于表示 selector 上限，不能用于推理或
提交。

| 方法 | pooled | macro | p90 | p95 | p99 | worst |
|---|---:|---:|---:|---:|---:|---:|
| final | 12.5185 | 9.1048 | 19.5754 | 27.2721 | 40.9618 | 69.7960 |
| 冻结规则 → hold | 12.6708 | 9.5019 | 19.4383 | 24.9487 | 37.6826 | 70.6394 |
| 冻结规则 → beam | 12.7573 | 9.5510 | 19.9696 | 25.8082 | 37.6826 | 75.2448 |
| 逐井 oracle(final, hold) | 9.2362 | 7.1538 | 13.6361 | 17.5173 | 29.6000 | 46.6231 |
| 逐井 oracle(final, beam) | 9.3056 | 7.1932 | 13.4737 | 17.9070 | 29.6000 | 46.6231 |
| 逐井 oracle(final, hold, PF, beam) | 8.8904 | 6.8028 | 12.8100 | 17.2149 | 28.9706 | 44.8276 |

相对 final：

| 固定 selector | Δ pooled | Δ macro | Δ p90 | Δ p95 | Δ p99 | Δ worst |
|---|---:|---:|---:|---:|---:|---:|
| → hold | +0.1523 | +0.3971 | -0.1371 | -2.3235 | -3.2792 | +0.8434 |
| → beam | +0.2389 | +0.4462 | +0.3942 | -1.4639 | -3.2792 | +5.4488 |

规则触发 179/773 井（23.16%），覆盖 941,478 个打分行：

- hold 在触发井中 72 胜、107 负。
- beam 在触发井中 67 胜、112 负。
- 未触发的 594 井保持 final。

最坏井也发生了改变：final 的最坏井为 `173ed841`（69.7960）；hold 和 beam
selector 的最坏井均为 `1b1eba53`，RMSE 分别为 70.6394 和 75.2448。

## 复核

已完成以下一致性检查：

- JSON 中恰有 773 个唯一井 ID，且 `successful_wells=773`、`failed_wells=0`。
- 配置中的 PF seeds 精确为 `[0,1,2,3,4,5,6,7]`。
- 每井诊断均声明 `target_free=true`，且未使用训练目标、同井查表、formation 或
  leaderboard bias。
- 从逐井 RMSE 和行数重新计算的 pooled 指标与 evaluator 聚合指标一致。
- 产物 SHA-256：
  `f6d2bed741366c6338dce197fc3e9a211e9efd454c9bdecf2dbfb3b188381ff1`。

## 后续限制

当前 evaluator 只在内存中保留逐行预测，JSON 仅保存逐井 RMSE 和诊断量。因此，
本次产物不能直接追加逐行 `U=TVT+Z` boundary continuity 后处理；若要验证
`cap=8, tau=240` 的固定规则，需要单独重跑模型，或先扩展 evaluator 以在线计算
该指标/持久化逐行预测。本轮未启动第二次验证。
