# ROGII public candidate scout

核验时间：2026-07-24（Asia/Shanghai）。指标为 RMSE（ft，越低越好）。

## 结论

**没有找到值得原样提交、同时具有完整可运行源码和隐藏井泛化证据的公开候选。**

- 最接近“可泛化源码”的是 [AmgedAlfaqih / High-Scoring Reproducible](https://www.kaggle.com/code/AmgedAlfaqih/high-scoring-reproducible)，页面实际分数 `6.435`。它只把 exact 基线的 balanced visible-prefix 参数从 `cap=0.36, clip_max=28` 改成 `cap=0.40, clip_max=30`；但该层在三个公开井上 `applied_wells=0`，所以 `6.435` 相对题称 `6.451` 的 `0.016` 改善无法归因于这处改动，且小于公开讨论中观测到的重复运行波动。
- 页面上更低的 `6.390/6.417/6.433/6.440` 均伴随固定公开井、固定行数、预测哈希/指纹或无法恢复最佳历史版本源码的问题。
- `fleongg/rogii-codex-public-6-40-guard` **不是 6.40 候选**：页面唯一版本实际 Public Score/Best Score 均为 `11.719`。标题不能作为分数证据。

因此本次 scout 的单一行动建议是：**不原样提交任何新增公开 notebook**。若必须继续研究，只应隔离做 target-free 的层级验证；不能把以下任何公开整本 notebook 当作 hidden 候选。

## 候选证据

| Notebook | 页面实际分数 | 日期与运行时间 | 当前源码及相对 exact `6.451` 源的差异 | 风险与结论 |
|---|---:|---|---|---|
| [fle3n — ROGII Codex Public 6.40 Guard](https://www.kaggle.com/code/fleongg/rogii-codex-public-6-40-guard?scriptVersionId=336942056) | Current/Best `11.719` V1 | 2026-07-21 13:48:35 UTC；`15m19s`，GPU T4 x2 | V1 的 56 个 code cells 和 56 个输出文件均公开可取。相对 exact 源：model-package 从 `0.00425/6` 改成 `0.49/192`，并把 p95 熔断从 `25` 放宽到 `1e9`；去掉 PF `gs*1.3`；VP alpha 乘数从 `1.30` 降为 `0.35`；branch hedge 从 strength/cap `0.60/2` 改为 `1.00/3`；新增 U-continuity 与 public exact-overlap 写入层。 | **严格 NO-GO，不能 exact 原样提交。** 页面分数已直接否定标题。输出显示 model-package 的 base 差异 p95 `26.7007 ft`，被选中的最终移动 mean/p95/max 为 `7.3151/12.8351/14.0306 ft`。三个公开井全部触发 same-ID contact override，合计覆盖全部 `14,151` 行；随后又全部触发 exact-geometry train-TVT blend（权重 `0.055`）。代码没有把三个井 ID 或 `14,151` 写成激活常量，但它依赖“公开 test 是同 ID、同几何 train 副本”的固定公开合同；隐藏新井会走完全不同的 fallback，缺乏泛化证据。 |
| [Evan — ROGII Public Score Frontier Lab \| VISUALS](https://www.kaggle.com/code/evansussex/rogii-public-score-frontier-lab-visuals) | Current `6.465` V2；Best `6.390` V1 | V2：2026-07-23 07:48:06 UTC，`12m53s`；V1 `15m13s` | 当前 V2 完整源码可取；exact 基线上增加 Q0522 score writer。V1 页面仍明确标为 “GS1.30 + Q0522”，但 V1 历史源码未由 CLI 独立恢复。 | **严格 NO-GO。** 当前 score writer 固定 base SHA、`rows=14151`、井 `00e12e8b`、井行数 `4301`，并对该井加公开榜调出的 `+0.522 ft`。`6.390` 不是隐藏泛化证据。 |
| [KAIWALYA RAUT — ROGII Solution](https://www.kaggle.com/code/kaiwalyaatulraut/rogii-solution) | Current `6.499`；Best `6.417` V3 | 当前 V4：2026-07-23 15:31:35 UTC，`16m20s`；V3 `13m06s` | 当前完整源码可取，48 个 code cells；含与 Q0522 相同的固定 base SHA/行数/井/井行数路径。最佳 V3 的历史源码未独立恢复。 | **NO-GO。** 无法把历史 `6.417` 与一份完整、无固定公开合同的源码绑定；当前源码本身不合格。 |
| [T.SHIGA — ROGII_v34](https://www.kaggle.com/code/takumashiga/rogii-v34) | Current/Best `6.433` V1 | 2026-07-23 21:16:38 UTC；`14m41s`，GPU T4 x2 | 完整源码可取。PF particles `500→1000`、selector seeds `128→256`、VP seeds `24/48→48/96`；另加 A28，固定公开 `rows=14151` 与 w0.60 输出指纹，强制 `0.62 SP45 / 0.38 learned`。 | **NO-GO。** 分数只比题称 `6.451` 低 `0.018`，仍在重复运行噪声带内；A28 又依赖固定公开指纹/行数，不能作为 hidden 证据。 |
| [Amged Alfaqih — High-Scoring Reproducible](https://www.kaggle.com/code/AmgedAlfaqih/high-scoring-reproducible) | Current/Best `6.435` V11 | 2026-07-23 00:04:56 UTC；`12m44s`，GPU P100 | 完整源码可取，45 个 code cells；仅 1 个 cell 不同：balanced VP `cap 0.36→0.40`、`clip_max 28→30`。其余 scoring path 与 exact 源一致，并保留未固定 Numba PF 随机性。 | **最接近可研究，但仍不提交。** 公开输出显示 changed layer 对三个井均未应用；`0.016` 差异无法归因，且该 exact 系列本身仍含同井/contact 公开展示路径。 |
| [Yusuke Togashi — ROGII Another approch 2nd](https://www.kaggle.com/code/yusuketogashi/rogii-another-approch-2nd) | Current/Best `6.440` V12 | 2026-07-23 00:04:09 UTC；`13m48s`，GPU T4 x2 | 完整源码可取，47 个 code cells；exact 基线上加 A31 mean-preserving toe tilt。 | **严格 NO-GO。** A31 固定 `rows=14151`、source/prediction SHA、井 `00e12e8b`、井行数 `4301` 和 branch shift `2.0`。页面改善仅 `0.011`。 |
| [Arnav Salkade — ROGII Public 6.451 Base](https://www.kaggle.com/code/arnavsalkade/rogii-public-6-451-base?scriptVersionId=337339786) | Current/Best `6.504` V1 | 2026-07-23 06:00:27 UTC；`13m46s`，GPU T4 x2 | 页面 scoring code 与本地 exact 源逐 code-cell 一致；标题中的 `6.451` 不是当前页面分数。 | 这是比较基准，不是 `6.451` 的权威复现结果。页面证据是 `6.504`，再次说明不能按标题筛候选。 |
| [Leonid Zaporozhets — New Strategy \[Score: 6.213\]](https://www.kaggle.com/code/leonidzaporozhets/new-strategy-score-6-213) | Current/Best `6.516` V1 | 2026-07-22 22:14:00 UTC；notebook duration 约 `12m56s` | 完整源码可取；45 个 code cells 与 exact 源逐字节一致。 | **拒绝。** 标题 `6.213` 无页面依据，且没有任何新模型。 |

exact 比较源：`kaggle_notebooks/pf_public_reproduction/rogii-pf-public-reproduction.ipynb`（45 个 code cells）。

## `6.40 Guard` 的固定合同与泛化审计

页面权威信息：

- Public Score：`11.719`
- Best Score：`11.719 V1`
- Runtime：`15m19s · GPU T4 x2`
- 页面版本：`1/1`
- 公开运行日期：2026-07-21 13:48:35 UTC

当前运行产物证明：

- `gold_contact_override_report.csv` 与 `guarded_overlap_override_report.csv`：`000d7d20`、`00bbac68`、`00e12e8b` 均为 `override`，覆盖 `3836 + 6014 + 4301 = 14151` 行。
- `public_overlap_calibration.json`：`activated_wells=3/3`；它在 exact MD/X/Y/Z/GR 几何匹配后直接读取同 ID train 的 `TVT`，以 `5.5%` 混入最终预测。
- `model_package_correction_report.csv`：`0.49/192` 分支被选中，所有行发生大幅移动；p95 熔断被 `1e9` 实质关闭。
- `hidden_submission_contract_audit.json` 只验证动态 sample 的列、ID 集、顺序、有限值与换行格式。它不是泛化验证；其中记录的 `source_notebook` 还是另一个 notebook（`beicicc/rogii-mha160-sep3-r2-20260720`），`source_public_score=null`。

这不是“写死三个井 ID”式的硬编码，但风险同样明确：**模型的高影响路径按公开 placeholder 的 same-ID/exact-geometry 特征激活，隐藏新井则不激活。** 因而：

1. 标题 `6.40` 与页面实际 `11.719` 冲突；
2. 当前源码的公开表现本身很差；
3. 公布的三井输出不能证明 hidden 泛化；
4. 高权重 model-package 层还有 12–14 ft 级移动风险。

结论：`fleongg/rogii-codex-public-6-40-guard` **不值得 exact 原样提交，也不应作为 v4/v4b 的分数上游证据**。其中 target-free 的 U-continuity 可以被隔离验证，但不能由这份混合栈的标题或当前分数背书。

## 公共讨论对判定阈值的影响

- [The public LB is a precise ruler and a biased one](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728477)：作者报告 byte-identical notebook 重提可相差约 `0.5 ft`，重设随机种子可移动 `0.2–0.3 ft`，严格控种流水线仍有 `0.028 ft`。因此 `6.433–6.440` 相对题称 `6.451` 的 `0.011–0.018` 不足以证明真实改进。
- [How are you validating locally?](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727570)：公开回复指出 `test/` 的三口展示井是 train 井副本，真正 rerun 会换成 hidden wells；对展示井按同 ID 读 train 真值等价于在训练数据上打分。建议使用 by-well / by-field CV。
- [Observation: a small `gs` tweak](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728712)：公开提到的 `gs*1.3` 已包含在 exact 比较源；它不是本次新候选。`6.40 Guard` 反而删除了该乘数。

## 范围与操作记录

- 以 Kaggle competition notebook 的 `scoreAscending` 公共排序为入口，逐页核对页面 Public Score/Best Score；不采用标题里的分数。
- 只下载公开 source/output 做静态比较；没有 push、run、submit 或修改任何模型源码。
- 历史最佳版本若无法恢复完整源码，按“证据不足”处理，不猜测其实现。
