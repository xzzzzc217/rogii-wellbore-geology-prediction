# ROGII 下一批公共候选审计

> **2026-07-24 page-audit correction:** this earlier prioritization used title
> numbers as provisional evidence. The authoritative Kaggle page score for
> `rogii-codex-public-6-40-guard` is 11.719, not 6.40; it is now a strict
> NO-GO and provides no score support for v4/v4b. The page score for
> `rogii-public-6-451-base` is 6.504, not 6.451. See
> `docs/rogii_public_candidate_scout.md` for the corrected audit.

审计日期：2026-07-24（Asia/Shanghai）

## 结论

下一次提交应优先做一个 **6.40 Guard 的去泄漏、限时版本**。第二选择是同一主干更保守的
**6.451 Base 去泄漏版本**。两者的公开标题都已经低于当前目标线 6.65，但标题分数不能原样
迁移到我们的修改版；能否达到 6.65 仍必须由隐藏重跑决定。

| 顺序 | 候选 | 公开信号 | 建议定位 |
|---:|---|---:|---|
| 1 | `rogii-codex-public-6-40-guard` | 标题 `6.40` | v3 主候选：保留 0.49/192 组合、弱化 VP 和 U-continuity，删除所有同井分支并降配到 8 小时内 |
| 2 | `rogii-public-6-451-base` | 标题 `6.451` | v3b 保守候选：保留原始 0.00425/6.0 小门控和通用 PF branch hedge，删除同井分支并降配 |

`rogii-public-6-40-guard-repro` 与 6.40 Guard 的 notebook SHA-256 完全相同，不构成独立复现
证据。`ultimate-pf-config-strategy-a-reproducible-score` 与 6.491 notebook 也完全相同。

## 1. 首选：6.40 Guard 的 target-free hidden 版本

### 为什么排第一

`fleongg/rogii-codex-public-6-40-guard` 的标题声称公开分 `6.40`，已经明显越过 6.65
目标线。它在公共 PF 主干上加入了三项对隐藏井也可用的改动：

1. balanced visible-prefix 的最终移动缩放为 `alpha × 0.35`；
2. model-package disagreement gate 改为 `max_weight=0.49, scale=192`；
3. 在 `U=TVT+Z` 边界上做 `cap=8 ft, tau=240 MD` 的指数衰减连续性修正。

源码注释声称这些参数在 300 口校准井、300 口不相交 holdout 井，以及覆盖 773 井的三组
pseudo-hidden partition 上审计过。仓库中没有相应 OOF 产物，因此这只能视为作者声明，不是
我们已复核的 CV 证据。它仍比纯标题分更有建模依据，尤其 U-continuity 完全由测试可见输入
决定。

### 标题分与当前源码是否对应

不能严格证明完全对应：

- notebook 顶层 papermill 记录是 2026-07-07、约 776 秒的 3 井可见数据运行；
- 当前关键 cell——0.49 model-package 配置后的 PF branch hedge、U-continuity、public overlap
  和 hidden contract——没有嵌入 execution duration；
- hidden audit 源码中的 `source_public_score` 仍是 `None`。

标题和说明暗示作者是在 notebook-origin code submission 完成后才写入 `6.40`，所以这是强
公开信号；但当前下载文件的嵌入输出不能独立证明“当前每一行源码 = 6.40 预测向量”。

`pi500phys/rogii-public-6-40-guard-repro` 与其字节级相同
（SHA-256 `618DDAB5...B755FFE`），只是镜像，不是另一次运行结果。

### 不能原样提交的原因

当前源码包含三层必须删除或默认关闭的同井逻辑：

- profile 中 `RUN_GUARDED_OVERLAP_OVERRIDE=True`；
- selector/特征构造中可见测试井会从同 ID 训练井取得 formation/TVT 路径，测试特征的
  `self_wid` 还是 `None`；
- 最后 `Public-test overlap calibration` 对几何完全相同的 train/test 井混入 5.5% 训练
  `TVT`。

这些分支对真正未见过的隐藏 ID 通常会自动跳过，因此未必解释隐藏公榜 6.40；但它们污染
可见 3 井、违反本项目的无同井策略，也让回测语义不一致，必须禁用。

没有发现 Kaggle API 自动 probe、静态隐藏 ID、固定 14,151 行合同或全局 TVT bias。最终
hidden contract 以运行时 `sample_submission.csv` 为准，能适配隐藏行数。`RUN_OVERLAP_DRY_RUN_PROBE`
只是本地重叠统计，不是 LB probe，但也应关闭。

### 外部资产

默认有效路径需要：

- `phongnguyn23021656/koolbox-offline`
- `ravaghi/wellbore-geology-prediction-artifacts`（约 7.39 GB 特征与 trainer pickle）
- `fleongg/rogii-claude-models-pub`
- `pilkwang/rogii-model-package`

metadata 中的 `nina2025/rogii-03`、`thbdh5765/rogii-v10-fresh-artifacts` 和
`needless090/rogii-tabicl-mirror` 未被默认有效路径需要，正式候选应移除。

### 明确配置建议

从当前 safe-v2 派生，不直接编辑公共原件。保留 6.40 的组合逻辑，但使用以下安全控制：

```python
SUBMISSION_PROFILE = "v3_guard_640_safe"

ALLOW_SAME_ID_TRAIN_LOOKUP = False
RUN_EXACT_MATCH_RECOVERY = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_PUBLIC_OVERLAP_CALIBRATION = False  # 新总开关，包住 cell 61

# 两套 test feature builder 都必须排除 query well
swid = wid

RUN_VISIBLE_PREFIX_CALIBRATION = True
VISIBLE_PREFIX_PROFILE = "balanced"
VISIBLE_PREFIX_FINAL_SELECTION = "profile"
VISIBLE_PREFIX_CAL_SEEDS = 12
VISIBLE_PREFIX_FINAL_SEEDS = 24
VISIBLE_PREFIX_PARTICLES = 250
VISIBLE_PREFIX_CUT_FRACS = (0.50, 0.65, 0.75)
VISIBLE_PREFIX_ALPHA_MULTIPLIER = 0.35

RUN_MODEL_PACKAGE_CORRECTION = True
MODEL_PACKAGE_REQUIRE = True
MODEL_PACKAGE_ALLOW_AUTO_SEARCH = False
MODEL_PACKAGE_GATED_MAX_WEIGHT = 0.49
MODEL_PACKAGE_GATED_SCALE = 192.0
MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0  # 不沿用源码的 1e9

SP45_SELECTOR_N_SEEDS = 96
SELECTOR_PF_SEEDS = 96
CFG.PF_SEEDS = 96

RUN_PF_BRANCH_HEDGE = True
PF_BRANCH_HEDGE_CAP = 3.0
RUN_U_CONTINUITY = True
U_CONTINUITY_CAP = 8.0
U_CONTINUITY_TAU = 240.0

RUN_CV_REPORT = False
RUN_FULL_STACK_CV_ABLATION = False
GLOBAL_TVT_BIAS = 0.0
```

其中 `MODEL_PACKAGE_DIFF_P95_DISABLE=25` 是必要的灾难保护；公共 6.40 源码把它设成 `1e9`，
等于关闭保护。最终还应要求：

- `fallback_rows == 0`
- sample ID 完整、顺序一致、TVT 全部有限
- 记录每阶段耗时和预计完成时间
- 预计总耗时超过 7.5 小时时，先跳过 model-package 或把 PF 从 96 降至 64，始终保留最近
  一份已审计 submission

### 运行时

公共主干在 3 个可见井的历史运行约 12.9–13.3 分钟。可分解的主要时间是：7.39 GB
artifact/特征约 281 秒、selector 约 184 秒、learned branch 约 66 秒、visible-prefix
约 119 秒、model-package 约 43 秒。

把当前 128/24/48/350 原样外推到约 200 井约为 7.5–10 小时，不能满足本项目 `<8h` 的硬
预算。上述 96-seed 主 PF 与 12/24/250 visible-prefix 配置预计约 5.5–7.2 小时；这是工程
估算，必须通过完整 Kaggle dry-run 校准。若 dry-run 超过 7.5 小时，此候选不准提交。

## 2. 次选：6.451 Base 的保守恢复版

### 为什么排第二

`arnavsalkade/rogii-public-6-451-base` 只比
`helloromiro/pf-config-strategy-high-scoring-6-491` 多一个 markdown cell，预测代码完全
相同。它的嵌入运行记录更新到 2026-07-22，约 800 秒，并且 model-package cell 实际运行
约 42.8 秒、PF branch hedge 实际运行约 0.14 秒；因此它比旧 6.491 下载版本更能证明当前
默认 `vp_balanced_modelpkg_005` 路径至少能完整执行。

标题 `6.451` 低于目标线，而且比旧标题 `6.491` 更强。不过当前可见运行仍有三口同 ID 井
被 guarded contact 路径覆盖，所以本地输出不能作为隐藏泛化证据。

### 标题分与 profile

当前 control cell 是：

```text
SUBMISSION_PROFILE = vp_balanced_modelpkg_005
SP45_BLEND_WEIGHT = 0.60
VISIBLE_PREFIX_PROFILE = balanced
MODEL_PACKAGE_GATED_MAX_WEIGHT = 0.00425
MODEL_PACKAGE_GATED_SCALE = 6.0
```

与旧 6.491 文件不同，6.451 Base 的嵌入执行确实跑过 model-package 和末端 branch hedge，
所以“当前 profile 已执行”的证据较强。仍没有提交 ID、隐藏运行日志或预测 SHA 能把标题
6.451 与该精确向量一一绑定，因此不能把 6.451 当作修改版的承诺分。

### 安全配置

使用与候选 1 相同的所有去同井和限时开关，但恢复更保守的公共参数：

```python
SUBMISSION_PROFILE = "v3_base_6451_safe"

ALLOW_SAME_ID_TRAIN_LOOKUP = False
RUN_EXACT_MATCH_RECOVERY = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_PUBLIC_OVERLAP_CALIBRATION = False
swid = wid

RUN_VISIBLE_PREFIX_CALIBRATION = True
VISIBLE_PREFIX_PROFILE = "balanced"
VISIBLE_PREFIX_FINAL_SELECTION = "profile"
VISIBLE_PREFIX_CAL_SEEDS = 12
VISIBLE_PREFIX_FINAL_SEEDS = 24
VISIBLE_PREFIX_PARTICLES = 250

RUN_MODEL_PACKAGE_CORRECTION = True
MODEL_PACKAGE_GATED_MAX_WEIGHT = 0.00425
MODEL_PACKAGE_GATED_SCALE = 6.0
MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0

SP45_SELECTOR_N_SEEDS = 96
SELECTOR_PF_SEEDS = 96
CFG.PF_SEEDS = 96

RUN_PF_BRANCH_HEDGE = True
PF_BRANCH_HEDGE_CAP = 2.0
RUN_U_CONTINUITY = False
GLOBAL_TVT_BIAS = 0.0
```

外部资产与候选 1 相同。运行时预计约 5.0–6.8 小时。它与候选 1 共用大量主干，因此只能
作为参数风险对冲，不是独立模型。

## 其他下载 notebook 的处理结论

| Notebook | 公开/标题信号 | 当前源码问题 | 结论 |
|---|---:|---|---|
| `pf-config-strategy-high-scoring-6-491` | `6.491` | 旧嵌入输出没有执行当前 model-package 和末端 hedge；同井路径默认开启 | 已被更新的 6.451 Base 覆盖 |
| `ultimate-pf-config-strategy-a-reproducible-score` | 无数字 | 与 6.491 文件 SHA 完全相同 | 非独立候选 |
| `rogii-public-6-40-guard-repro` | `6.40` | 与 6.40 Guard 文件 SHA 完全相同 | 只能作镜像 |
| `rogii-another-approch-2nd` | 上游称 `6.451`；作者当前队伍约 `6.486` | A31 固定 14,151 行、井 `00e12e8b`、4,301 行和预测 SHA；隐藏运行会直接 abort | 不可原样提交；队伍分无法归因于该 exact source |
| `rogii-another-approach` | 源 6.858/6.979 一带实验 | 含从 LB canary 解出的全局 `-0.40 ft`、单井定向 probe 和静态公开井逻辑 | 排除 |
| `rogii-dual-track-prefix-calibrated-geosteering` | 无可绑定数字 | `GLOBAL_TVT_BIAS=-0.50`，同井/VP/modelpkg 全开；下载文件无 cell 执行证据 | 只有去 bias 后才可研究，暂不占提交位 |
| `rogii-lb7295-public-rebuild` | `7.295` | aggressive VP、同井覆盖，且无可靠运行时元数据 | 标题已弱于当前目标，不提交 |
| `rogii-target-free-tvt-geosteering` | 无标题分 | 当前 profile 的关键 cell 缺执行记录；历史 108 分钟来自另一条 super-stack；测试特征 `self_wid=None` | 可作独立第三路线，但不是下一次提交 |

## LB bias、probe 与 canary 红线

所有正式候选必须满足：

```text
GLOBAL_TVT_BIAS = 0
RUN_EXACT_MATCH_RECOVERY = False
RUN_GUARDED_OVERLAP_OVERRIDE = False
RUN_OVERLAP_DRY_RUN_PROBE = False
RUN_PUBLIC_OVERLAP_CALIBRATION = False
ROGII_PROBE unset / "0"
no fixed test well IDs
no fixed sample row count
no expected public prediction SHA
no train-TVT lookup for a test well
```

0.49/192、0.35、cap 8/tau 240 也可能有公共选择偏差，但它们不是按隐藏 ID 查表或统一偏置，
而且有作者声称的分井回测依据，因此可以作为预注册候选保留。每次提交只允许比较这两个预先
登记的配置，不再用 leaderboard 逐常数爬坡。

## 推荐执行次序

1. 先构建候选 1 的新私有 notebook，但不改动任何下载的公共 notebook。
2. 静态检查所有 same-ID、固定行数、固定 SHA、probe 和 bias 均已失效。
3. 完整 Kaggle dry-run；若预计或实际超过 7.5 小时，按“跳过 model-package，再降 PF 64”
   的顺序回退。
4. 只有 dry-run 的 schema、finite、fallback、运行时和阶段报告全部通过，才提交候选 1。
5. 候选 2 只在候选 1 完成后运行，作为较小 model-package gate、无 continuity 的风险对冲。

这两个候选最有希望把 7.514 拉到 6.65 附近或以下，但公开分数均来自未完全可验证的公共
版本；修改后的隐藏成绩仍有显著不确定性，不能保证。
