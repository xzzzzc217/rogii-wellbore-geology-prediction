# ROGII 井眼地质预测：Kaggle 奖牌冲刺

这是 [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
竞赛的完整研究与复现仓库。项目目标是在很短的赛程内建立一个可审计、可复现、能在
Kaggle 无网络 Notebook 环境中稳定运行的地质轨迹预测系统，并尽可能取得奖牌。

> 仓库设为 **Private**。竞赛原始数据、模型权重、提交文件、排行榜快照、Kaggle
> 凭据和运行产物不会提交到 Git。

## 1. 当前状态

截至 2026-08-03：

| 项目 | 状态 |
|---|---:|
| 已加入竞赛并接受规则 | 完成 |
| 训练井 / 可见测试井 | 773 / 3 |
| 当前最佳 Public RMSE | **6.477** |
| 当前公开榜排名 | **1,164 / 6,060，前 19.21%** |
| 公开榜前 10% 边界 | 6.418 |
| 公开榜前 5% 边界 | 6.373 |
| 当前奖牌判断 | 暂处公开榜铜牌区外，最终奖牌未确定 |
| Kaggle 最终入选状态 | **2 / 2；`54952634` + `54967104` 已选并经 API 复核** |
| A28 PF-1.3 w0.62 高粒子版 | 隐藏重跑超时，无分数 |
| A28 runtime-fast exact-core | **7.678**，较最佳差 1.201，NO-GO |
| A28/Q0522 50/50 融合 | 隐藏重跑超时，无分数 |
| Frontier II DYNQ0196 | **7.507**，较最佳差 1.030，NO-GO |
| GPU TCN + target-free well gate | gate 仅改善 0.0062，门槛 FAIL |
| GPU TCN uniform-well sampling | pooled 恶化 0.0496，门槛 FAIL |
| Target-free dip-rate HMM | scale1 ref `55180816`：**6.573**，较最佳差 0.096，NO-GO |
| Dip-rate HMM scale2 排名冲刺 | ref `55181026`：**6.664**，较最佳差 0.187、较 scale1 差 0.091，NO-GO |
| Dip-rate HMM scale2.3 条件候选 | scale2 Public 门槛失败；永久关闭，不推送、不提交 |
| HMM scale2 + Fixed Ridge–PLS | Ridge–PLS Public 为 **6.482**，未严格优于 6.477；条件组合永久关闭 |
| GR strongest-periodic notch | pooled 恶化 4.2925，门槛 FAIL |
| Harshini 空间曲面 + PF | 严格空间核心 17.944，差于 flat 14.969，NO-GO |
| MHA180 target-free midpoint | 0 口井命中，等同已得 6.604 的 no-hedge，未重复提交 |
| PF branch cap 2.5 target-free | **6.654**，较最佳差 0.177，NO-GO |
| PF branch cap 4.0 target-free | **6.661**，较最佳差 0.184，NO-GO |
| Functional Ridge residual curve | ref `55178775`：**6.509**，较最佳差 0.032，NO-GO |
| Fixed 75/25 Functional Ridge–PLS | ref `55179387`：**6.482**，较最佳差 0.005，NO-GO |
| cap4 + Fixed Ridge–PLS | cap4 Public 门槛失败，条件组合永久关闭 |

当前 6.477 比公开榜前 10% 边界差约 0.058。公开榜只覆盖约 26% 测试数据，最终排名
由约 74% 的 Private 数据决定，所以项目不会把公开榜位置当作最终结论，也不会为了
小幅 Public 提升牺牲跨井稳定性。

详细时间线见 [进展日志](docs/progress_log.md)，竞赛选择和最终冻结依据见
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

### 5.10 MHA180 target-free midpoint 审计

公开 Notebook `youill0317/rogii-mha180sep3-public-repro` 同时包含运行时 PF 双峰 midpoint
修正、由排行榜 canary 解码的全局 -0.40 偏置以及开关控制的 probe。私有构建
`heralces/rogii-mha180-target-free-v1` 只保留目标无关的运行时 midpoint 分支：
`alpha=1.8`、minor mass 至少 0.22、双峰间距 3--40 ft、单井修正上限 4 ft；删除
canary 偏置、probe 和后续排行榜诊断。最终报告验证了 14,151 行、ID 同序且唯一、
预测全为有限值，并用 SHA-256 绑定报告与 `submission.csv`。

普通 GPU 运行约 889 秒。运行时只发现 3 口具有 PF 统计的井，但它们都已由更早的
train/test overlap 保护事务处理，midpoint 分支按设计全部跳过：命中井数为 0，
移动行数为 0。最终文件 SHA-256 为
`fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9`，与此前
“去掉 +2 ft hedge”的提交逐字节相同；该文件已有 Public RMSE **6.604**，比当前
6.477 差 0.127。因此不为相同文件创建新的 Code Submission，也不围绕公开榜来源的
`alpha` 继续调参。该路线判定为 **NO-GO**。

### 5.11 Target-free PF branch cap 2.5

7 月 28 日的新扫描发现 `mtoshidesu/rogii-test-another-approch-2nd`：作者当前公开榜
成绩为 6.407，但公开 A23 尾部把井 `00e12e8b`、4,301 行、历史输出哈希和公开分数
直接写入评分事务。该固定井实现不进入我们的候选。

可验证的算法增量位于更早的通用 PF seed-branch selector。它根据运行时双峰中心、
双峰质量和加权中心计算修正：

$$
\Delta_w=\operatorname{clip}\left[
0.60\left(\frac{c_{low}+c_{high}}{2}-c_{weighted}\right),
-C,C
\right].
$$

现有 6.477 管线使用 $C=2.0$ ft；新候选只把通用上限改为 $C=2.5$ ft，保持
minor-mass 门槛 0.25 和双峰间距 4--40 ft 不变。私有构建删除固定测试井、固定行数、
固定哈希、排行榜分数假设及 A23 后处理，并关闭无关 dry-run probe；最终报告将逐项
绑定动态样本、分支报告、移动行数和输出 SHA。普通 GPU v1
`heralces/rogii-pf-branch-cap25-targetfree-v1` 在 779.6 秒内完成：通用选择器命中
1 口井，4,301 行统一移动 +2.5 ft，输出与公开 A23 文件逐字节一致。唯一一次受控
Code Submission 得到 Public RMSE **6.654**，比 6.477 主候选差 0.177，也比移除
分支 hedge 的 6.604 差 0.050。公开作者的 6.407 因而不能归因于这个可复现改动；
该候选判定为 **NO-GO**，不替换最佳，也不围绕 cap 继续公开榜调参。

### 5.12 TCN target-free 逐井 gate

TCN v2 的 pooled RMSE 有明确改善，但原始 0.20 overlay 只在 44.71% 的验证井上不变差。
为了判断收益是否能在不知道标签的情况下被识别，新增
`scripts/analyze_sequence_tcn_well_gate.py`，直接冻结 TCN checkpoint、0.20 scale 和
所有超参数，不重新训练序列模型。gate 只读取 51 个无标签输入特征的逐井统计、强基线
预测和 TCN 输出统计；绝对坐标仅用于把 170 口验证井划成五个空间子区域，不作为模型
输入。相同 Typewell 指纹始终放在同一外层 fold，每口井的 gate 概率均由未见过该空间
子区域标签的模型产生。

为防止“未选择的井保持原值”机械抬高不变差比例，除原 pooled、p95、worst 和覆盖率
门槛外，还预先要求至少选择 20% 的井、被选择井胜率至少 55%、五个空间 fold 至少四个
改善且每折都实际选择井。结果为：

- 选择 82 / 170 口井，即 48.24%；
- 被选择井胜率只有 **51.22%**；
- gated pooled delta 只有 **-0.006155**，未达到 -0.020；
- 五个空间 fold 只有 2 个改善；
- gate 概率与实际逐井收益几乎无相关性。

因此 gate 为 **FAIL / NO-GO**。它没有授权 TCN 多折 GPU 训练，也没有生成提交。

### 5.13 A28 runtime-fast exact-core

旧 A28 普通运行有效，但 Code Submission 超过九小时。阶段审计发现它把 SP45 selector
从 `500 particles × 128 seeds` 提升到 `1000 × 256`，并在生成 SP45 与 learned
轨迹后继续运行可见前缀、模型包和分支 hedge；这些后续输出最后都会被 A28
`0.62 × SP45 + 0.38 × learned` 原子覆盖。

新的私有构建 `heralces/rogii-a28-runtime-fast-v1` 复用已成功完成隐藏提交的 exact
核心配置 `500 × 128`，在两个必要轨迹文件生成后立即融合并停止。构建器删除固定演示
井和无关 probe，不读取排行榜分数、固定测试井、固定行数或预期输出哈希。普通 v1
在 **590.3 秒**完成；14,151 行与官方 sample ID 完全同序且唯一，预测全为有限值，
三份报告与最终 `submission.csv` 的 SHA-256
`9307dc0e30eda8bca81c4d03f5828da340d8d98e5585dda3981f312b12696033`
一致。

唯一一次受控 Code Submission 得到 Public RMSE **7.678**。它比 6.477 主候选差
1.201，比 2026-07-29 的公开 top-10% / top-5% 边界分别差 1.226 / 1.264，也没有
复现公开来源的 6.433。运行时问题已解决，但模型候选本身被明确否定；A28
runtime-fast 是 **NO-GO**，不替换最佳，也不继续围绕 0.62 权重调参。

### 5.14 公开候选复查、全局偏置与 balanced TCN

7 月 29 日再次下载最新公开产物后，`my0705` 与 `tamerlan` 的强候选都被还原为同一个
固定 Q0522 文件，SHA-256 为
`8328487738fbb7e1e89c733b80367ced2281e6dd19f4746804adfd77a923071d`。
两份 Notebook 都从与当前 6.477 主候选一致的 pre-Q0522 b192 文件出发，再对固定测试井
`00e12e8b` 的 4,301 / 14,151 行应用可见井修正；因此没有形成新的 hidden-safe 候选。

同时复查了“给全部 OOF 预测加一个全局标量偏置”的低成本方向。它在所有严格协议上都
恶化 pooled RMSE：

| OOF 协议 | pooled delta | 改善 fold |
|---|---:|---:|
| `spatial_cluster` | +0.013658 | 1 / 5 |
| `spatial_cluster_6` | +0.017874 | 2 / 6 |
| `spatial_cluster_7` | +0.022172 | 3 / 7 |
| Typewell fingerprint | +0.019451 | 1 / 5 |

原 TCN 使用按井长度加权的窗口抽样，0.20 overlay 虽把 pooled RMSE 改善
**-0.055218**，却只有 **44.71%** 的井不变差。为了检验收益是否只是被大井主导，
额外运行了一次预注册的 uniform-well pilot：每口训练井被等概率抽样，协议仍冻结为
`spatial_cluster_7_seed_20260727` 的 fold 0，主 scale 0.20、4,000 steps，且不写
`submission.csv`。Tesla T4 运行 **83.58 秒**后得到：

- base pooled RMSE 9.639581，0.20 overlay 为 9.689167，delta **+0.049586**；
- well-level p95 delta **+0.104819**，未通过不高于 +0.10 的门槛；
- worst-well delta **-0.774178**，通过不高于 +0.50 的门槛；
- 不变差井比例仍为 **44.71%**，未达到 50%；
- pooled、p95、覆盖率三项失败，只有 worst-well 一项通过。

最终报告为 **FAIL**，`full_multifold_authorized=false` 且
`kaggle_submission_authorized=false`。均衡抽样没有扩散原 TCN 的逐井收益，反而把
pooled 改善翻转为恶化；这条路线停止，不调 scale、不增加 steps、不训练多折，也不创建
竞赛提交。

### 5.15 GR 最强周期分量 notch

公共全栈源码中长期关闭的 `RUN_GR_FFT_DENOISE` 是少数尚未被隔离验证、且完全目标无关的
结构增量。实验在查看任何 notch 分数前冻结：至少 64 个有限 GR 样本，线性填补仅用于
频谱计算，移除最强非直流 `rFFT` 频点及左右各一个频点，再用完全不变的八 seed
target-free PF/beam 配置评估全部 773 井。它不读取标签、井 ID、formation、排行榜、
probe、canary 或公开预测文件。

六进程全量运行在 **716.7 秒**内完成，773 / 773 井成功，notch 在全部井上激活：

- pooled RMSE：12.518489 → 16.811012，delta **+4.292523**；
- macro-well RMSE：9.104812 → 13.448156，delta **+4.343344**；
- 225 井改善、548 井恶化；
- paired-well bootstrap 95% 区间为 **[+3.0695, +5.5427]**；
- 五个空间簇全部恶化，delta 从 +2.9535 到 +5.3763。

预注册的 pooled、macro、bootstrap、4/5 空间簇和最坏空间簇五项门槛全部失败。强周期峰值
显然主要承载地层匹配信号而非可安全删除的采集噪声。该路线为 **NO-GO**：不调 notch
半径、频点或激活阈值，不启用 GPU，不构建 Notebook，也不提交。

### 5.16 NCC 去重与 family-balanced OOF stack

继续审计公开全栈后，确认多尺度 GR-NCC 并不是独立的新信号：现有 learned/model-package
分支已经包含 `pred_delta_drift_ncc_*` 的 LightGBM、CatBoost、HGB 与 XGB 轨迹。因此
没有把直接 NCC 重新包装成候选，也没有消耗一次排行榜提交。

剩余的低成本结构问题是五个公开 grouped-OOF 模型按 `3 × LightGBM + 2 × CatBoost`
并列进入正约束 Ridge，是否因家族数量不均形成隐性过权。提交
`1e7e360` 在读取新结果前冻结唯一候选：先分别求三个 LightGBM 和两个 CatBoost 的
家族均值，再用完全相同的正约束 Ridge、`0.91 / 0.09` Ridge/PF 权重和 `tau=85`
深度 ramp。五折 meta OOF 仍按整井分组；Typewell 五分区和固定 seed 的七个空间簇只做
稳健性切片。由于没有在这些切片内嵌套重训五个昂贵基模型，即使 PASS 也只会授权进一步
集成审计，而不会直接授权提交。

CPU 全量审计在 **51.2 秒**内覆盖 3,783,989 行、773 井：

- 五列基线 pooled RMSE 10.370056，家族平衡后 10.399843，delta **+0.029786**；
- macro-well delta **+0.001060**，386 井改善、387 井恶化；
- paired-well bootstrap 95% 区间为 **[-0.007816, +0.067396]**；
- Typewell 分区只有 2 / 5 不变差，空间分区只有 3 / 7 不变差；
- 最差分区 delta **+0.103686**，p95-well delta **+0.148644**；
- worst-well delta +0.433701 是唯一通过的保护项。

其余七项预注册门槛全部失败。原正约束 Ridge 已能把弱冗余列压到零；强行家族内平均反而
丢失有用差异。这条路线为 **NO-GO**，不调家族权重、Ridge alpha、PF 权重或 ramp，
不构建 Notebook，也不提交。

### 5.17 最终候选冻结与可复现审计

2026-07-31 使用已认证 Kaggle API/CLI 重新核对全部九个提交、对应 Notebook 终态和本地
产物。目前 Kaggle 的最终入选列表为空；这不是模型错误，但若一直不处理会把最终选择权
留给平台自动规则。因此项目把 **2026-08-05 20:00 UTC** 设为内部选择截止，较官方
`2026-08-05 23:59 UTC` 保留约四小时恢复窗口。

首选冻结为 Public RMSE **6.477** 的 exact full stack：

- submission ref：`54952634`；Notebook：`heralces/rogii-pf-public-reproduction`，
  version id `337405234`，终态 `COMPLETE`；
- `submission.csv` 为 14,151 行、列顺序 `id,tvt`、ID 与官方样例完全同序且唯一、预测全为
  有限值；
- SHA-256：`b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`。

第二候选建议保留 Public RMSE **6.604** 的 no-hedge 消融：

- submission ref：`54967104`；Notebook：`heralces/rogii-pf-public-no-hedge`，
  version id `337779623`，终态 `COMPLETE`；
- 同样通过 14,151 行、列、ID 顺序/唯一性和有限值检查；
- SHA-256：`fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9`；
- 它与主候选仅在一个运行时命中的 PF 分支上不同：4,301 行全部少 2 ft，两文件 RMSE
  距离为 1.102608。它不是独立模型，但以较小 Public 代价覆盖了最终分支 hedge 的方向风险。

2026-08-02 登录后的 Submissions 页面最初显示 **0/2**，并说明最多选择两份；若少于
两份，Kaggle 会自动从最高分提交中选择。得到用户确认后已主动选择
`54952634 + 54967104`，页面显示 **2/2**，独立 API 返回的 ref 集合也精确匹配，避免了
把最终组合交给自动规则。Public 7.514 的 safe-v2
(`54927581`) 保留为结构独立的灾备 Notebook，
但与主候选相差 1.037，当前不建议挤占第二个最终名额。最终操作前仍需在 Kaggle 页面
再次核对选中数量和 Notebook 版本，禁止创建重复提交。

同期审计 `farhanabidtech786/rogii-best-score-wellbore-geology-prediction` 后，确认其最终
文件 SHA 与 6.477 主候选逐字节相同，且源码仍是同一 public full stack/分支 hedge，
不是新的独立信号，故不构建镜像 Notebook、不提交。

### 5.18 Public 6.432 分支延伸复现

2026-08-02 的新公开候选扫描发现
`raunakdey07/rogii-physics-informed-stacked-ensemble-v2` 对应团队 Public RMSE 为
**6.432**。源码虽然包含耗时约 5.5 小时的 MDN/GBDT 实验，但作者最终提交的并不是
新模型输出：其 `submission.csv` 仅在当前 6.477 主候选运行时命中的同一 PF 分支上再加
2 ft，共影响 4,301 行，其余两口井完全不变。该文件 SHA-256 为
`af43d97ab5b76f7f0bbbc5c72fb6e9a829e939c48582fc4130144340547360a2`。

项目没有复制公开 Notebook 中的固定井、固定行数、预期哈希、probe 或榜单反推逻辑，
而是新增私有动态构建器 `build_branch_cap40_targetfree_v1.py`：仍由本次运行生成的
`PF_SEED_BRANCH_STATS` 选择合格分支，只把通用 hedge strength 从 0.60 调到 0.62，
并把位移上限从 2.0 调到 4.0 ft。候选选择本身明确标记为
`candidate_selection_public_score_informed=true`，但运行时评分路径不读取任何 Public
分数或可见测试常量。

审计过程保留了一个有价值的失败门槛：v1 使用 strength 0.60 时只移动
3.908582 ft，SHA 不匹配，因此没有创建竞赛提交。修正后的私有 Notebook v2 运行
785.8 秒，动态命中一口井、移动 4,301 行、精确 +4.0 ft；14,151 行、`id,tvt`
列、官方 ID 顺序、ID 唯一性、有限值、报告 SHA 和逐井增量全部通过检查。普通可见运行
的最终 SHA 与上述 6.432 文件逐字节一致，但 Code Competition 会在隐藏数据上重新执行
Notebook；可见 CSV/SHA 相同并不能证明隐藏评分路径和 Public 分数也相同。cap 2.5 的反例
已说明这一点，因此 6.432 只作为候选来源信号，ref `55177746` 的实际分数必须等待 Kaggle
终态。

唯一一次 Code Submission 为 ref `55177746`，描述
`Target-free PF seed-branch cap4.0 controlled verification`，终态 Public RMSE 为
**6.661**，比当前 6.477 主候选差 **0.184**，也比 cap2.5 的 6.654 略差。该结果否定了
“更大通用分支位移可迁移到隐藏井”的假设；cap4 不替换任何最终候选，所有 cap4 +
Functional Ridge/Ridge–PLS 条件构建均因组件 Public 门槛失败而永久关闭。当前已选的
`54952634 + 54967104` 保持不变。

### 5.19 Target-free well-functional residual Ridge

为避免继续只围绕同一 PF 分支做榜单微调，项目实现了一条结构独立的残差曲线迁移路线。
每口井先用 51 个测试时可用、已移除绝对位置的序列特征，在 0%、25%、50%、75%、100%
五个相对深度位置取样，形成 255 维井级签名；训练目标是该井相对竞争性 PF/Ridge OOF
基线的 64 点残差曲线。部署时将预测曲线插值回隐藏后缀，仅以冻结权重 0.20 叠加，并把
单点残差裁剪到 ±20 ft。模型不把井号、绝对坐标、测试标签、固定可见井或排行榜分数作为
特征。

第一项预注册实验使用 16 邻居的曲线迁移，空间 7 折 pooled delta 为 **+0.008253**，
Typewell-fingerprint 5 折为 **+0.028481**，双协议均恶化，因此冻结为 NO-GO，不再调邻居数
或距离。随后只允许一次结构不同的多输出 Ridge：`alpha=100`、`scale=0.20`、64 点曲线和
±20 ft 裁剪均在读取结果前固定。它首次在两套严格协议上同向通过：

- 空间 7 折：10.370043 → 10.342440，pooled delta **-0.027603**，5/7 folds 改善，
  p95-well delta -0.07058，worst-well delta +0.19252；
- Typewell 5 折：10.370043 → 10.351276，pooled delta **-0.018767**，5/5 folds 改善，
  p95-well delta -0.05606，worst-well delta +0.01616。

本地可见部署在三口测试井上生成有限、全覆盖且幅度温和的修正，全体 14,151 行 correction
RMS 为约 0.410 ft，范围约 [-1.156, +0.670] ft。私有 Notebook
`heralces/rogii-functional-ridge-v1` version 1 已运行；它把该层插在当前
6.477 full-stack 的 PF 分支之后、最终审计之前，并额外校验动态样本长度、`id,tvt`、ID
同序/唯一、有限值、逐井覆盖、报告 SHA 和源代码 SHA。需要诚实保留的风险是：部署父级
full-stack 与用来守门的竞争性 OOF base 相关但不完全同构，所以只有 Kaggle 产物契约通过后
才允许创建一次受控提交，当前尚未提交比赛。v1 在 766 秒后于 overlay 起点主动失败：缓存
证明字段实际位于 `metadata["contract"]`，而 v1 误读了顶层字段。父级 full-stack、输入
挂载、源代码 SHA 和 `submission.csv` 均已正常生成；错误发生在新模型拟合前，不产生候选
分数。修复只更正契约路径，并同时验证 training-only、无 target leakage 和无井号编码，
不改变任何模型参数或预测逻辑；修复版已作为同一私有 Notebook 的 v2 唯一推送并完成。
v2 用时 846.0 秒；下载产物的 14,151 行与官方 ID 完全同序、ID 唯一、预测有限，父文件、
最终文件、双报告 SHA 和三口井逐井增量全部一致，最终 SHA-256 为
`a30f6384...180de`。已从 v2 创建唯一一次 Code Submission：ref `55178775`，最终 Public
RMSE 为 **6.509**，较 6.477 差 0.032，不能改变当前最佳或最终入选组合。

等待 Kaggle 运行期间又预注册了一项且仅一项低秩响应消融：每个外折只用训练井把 64 点
残差曲线压缩成 8 个 PCA 系数，再用相同 Ridge 预测。8 维已解释约 96.9%–97.1% 的训练
曲线方差，但空间 pooled delta 为 **-0.027280**、Typewell 为 **-0.018631**，分别比普通
Ridge 差 0.000323 和 0.000136，双双未达到“至少再改善 0.002”的预注册支配门槛。因此
低秩版本为 NO-GO，不调 PCA 维数、alpha、scale 或 clip，也不构建 Notebook；普通
Functional Ridge 保持唯一部署候选。

为压缩截止日前的响应时间，仓库还只离线准备了组合构建器
`build_functional_ridge_cap40_v1.py`。它冻结 cap4 的 strength 0.62 / cap 4.0 和上述
Functional Ridge 参数，并增加跨报告 SHA 与分支上限审计。该 Notebook 当前明确标记为
`PREPARED_NOT_AUTHORIZED`：只有 cap4 ref `55177746` 和普通 Functional Ridge 的独立
Kaggle 分数都严格优于 6.477 时才允许推送；任一单项失败就不运行组合，避免用交互猜测
浪费提交额度。cap4 后续以 6.661 完成，已触发失败条件；该组合现为
`CLOSED_PUBLIC_GATE_FAIL`，不推送 Notebook、不创建提交。

### 5.20 增量公开审计与去固定井 GR datum 复核

2026-08-02 第二次低频扫描覆盖 6,056 支队伍和最新 200 个公开 Notebook。Georgy 当前
团队分数 6.320 的 Notebook 只分析噪声上限，没有生成提交；Evgendvorkin 当前 6.405 的
最终 `submission.csv` 与我们的 cap4 可见产物 SHA `af43d97a...60a2` 完全相同，仍不是
新模型。Yaroslav 当前 6.419 的 Notebook 在 cap4 后增加了 GR/Typewell datum transfer，
但先固定 `00e12e8b`、4,301 行和 1.75 ft 目标位移，再用该可见测试井反推出全体缩放，
不能直接视为隐藏安全候选。

项目只抽取其 target-free raw datum 机制，并在读取结果前冻结干净的跨井校准：每个外折
只用训练井真值拟合无截距、按隐藏行数加权的单一 scale，scale 限制 ±0.25，最终逐井
常数修正限制 ±2.5 ft；不使用固定测试井、行数、哈希或榜单分数。773/773 井都成功生成
raw signal，但 scale 在不同 folds 间频繁变号。空间 7 折 pooled delta **+0.008490**、
0/7 folds 改善；Typewell 5 折 **+0.033743**、0/5 folds 改善，p95 delta +0.05872，
worst-well delta +0.38970。结论是公开交易的效果依赖可见井锚定，不能跨井迁移；该路线
为 NO-GO，不调 grid、平滑宽度、scale guard 或 correction cap，也不构建 Notebook。

### 5.21 截止日前公开前沿去重

2026-08-02 再次以 UTF-8 完整读取公开 Notebook 前沿。Roman、Nikita、Raunak Ultra、
Boristown 与 i-aztec 的候选均是现有 full-stack/cap4 的克隆或展示层；Nikita 的运行产物
明确只对固定井 `00e12e8b` 的 4,301 行追加 `+0.522 ft`，最终预测哈希仍是已审计的
Q0522 固定公开候选。Roman 的所谓 CatBoost baseline 最终同样在 cap4 文件上对这口井
追加 `+0.522 ft`，没有形成新的隐藏井模型。

Blacklions 的 two-coefficient well-level GBDT 是本轮唯一结构上新增的模型，但其真实运行
报告已经自行否决：随机按井 5 折的 OOF 从 10.372253 恶化到 10.415607，gain 为
**-0.043355**；well-level P90 进一步恶化 **+0.301558**，median 恶化 **+0.045423**。
作者门控因此为 `oof_gate_failed`，最终文件回退到固定 Q0522 锚点。由于它连较宽松的
随机按井 OOF 都失败，不再消耗算力做更严格的空间/Typewell 复验，也不构建 Notebook
或提交。

### 5.22 HMM 深度仿射倍率复核

等待 Kaggle 评分期间预注册并运行了一次两参数深度仿射校准：固定 HMM scale1 修正乘以
从隐藏后缀起点倍率到终点倍率的线性插值。两个倍率只在每个外折的训练井上拟合，限制在
`[0, 2.3]`；排除 32 口 HMM discovery 井和三口当前可见测试井，不使用井号、绝对坐标、
榜单分数或 probe。

Typewell 5 折相对 scale2 额外改善 **0.006569**、4/5 折改善，单协议通过；空间 7 折
只额外改善 **0.007625**（低于 0.01 门槛），且仅 5/7 折改善（低于 6/7 门槛），因此
总体 `FAIL_STOP`。更关键的是全量拟合的起点和终点倍率都等于上限 **2.3**，候选实际
退化成已经预注册的全局 scale2.3，而没有发现可迁移的深度效应。项目不再调分段点、
基函数或倍率上限，不构建 Notebook 或提交。

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
| A28 runtime-fast exact-core | 7.678 | 较 6.477 差 1.201，NO-GO |
| Frontier II DYNQ0196 | 7.507 | 较 6.477 差 1.030，作者 6.39 未验证，NO-GO |
| Sequence TCN v1/v2 | 不提交 | pooled -0.0552，但仅 44.71% 井改善，NO-GO |
| Sequence TCN target-free well gate | 不提交 | gated pooled -0.0062、选择井胜率 51.2%，NO-GO |
| Sequence TCN uniform-well pilot | 不提交 | pooled +0.0496、仅 44.71% 井改善，NO-GO |
| GR strongest-periodic notch | 不提交 | 773 井 pooled +4.2925，五个空间簇全恶化 |
| Family-balanced OOF stack | 不提交 | pooled +0.0298，仅 2/5 Typewell、3/7 空间分区不变差 |
| Harshini 空间曲面 + PF | 不提交 | 严格空间核心 17.944，差于 flat 14.969，NO-GO |
| MHA180 target-free midpoint | 不重复提交 | 0 口井命中；文件等同已得 6.604 的 no-hedge |
| PF branch cap 2.5 target-free | 6.654 | 较 6.477 差 0.177，NO-GO |
| Functional kNN residual curve | 不提交 | 空间/Typewell pooled 均恶化，NO-GO |
| Functional Ridge residual curve | 6.509 | 较 6.477 差 0.032，NO-GO |
| Functional Ridge 8D response-PCA | 不提交 | 双协议略差于普通 Ridge，未过支配门槛 |
| Functional RBF Kernel Ridge | 不提交 | 空间/Typewell pooled 均恶化，非线性相似度 NO-GO |
| Augmented Functional Ridge | 不提交 | 空间明显改善，但 Typewell 仅 4/5 折且弱于普通 Ridge |
| Supervised Functional PLS | 单模型不提交 | pooled 双协议显著改善，但 Typewell fold 0 恶化 |
| Fixed 75/25 Ridge–PLS | 6.482 | 比 plain Ridge 改善 0.027，但较最佳差 0.005，NO-GO |
| Distance-gated Ridge–PLS | 不提交 | pooled 更优但 Typewell 仅 4/5 折，门槛 FAIL |
| cap4 + Fixed Ridge–PLS | CLOSED_PUBLIC_GATE_FAIL | cap4 为 6.661，不优于 6.477；不推送 |
| HMM scale2 + Fixed Ridge–PLS | CLOSED_PUBLIC_GATE_FAIL | Ridge–PLS 为 6.482，组件 Public 门槛失败；不推送 |
| GR/Typewell datum transfer | 不提交 | 空间与 Typewell 全部 folds 恶化，固定井收益不可迁移 |
| Bidirectional HMM smoother | 不提交 | 相对 forward HMM2 pooled 改善 0.0633，但 true-p95 +0.4044 且平均 46.7 秒/井，门槛 FAIL |
| 严格跨井平移副本转移 | 不提交 | 773 井中无不同 ID 的完整动态指纹重复，0 口井可转移 |
| Exact-Typewell 最近井模板 | 不提交 | 34 井 pooled 10.0945 → 10.4283，全体 OOF 恶化 0.01336，NO-GO |
| Global Viterbi HMM | 不提交 | pooled 相对 HMM2 改善 0.3578，但 p95 +1.4213、worst +2.7004，尾部 FAIL |
| Uncertainty-fused Viterbi HMM | 不提交 | 独立 32 井 pooled 反而恶化 0.00340，50% 井改善，NO-GO |
| Well-level two-coefficient GBDT | 不提交 | 作者随机按井 OOF 恶化 0.0434，P90 恶化 0.3016；回退固定 Q0522 |
| HMM depth-affine scale | 不提交 | Typewell PASS；空间仅增益 0.00763、5/7 折，且双端点均撞 2.3，退化为全局 scale2.3 |

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

1. 保留 6.477 为当前主候选，不重复提交任何 A28/Pair 版本；
2. A28 runtime-fast 已解决超时但仅得 7.678；不继续围绕 0.62 权重或粒子数调参；
3. Dip-aware HMM + GBM 已取消且无产物，不根据未经验证的 5.14 声称提交；
4. Frontier II 已以 7.507 判定 NO-GO；不根据作者自报的 6.39 重复调参；
5. TCN v2、target-free well gate 与 uniform-well pilot 均已失败；不继续多折 GPU、
   调 scale 或增加 steps；
6. GR strongest-periodic notch 已在 773 井严重恶化；不调 notch 半径、频点或阈值；
7. 直接 NCC 已被 learned 分支吸收，family-balanced stack 也已失败；不重复包装或调家族权重；
8. Harshini 曲面路线已按严格空间门槛停止；不运行 3.46 小时 boosted 多折或隐藏提交；
9. MHA180 target-free midpoint 已证明退化为 6.604 no-hedge；不重复提交或继续公开榜调参；
10. PF branch cap 2.5 已以 6.654 判定 NO-GO；不围绕公开作者的 6.407 继续调 cap；
11. 已冻结并复核 6.477 主候选、6.604 no-hedge 备选和 7.514 safe-v2 灾备；主/备均有
    精确 submission ref、Notebook version、终态、动态 CSV 契约与 SHA 记录；
12. 8 月 2 日选择预检和执行均已完成：页面确认上限为两份并已主动选择
    `54952634 + 54967104`，当前 2/2，不依赖 Kaggle 自动选择；
13. 截止前只做低频只读核验；若选择集合无异常，不再更换最终 Notebook、不新增探榜提交。
