# ROGII Wellbore Geology Prediction：公开研究纪要

> 快照时间：2026-07-23（Asia/Shanghai）
> 范围：仅使用 Kaggle 官方公开页面、公开讨论、公开 Notebook 源码及已登录 Kaggle CLI 的只读结果；未提交、组队、发帖、投票或修改任何 Kaggle 状态。
> 证据分级：**官方/主办方说明** > **公开源码可核查实现** > **社区实验报告** > **Notebook 标题中的自报分数**。

## 1. 结论先行

- 竞赛截止时间为 **2026-08-05 23:59 UTC（北京时间 2026-08-06 07:59）**，早于 8 月 8 日；指标是对所有提交行计算的 pooled RMSE。[竞赛页](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- 这是**轨迹/序列反演问题，不是独立行回归问题**。公开证据最一致的强路线是：以最后已知 TVT 为锚预测残差，结合受约束的 PF/beam/DTW 类 GR 对齐候选、GBDT/序列模型、`U = TVT + Z` 的平滑地质先验，再做 OOF 学习的双轨融合。
- 最重要的胜负手不是继续堆更多相似 backbone，而是：
  1. 构建严格、贴近隐藏重跑的验证；
  2. 处理长段漂移和少数灾难井；
  3. 用不确定性和前缀回测做保守的逐井候选选择；
  4. 坚决隔离公开可见测试样例带来的泄漏。
- Kaggle 官方数据页明确说明，当前可见 `test/` 只是少量训练集示例，提交 Notebook 在隐藏重跑时会被实际约 200 口测试井替换。[数据说明](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data) 因而同 ID 覆盖、静态行号映射、公开 CSV 查表、Leaderboard probe/bias 都不能作为私榜方案。

## 2. 官方机制与数据事实

- 每口训练井包括 horizontal CSV、typewell CSV 和 PNG；horizontal 中有 MD/X/Y/Z/GR/TVT_input，训练集另有 formation surfaces 和目标 TVT；typewell 中有 TVT、GR、Geology。[数据说明](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)
- 只允许 Notebook 提交；CPU/GPU 最长 9 小时、关闭互联网；允许使用公开且免费可得的外部数据/预训练模型；最终必须生成 `submission.csv`。[代码要求](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/code-requirements)
- 2026-07-23 快照：5,520 支队伍、114,669 次提交。官方曾：
  - 修复隐藏重跑数据集的后端问题，无需重新下载数据：[官方修复说明](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697400)。
  - 从 private scoring 排除一口异常井；只影响私榜，不改变公榜：[官方私测更新](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707695)。

## 3. 讨论区信号

以下票数与排序均为 2026-07-23 快照；社区数字不是官方结论。

### 高票主题

| 主题 | 票数 | 可用结论 |
|---|---:|---|
| [问题示意图](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697418) | 166 | 适合建立 TVT/轨迹/typewell 的直觉；回复中的数值仍属社区经验。 |
| [可视化 UI](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424) | 92 | 说明逐井可视化、轨迹诊断是高价值工具。 |
| [主办方地质学建议](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698825) | 73 | 邻近井 formation dip 应相近；横井自身 GR 相关性有价值；进入负 TVT 后可优先与预测起点前的横井 GR 对齐。 |
| [MTP / CNN 多轨迹设想](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853) | 61 | 反问题具有多解性，应保留多条候选路径/概率，而非只回归单一路径。 |
| [Formation 解释](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697406) | 60 | 地质背景参考。 |
| [DTW 路线](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431) | 45 | 受方向和局部约束的动态时间规整可产生候选；普通 DTW 会因多值、反向和周期歧义失败。 |
| [问题拆解](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367) | 37 | 公开基线思路汇总。 |
| [重复 typewell](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449) | 35 | 主办方确认重复是有意的人工选择；typewell 不一定是今天空间上最近的井，也可能是先前横井解释得到的 pseudo-typewell。 |
| [Forward simulation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702474) | 32 | 可作为物理候选生成思路。 |
| [朴素 XGBoost 的瓶颈](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701041) | 30 | 独立行模型无法充分处理序列对齐和累计漂移。 |
| [空间/序列上下文](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699289) | 29 | 强调跨行上下文。 |
| [`TVT=f(MD,Z)` 线性先验](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326) | 28 | 几何先验可作弱基线或正则，但不能独立解决局部层位变化。 |

### 热门与近期主题

- [Top-team signal 在哪里？](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465)：发帖者的分组井 CV 约 8.8；其诊断显示每井平滑 oracle 可低很多，且近距离同 frame 邻井只覆盖少量井。另一方面，当时公榜第 8 的参赛者回复称不使用邻井也能做到单模型 CV < 5。**公开证据相互冲突**：GR 序列模型应作为主线，邻井迁移只作为严格验证、距离门控的可选分支。
- [本地验证讨论](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727570)：当时 rank 8 参赛者报告 5 folds × 5 seeds CV 4.98、公榜 5.7；其他参赛者常见 CV 6–8、公榜 8–10。复杂融合、预训练权重、GPU 非确定性会放大 CV–LB 偏差。
- [Sub-6 与 field CV](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727149)：社区实验称 heel gain/offset 校准有效；PF posterior spread 与误差有弱相关，可用作置信度；邻井借用只在极近距离子集稳定有益；whole-field holdout 比普通 grouped split 更差但差距因管线而异。
- [最差井诊断](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723815)：多名高排名参赛者在同一难井出现约 40+ RMSE，错误由早期漂移开始并持续累积。pooled RMSE 会被长井/灾难井主导，应设计漂移检测、重置/回溯和保守 fallback。
- [六种架构的共同盲点](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726834)：多个不同模型在长段上高度相关地偏错，提示层序周期歧义/不可观测模式；继续平均相似错误不会自动改善。
- [初学者 row-wise ML 失败](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726751)：公开实验中 anchor-hold OOF 约 16.1，而直接 row-wise LightGBM 约 17.46，再次支持以轨迹为单位建模。

## 4. 高票公开 Notebook

下表按 Kaggle CLI 的 `voteCount` 排序。标题中的 “LB x.xxx” 是作者自报的**公榜**信号，并非私榜保证；公开源码之间大量复用模型包和中间产物，票数也不等于可复现性。

| 排名 | Notebook | 票数 | 源码可见的核心思路 |
|---:|---|---:|---|
| 1 | [9.251 DWT-based](https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based) | 690 | 实际使用受约束 DTW、PF、beam、多尺度 NCC、LGBM/CatBoost、hill-climb 融合。 |
| 2 | [LB 7.295 Public Rebuild](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild) | 572 | 双管线、PF/beam/GBM、contact/前缀校准；最新版含明显公榜 bias/probe 逻辑，需拆除。 |
| 3 | [Dual-track Prefix-calibrated](https://www.kaggle.com/code/pilkwang/rogii-dual-track-prefix-calibrated-geosteering) | 338 | residual ensemble、物理 selector、`U=TVT+Z` 投影、heel 校准、前缀伪回测；也有小幅全局 bias。 |
| 4 | [Dual-pipeline Self-verifying](https://www.kaggle.com/code/lightningv08/rogii-dual-pipeline-self-verifying) | 290 | 两条独立预测管线与守卫式自检。 |
| 5 | [Wellbore Prediction Ridge](https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-ridge) | 281 | GroupKFold；3 LGBM + 3 CatBoost + 正权重 Ridge；PF/beam、SG 平滑、鲁棒四次投影。 |
| 6 | [Dual-pipeline Blend](https://www.kaggle.com/code/pixiux/rogii-dual-pipeline-blend) | 233 | ridge-SP45 与 likelihood-PF/KNN/GBM 双轨 0.55/0.45；含仅对可见重叠样例生效的 same-ID override。 |
| 7 | [7.091 Public](https://www.kaggle.com/code/AmgedAlfaqih/7-091-public) | 227 | 高分公开方案，需单独审计其挂载资产和隐藏重跑兼容性。 |
| 8 | [Public Score 7.159](https://www.kaggle.com/code/degnonguidi/public-score-rogii-lb-7-159) | 224 | 公榜分数导向方案，不能把标题分数当私榜证据。 |
| 9 | [Target-free TVT Geosteering](https://www.kaggle.com/code/pilkwang/rogii-target-free-tvt-geosteering) | 220 | ridge/PF anchor、鲁棒 `U` 投影、预训练 LGBM + likelihood-PF、双轨融合。 |
| 10 | [hahaha nondet AGI](https://www.kaggle.com/code/johnjanson/hahaha-nondet-agi) | 218 | 高票但标题已提示非确定性，应先做多次重复运行审计。 |
| 11 | [XGB Starter CV 15](https://www.kaggle.com/code/cdeotte/xgb-starter-cv-15) | 212 | 按井 GroupKFold、模拟 TVT_input 缺失区间、预测相对最后锚点的 residual；是干净验证骨架。 |
| 12 | [Hill Climbing](https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing) | 198 | OOF 融合权重搜索。 |
| 13 | [LB 7.201](https://www.kaggle.com/code/baidalinadilzhan/rogii-lb-7-201) | 192 | 公榜方案，需审计依赖资产。 |
| 14 | [Super Solution LB Top-3](https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3) | 177 | 高公榜导向融合，私榜泛化未知。 |
| 15 | [H-blend v1](https://www.kaggle.com/code/nina2025/rogii-h-blend-v1) | 174 | 多方案融合。 |
| 16 | [EDA Target-free Alignment](https://www.kaggle.com/code/pilkwang/rogii-eda-target-free-alignment-for-tvt) | 165 | target-free 对齐诊断。 |
| 17 | [LB 7.156 Baseline](https://www.kaggle.com/code/yusuketogashi/rogii-lb7156-baseline) | 163 | 公榜 baseline。 |
| 18 | [SP45 Blend v2](https://www.kaggle.com/code/jaemin3404/rogii-sp45-fleongg-blend-v2) | 159 | SP45 系融合。 |
| 19 | [Working Note: Target-free TVT](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering) | 159 | 对齐路线实验记录。 |
| 20 | [LightGBM](https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-lightgbm) | 157 | 树模型分支。 |

另外两个近期强公榜标题值得**审计但不宜直接继承**：

- [Verified Public 6.979 + Visuals](https://www.kaggle.com/code/prvsiyan/rogii-verified-public-6-979-visuals)：源码可见 canary/probe、全局约 `-0.40` bias 和重叠样例逻辑，属于公榜过拟合风险案例。
- [PF Config Strategy 6.491](https://www.kaggle.com/code/helloromiro/pf-config-strategy-high-scoring-6-491)：contact-gated 对齐、ridge/PF anchor、`U` 投影和多个激进度 profile；标题分数强，但依赖公开模型资产，需自行复现 OOF。

## 5. 可落地的建模方案

### A. 干净基线与验证骨架

1. 以井为 group 做 5-fold；在每口验证井上随机/分层选多个 cutoff，把 cutoff 后的 `TVT_input` 置空，模拟真实 suffix 预测。
2. 指标必须按所有验证行合并后的 pooled RMSE 计算，同时记录：
   - 每井 RMSE、中位数、P90/P95、最差井；
   - 误差随预测距离增长的曲线；
   - cutoff、井长、Z span、GR 质量、typewell cluster 分层。
3. 基线预测目标用 `ΔTVT = TVT - last_known_TVT`，而不是直接预测绝对 TVT；carry-forward 约 16 量级可作为 sanity check，公开 XGB 骨架约 CV 15。

### B. 主模型

1. **候选轨迹生成**：PF/HMM/beam + 有限窗口、方向约束的 DTW/NCC，将 horizontal GR 与 typewell GR、预测起点前自身 GR 对齐；保留 top-k 路径及 posterior spread，不只输出 argmax。
2. **学习残差**：
   - 强且易复现：多 seed LGBM/CatBoost + 正则化线性/Ridge OOF 融合；
   - 并行试验：GRU/TCN/1D-CNN/Transformer 序列模型，但必须与树模型使用完全相同的 masked-suffix folds。
3. **特征**：GR 多尺度窗口/梯度/自相关、MD/X/Y/Z 几何、切线方向、距锚点距离、typewell cross-correlation、PF/beam 候选、候选间分歧和不确定性。
4. **地质正则**：对 `U = TVT + Z` 随归一化 MD 做鲁棒低阶/四次投影；预测起点后使用 warm-up damping，末端可做轻量 Savitzky–Golay 平滑。所有系数只由 OOF 决定。
5. **双轨融合**：一条偏物理 selector，一条偏学习 residual/sequence；公开强方案常见约 0.55/0.45，但本项目应重新以 OOF 学权重。

### C. 泛化与灾难井控制

- 对每口测试井在可见前缀内构造 0.50/0.65/0.75 等伪 cutoff，比较候选的中位 RMSE + 稳定性惩罚；只做**有上限、渐进式**的权重调整，不硬切到单一路线。
- 学习局部 GR gain/offset 校准；对可能重复的 typewell 建 cluster ID，让验证折不共享同一参考来源。
- PF spread、模型分歧、GR 对齐峰值比、前缀回测误差可组成 trust score；低置信度时收缩到平滑几何/anchor fallback。
- 增加漂移检测、回溯/重置候选、斜率和曲率软约束、有限幅 clipping；优化目标可在 pooled RMSE 之外加入 worst-well/P95 正则。
- 邻井/formation surface 插值只作为可选分支，并使用空间距离、同 frame/方向和验证可信度门控；社区结果表明无条件远距离借用可能显著变坏。

## 6. 验证与泄漏红线

| 风险 | 为什么危险 | 控制方式 |
|---|---|---|
| 随机行切分 | 同一井前后行高度相关，CV 会虚高 | 至少 GroupKFold by well；序列模型整井隔离 |
| typewell 重复 | 井分开但同一参考曲线可能跨折 | 对 typewell 内容哈希/聚类，做 cluster-group holdout |
| 邻井/空间特征 | 普通按井分折仍共享 field 信息 | 额外做 spatial-block / whole-field holdout |
| 可见 `test/` 是训练示例 | same-ID、formation contact copy 可在公榜有效但隐藏重跑失效 | 禁止 ID 查表、静态行号、训练 TVT/contact 覆盖；只保留 test-available 特征 |
| 公开模型/CSV 资产 | 可能含未知 folds、probe、泄漏或 test 特化 | 资产逐项审计；核心模型从原始训练数据自行重训并生成 OOF |
| LB probe / global bias / canary | 密集公榜可被微小调参“拟合” | 在冻结 CV 前不根据 LB 改偏置；最终只比较少量预注册候选 |
| `X/Y/Z` 的因果后门 | 钻井轨迹可能是地质解释后的结果，虽合法但分布依赖 | 做去除/扰动消融和 field holdout，不作为唯一信号 |
| formation surfaces | 训练有、隐藏测试无 | 不能直接作测试特征；只能训练可从测试可见输入推断的先验 |
| 非确定 PF/GPU | 单次 CV/LB 波动可误导 | 固定 seed、至少 5 seeds、重复推理一致性检查 |
| pooled RMSE | 长井和单口灾难井权重很大 | 同时看 pooled 与逐井尾部指标；专门控制累计漂移 |
| 隐藏重跑约束 | 本地成功不代表 Kaggle 9h/离线成功 | 在无网络、接近正式硬件/内存下全量 dry-run；严格校验 ID 顺序、行数、有限值 |

## 7. 公榜与排名信号

Kaggle CLI 公开榜快照：

| 公榜名次 | RMSE |
|---:|---:|
| 1 | 4.859 |
| 10 | 5.511 |
| 20 | 5.940 |
| 50 | 6.309 |
| 100 | 6.435 |
| 276（约前 5%） | 6.515 |
| 552（约前 10%） | 6.644 |
| 600 | 6.661 |

- 名次 100 到 552 只差约 **0.209 RMSE**，公榜极其拥挤；0.02–0.05 的噪声或 bias 都会造成大量名次变化。
- 标题自报 6.491 在该快照约位于 200 名附近，但这只说明当前公榜位置；最终奖牌由截止时有效队伍数、规则和私榜决定，不能据此承诺。
- 更可靠的阶段目标应是：先得到无可见测试泄漏、跨 seed 稳定的 grouped CV；再把公榜作为方向性校验，而不是超参数优化器。

## 8. 推荐执行优先级

1. 复现干净 XGB/GBDT residual baseline 与 masked-suffix GroupKFold，锁定评估脚本。
2. 复现 ridge/PF/beam + `U=TVT+Z` 投影主干，去掉 same-ID、contact overwrite、probe 和全局 LB bias。
3. 建立多 seed OOF 候选库，做受约束的正权重融合。
4. 加入逐井前缀伪回测、trust score、warm-up 和灾难井 fallback。
5. 并行训练一个真正按序列建模的 GRU/TCN 分支；只有在严格 folds 上提供独立增益才进入融合。
6. 最后再测试空间邻井和 typewell cluster 特征，并以 spatial/field holdout 为准入条件。
7. 预留至少一天做 9 小时离线重跑、提交格式和非确定性审计；不把最后一天用于新架构。

这套路线比直接复制高票 Notebook 更慢一点起步，但能最大限度避免“公榜很好、隐藏重跑失效”的典型陷阱，同时保留公开强方案中真正可泛化的 PF/beam、residual、`U` 投影、前缀自检和双轨融合思想。
