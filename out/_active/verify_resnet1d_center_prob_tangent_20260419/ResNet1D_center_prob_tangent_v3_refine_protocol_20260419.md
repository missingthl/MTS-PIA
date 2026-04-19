# ResNet1D + DLCR `center_prob_tangent_v3` 修复清单（V3 Refine Protocol）

更新时间：2026-04-19

## 1. 当前定位

`center_prob_tangent_v3` 已经完成：

- `Phase 0`
- `Phase 0.5`
- `Phase 1`

并且已经证明：

- 新线不是局部成功后在 21 数据集上整体崩盘的失败分支
- `LW + MDL` 这条统一数学路线是成立的
- 新线已经进入“全局可接受”状态

但这版还没有收圆成统一默认主线。  
当前最需要修的，不是主框架，而是两处深层数值物理不自洽：

1. `PPCA posterior` 塌缩
2. `MDL` 的 `k=0` 校准还不干净

## 2. Bug A：PPCA 后验塌缩

### 2.1 当前观测到的真实现象

从 [center_prob_tangent_fullscale_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_prob_tangent_20260419/center_prob_tangent_fullscale_table.csv) 中抽取 `posterior_confidence_mean` 后，当前分布是：

- 数据集数：`21`
- 全局平均：`≈ 2.24e-12`
- 中位数：`≈ 2.95e-20`
- 最大值：`≈ 4.36e-11`（`Heartbeat`）
- 最小值：`0.0`（如 `SelfRegulationSCP1 / BasicMotions / ArticularyWordRecognition / JapaneseVowels / Cricket`）

阈值统计：

- `21 / 21` 个数据集都 `<= 1e-9`
- `19 / 21` 个数据集都 `<= 1e-12`
- 没有任何数据集的 `posterior_confidence_mean > 1e-6`

对应地：

- 全局平均 `posterior_confidence_far_decay_mean ≈ 0.9999999999977611`

这说明：

**当前 `PPCA posterior` 不是“方向不够理想”，而是几乎已经完全数值塌缩。**

### 2.2 当前最合理的物理解释

当前后验实现位于 [local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py) 的 `_compute_prob_tangent_posterior`。

当前核心形式是：

- `residual_energy = ||residual||^2`
- `confidence = exp(- residual_energy / (2 * sigma2 * residual_dim))`

这在高维特征空间下有一个典型问题：

- `residual_energy` 的量级会随特征维度增长
- 但 `sigma2` 当前来自被舍弃特征值均值，量级偏局部、偏小
- 结果是指数项迅速下溢到 `0`

也就是说：

**当前不是“Query 不属于该切面”，而是“后验公式在当前维度尺度下失去分辨率”。**

### 2.3 第一优先级修复方向

当前不建议先换掉整套 `PPCA`，而是先做维度与尺度归一化修复。

优先修复顺序：

1. 把 `residual_energy` 改成**按特征维度归一化**的版本  
   例如优先测试：
   - `residual_energy / feature_dim`
   - `residual_energy / max(1, residual_dim)`

2. 把 `sigma2` 改成**可与高维残差同量纲比较**的有效厚度  
   例如：
   - `sigma2_eff = sigma2 * feature_dim`
   - 或 `sigma2_eff = sigma2 * residual_dim`

3. 若高斯指数后验仍持续下溢，则切换为**重尾权重**  
   备用优先方案：
   - Student-t
   - Rational quadratic / Cauchy 型权重

### 2.4 必做执行动作 A

下一轮代码修改前，必须先补齐以下诊断量：

- `posterior_residual_energy_mean`
- `posterior_residual_energy_per_dim_mean`
- `posterior_sigma2_eff_mean`
- `posterior_log_confidence_mean`
- `posterior_confidence_quantiles`

目标不是再看一遍 `confidence`，而是直接确认：

- 是 `residual_energy` 太大
- 还是 `sigma2` 太小
- 还是两者的量纲本来就没对齐

## 3. Bug B：MDL 的 `k=0` 校准还不干净

### 3.1 当前观测到的真实现象

全量结果里：

- 全局平均 `mean_selected_rank ≈ 2.2603`
- 全局平均 `k0_fallback_rate ≈ 0.4349`

这说明：

- `MDL` 不是没工作
- 自动定秩也没有塌成常数秩

但当前问题是：

**`k=0` 的触发还没有形成干净的结构选择性。**

高触发数据集包括：

- `SelfRegulationSCP1 = 1.0000`
- `Pendigits ≈ 0.8854`
- `HAR ≈ 0.8185`
- `Cricket ≈ 0.6667`
- `MotorImagery ≈ 0.6600`

低触发数据集包括：

- `FingerMovements = 0.0000`
- `Heartbeat = 0.0000`
- `SelfRegulationSCP2 = 0.0000`

这组结果说明：

- 当前 `MDL` 已经带来了真实的离散选择性
- 但它并没有只在“最不可信局部结构”上退化

### 3.2 当前最合理的物理解释

当前 `MDL` 位于 [local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py) 的 `_select_prob_tangent_rank`。

当前惩罚结构本质上仍是经典大样本 model-order penalty：

- 惩罚项核心依赖 `log(N)`
- 其中当前 `sample_count = deviations.shape[0]`
- 对这条新线来说，`N` 实际上只是每类的 `4` 个子原型偏移

于是当前系统里：

- `N = 4`
- `log(4) ≈ 1.386`

这意味着经典渐近理论里的惩罚尺度，在这里很可能是错位的。

也就是说：

**当前问题不是“MDL 不该用”，而是“MDL 的惩罚强度还没有针对 `N=4` 这个极小样本局部谱系统做全局校准”。**

### 3.3 第一优先级修复方向

当前最稳的修复不是 per-dataset 调参，而是引入**全局统一校准系数**：

- `mdl_penalty_beta`

让惩罚项从：

- `base_penalty`

改成：

- `beta * base_penalty`

这里的 `beta` 不是新的主矩阵超参，而是：

- 面向 `N=4` 局部谱系统的一次性全局数值校准

### 3.4 必做执行动作 B

下一轮必须做一次 `MDL` 惩罚曲线回放。

首选回放数据集：

- `FingerMovements`

原因：

- 当前 `mean_selected_rank = 4.0`
- `k0_fallback_rate = 0.0`
- 同时它又是新线最明显失败点之一：
  - `center_only = 0.58`
  - `best center_subproto = 0.57`
  - `center_prob_tangent_v3 = 0.46`

这类现象最像：

**该退化时没有退化，结果保留了过强的刚性局部结构。**

备选回放数据集：

- `Heartbeat`

它同样呈现：

- `mean_selected_rank = 4.0`
- `k0_fallback_rate = 0.0`
- 而且新线表现明显弱于 `center_only`

回放时必须逐 rank 打出：

- `rank`
- `sigma2`
- `selected_logdet`
- `neg_log_likelihood`
- `bic_score`
- `mdl_score`
- `penalty_term`
- `likelihood_term`
- `beta_scaled_penalty`

这样我们才能直接回答：

- 当前是似然项压过惩罚项
- 还是惩罚项本身在 `N=4` 口径下就太弱

## 4. 下一阶段修复协议

### 4.1 第一组改动：Posterior 数值修复

目标：

- 先把 `PPCA posterior` 从“全体下溢”修到“至少有可用动态范围”

建议顺序：

1. 增加 `log_confidence` 与 `residual_energy_per_dim` 诊断落盘
2. 测试 `gaussian + dimension normalization`
3. 若仍塌缩，切到 `student_t posterior`

首轮只做代表性四数据集：

- `NATOPS`
- `FingerMovements`
- `SelfRegulationSCP1`
- `Epilepsy`

判据：

- `posterior_confidence_mean` 不再全部落在 `1e-9` 以下
- `NATOPS / Epilepsy` 的置信度均值应系统性高于 `SCP1 / FM`

### 4.2 第二组改动：MDL 全局校准

目标：

- 让 `k=0` 更像有选择地保命，而不是随机地高频触发或完全不触发

建议顺序：

1. 在 `_select_prob_tangent_rank` 中加入 `mdl_penalty_beta`
2. 默认保持 `beta = 1.0`
3. 用 `FingerMovements` 的惩罚曲线回放决定全局候选 `beta`
4. 再在四数据集短程矩阵中验证：
   - `beta_low`
   - `beta_mid`
   - `beta_high`

这里的目标不是扫很多值，而是尽快找出：

- 当前惩罚是偏弱
- 还是偏强

### 4.3 第三组改动：重新组织 `v3`

当上面两组都完成后，再进入：

- `center_prob_tangent_v3_refined`

此时它的定义应该变成：

- `LW`
- `MDL(beta-calibrated)`
- `PPCA posterior (dimension-normalized or heavy-tailed)`
- `pinv`

到那时，`v3 refined` 才是新线真正有资格冲击统一主线的版本。

## 5. 当前总判断

现在最重要的判断不是“新线值不值得继续”，这轮全量已经回答了：

- 值得继续

现在真正的问题是：

1. `PPCA posterior` 还没有形成有效概率边界  
2. `MDL` 的 `k=0` 还没有被校准成可信的结构回退  

一句话说：

**当前 `center_prob_tangent_v3` 的主框架是对的，卡住它的不是理论路线，而是两个可以被精确修掉的数值层 Bug。**
