# PIA Operator P0a.1 Stage C Next Promote

## 阅读提示

这份文档当前保留为：

- `same-only + 修 Lambda` 这条线的阶段性收束记录
- 它解释了为什么“只修权重矩阵”不足以把模板方向稳定掰正

它现在不是默认主线入口。

当前默认入口请先看：

- [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md)

## 任务名称
`SCP-Branch P0a.1-C_next: Lambda Geometry Repair Probe`

## 零、战略定位
`P0a.1` 的阶段 A 已经把单模板 PIA 的响应器修到了一个更健康、也更接近可工作的状态：

- `A2r = local median-centering + frozen IQR scale`
- 在 `NATOPS + dense_dynamic_gru` 上：
  - `test_macro_f1 = 0.8231`
  - `response_vs_margin_correlation = +0.1770`
  - `activation_coverage_ratio = 0.0002`
  - 与 `no_shaping baseline = 0.8287` 的差距已明显缩小

这说明：

> 当前单模板 PIA 的主问题已经不再首先是“响应域失控”，而更像是“模板方向仍未被学成局部主判别轴”。

阶段 C 的第一版也已经给出了一个非常关键的负结果：

- `A2r + soft_weighted_fit(mean_dist)` 在 `NATOPS` 上掉到 `0.7891`
- `template_mean_direction_cosine` 只从 `0.0111` 升到 `0.0232`
- `effective_sample_size = 191.85 / 192`

这说明：

> 第一版权重核并没有真正让样本度量矩阵 `Λ` 长出几何分层能力；它在数值上仍近似 `cI`，因此 weighted fit 和 unweighted fit 在几何上几乎没有本质差别。

所以 `C_next` 的任务不是：

- 回退广义逆/闭式解
- 提前上 slow refresh
- 提前放开 `r`

而是：

> 在保留 `A2r` 响应器和 `TELM2` 广义逆主线不变的前提下，继续修阶段 C 的样本度量矩阵 `Λ`，让核心样本对模板方向真正形成主导。

一句话说：

> `C_next` 不是换求解路线，而是继续修 `Λ` 的几何分层能力。

---

## 一、唯一核心问题
**在固定 `A2r` 前端、固定参考几何、固定 `r=1`、固定广义逆闭式解的前提下，能否仅通过修正样本度量矩阵 `Λ`，把单模板 PIA 的模板方向从“近乎无判别意义的弱轴”推向“prototype-local 的局部主判别轴”。**

这一步只问：

- 当前阶段 C 的失败是否主要来自 `Λ` 太平
- 更陡的、仍然分布驱动的 `Λ` 是否能改善模板方向
- 这种方向改善是否能转成更接近 `baseline` 的终端收益

当前不问：

- `r=1` 是否已经足够作为最终系统容量
- `median-min` 是否就是最终权重内核
- slow refresh 是否已经应该进入
- 阶段 B 的连续几何力场是否已经需要并入

---

## 二、当前阶段的数学定位
阶段 C 当前修的不是响应器，而是：

\[
W=(P^T\Lambda P+\lambda I)^{-1}P^T\Lambda Y
\]

中的样本度量矩阵 `\Lambda`。

当前第一版失败，不是因为：

- 广义逆/闭式解路线有问题
- `TELM2` 本体必须被替换

而是因为：

\[
\Lambda \approx cI
\]

导致：

- weighted fit 与 unweighted fit 几乎同构
- 模板方向仍被 admitted 样本云的主导分布牵引
- 核心样本和边界样本没有被真正区分开

因此，`C_next` 的唯一目标是：

> 在不改变闭式解主线的前提下，让 `Λ` 不再近似常数矩阵。

---

## 三、必须守住的边界

### 1. 不改广义逆主线
这一轮仍必须保留：

- `TELM2` 的闭式/正规方程/广义逆式求解
- train-only operator fit
- freeze 后作用于 `train/val/test`

不允许改成：

- 反向传播
- 多轮优化器
- 端到端梯度训练

### 2. `r=1` 继续冻结，但不作过度宣称
这一轮继续固定：

- `single-template`
- `r_dimension = 1`

但这一步的含义只限于：

> `r=1` 足够作为当前“PIA 本体验证”和“计算链修复”的最小载体。

不允许在本轮主结论里写成：

- `r=1` 已经证明足够解决最终问题

### 3. `median-min` 只是候选核，不是最终内核
这一轮可以测试更陡的分布驱动核，但必须明确：

- 它是 `C_next distribution-driven steepening candidate`
- 不是已经被宣布的最终阶段 C 内核

### 4. 不动阶段 A，不提前混阶段 B
这一轮固定：

- 前端固定为 `A2r`
- 不再改 clipping / local median / frozen scale
- 不并入 continuous geometric force field
- 不做 slow refresh

---

## 四、当前我们在整体框架里的位置
当前大系统的位置可以写成：

1. `P0` 已证明：
   - 单模板 PIA 不是完全无效
   - 但还没站住相对 `mean-centered` 的独立必要性

2. `P0a.1` 阶段 A 已证明：
   - 响应器可以被修到“数值健康且部分可工作”

3. 现在进入 `P0a.1` 的 `C_next`
   - 目标不是修“油门/刹车”
   - 而是修“方向盘背后的样本度量矩阵”

4. 只有 `C_next` 至少把模板方向显著掰正，才值得进入阶段 B
   - 即 continuous geometric force field

---

## 五、`C_next` 的主比较对象
这一轮主场仍只做：

- dataset: `natops`
- terminal: `dense_dynamic_gru`
- seed: `1`

并且只比较三条线：

### `C0`
`A2r + current_unweighted_fit`

作用：
- 给出当前最佳 Stage A 前端下的无权模板方向参考

### `C1`
`A2r + current_mean_dist_weighted_fit`

作用：
- 作为当前第一版 Stage C 的失败对照
- 验证“权重太平”这一判断是否成立

### `C2`
`A2r + median_min_weighted_fit`

作用：
- 作为下一版权重核候选
- 检查更陡的、仍分布驱动的 `Λ` 是否能真正产生几何分层

当前不进入：

- 阶段 B
- slow refresh
- `Arm C`
- multi-template

---

## 六、`C_next` 的唯一实现对象
这一步只改：

- `prototype-local` 样本权重定义
- 以及由该权重诱导的 `\Lambda`

不改：

- admitted 集合的上游定义
- `A2r` 响应器
- operator budget matching
- `epsilon`
- 平滑系数
- 阶段 B 力场

---

## 七、三条线的精确定义

### `C0`: `A2r + unweighted`
\[
\Lambda = I
\]

这条线是当前无权闭式解参考。

### `C1`: `A2r + mean_dist_weighted`
对每个拟合样本 `z_i`：

- 在其所属 `(class_id, prototype_id)` 的 admitted 统计域内
- 记 `d_i = ||z_i - p_same(i)||_2`
- 记 `\bar d_{proto} = mean(d)`

则：

\[
w_i=\exp\left(-\frac{d_i}{\bar d_{proto}+1e-8}\right)
\]

这条线的定位：

- 不是候选最终解
- 而是当前 Stage C 第一版的失败对照

### `C2`: `A2r + median_min_weighted`
对每个拟合样本 `z_i`：

- 在其所属 `(class_id, prototype_id)` 的 admitted 统计域内
- 记：
  - `d_i = ||z_i - p_same(i)||_2`
  - `d_min = min(d)`
  - `d_med = median(d)`
  - `s_proto = max(d_med - d_min, 1e-8)`

定义权重：

\[
w_i=\exp\left(-\frac{d_i-d_{min}}{s_{proto}}\right)
\]

解释：

- 最内核样本权重为 `1`
- 中位位置样本权重约为 `exp(-1)`
- 更外层样本权重自然衰减
- 不引入显式温度常数 `\tau`

极小样本簇保护：

- 若某个 `prototype-local` 拟合池的样本数 `N <= 3`
- 则该 prototype 直接返回全 `1` 权重
- 即在该局部簇上退化回 `unweighted`

原因：

> 在拟合样本极度稀缺时，分布驱动的几何分层没有稳定统计意义；此时用 OLS/无权闭式解作为保底更稳。

但必须明确：

> `C2` 只是下一版 Stage C 的候选核，不预设为最终内核。

---

## 八、实验执行流程

### Step 0：冻结当前已知最优前端
固定：

- `A2r = sigmoid_clip_tanh_local_median_scaled_iqr`

不再重跑整套阶段 A 扫描。

### Step 1：主场只跑 `NATOPS seed=1`
固定：

- dataset: `natops`
- terminal: `dense_dynamic_gru`
- seed: `1`

### Step 2：只比较 `C0 / C1 / C2`
必须同时产出：

- `C0 = A2r + unweighted`
- `C1 = A2r + mean_dist_weighted`
- `C2 = A2r + median_min_weighted`

### Step 3：只在 `C2` 不退化时，才考虑进入阶段 B
只有当 `C2` 至少达到本页定义的“中等成立”，阶段 B 才允许进入实现队列。

其中“不退化”至少指：

- `C2` 的 `effective_sample_ratio` 明显低于 `C1`
- 但 `min_proto_effective_sample_size` 没有塌到 `1~2`
- `template_mean_direction_cosine` 明显高于 `C0/C1`
- `test_macro_f1` 不显著差于 `A2r`

若不满足：

- 本轮停在 `C_next`
- 不进入阶段 B

---

## 九、必须新增或强化的诊断指标

### 已有并继续保留
- `template_mean_direction_cosine`
- `response_vs_margin_correlation`
- `activation_coverage_ratio`
- `margin_gain_per_unit_distortion`
- `effective_sample_size`
- `effective_sample_ratio`
- `fit_anchor_margin_mean`
- `fit_anchor_same_dist_mean`

### `C_next` 必须新增
- `min_proto_effective_sample_size`
  - 所有 prototype-local 拟合池中最小的 `N_eff`
  - 用于检查是否已经退化成伪硬截断

- `median_proto_effective_sample_size`
  - 看整体 prototype-local 权重是否形成稳定分层，而不是只在个别 prototype 上起作用

- `proto_weight_scale_min`
  - 所有 prototype-local `s_proto` 的最小值
  - 用于审计是否存在局部簇尺度极小、从而把 soft weight 推成伪硬截断的风险

- `weight_kernel_name`
  - 明确区分：
    - `identity`
    - `exp_distance_over_mean`
    - `exp_relative_distance_over_median_minus_min`

- `proto_weight_scale_mean`
  - 记录各 prototype 的 `s_proto`
  - 用于审计 `median-min` 是否在某些 prototype 上过小

指标命名收口：

- 当前阶段 C 的主方向指标只保留：
  - `template_mean_direction_cosine`
- 其定义明确为：
  - 学得的 `u_template` 与各 `fit sample` 的 `u_geom,i = normalize(p_same(i)-p_opp(i))` 的平均 cosine
- 若实现层为了兼容已有 CSV 仍保留 `geometry_alignment_cosine_mean`
  - 则它在当前阶段 C 中应视为 `template_mean_direction_cosine` 的同义别名
  - 不允许再实现出第二套近义但定义不同的方向指标

---

## 十、必须输出的文件
- `pia_operator_p0a1_cnext_config_table.csv`
- `pia_operator_p0a1_cnext_per_seed.csv`
- `pia_operator_p0a1_cnext_structure_diagnostics.csv`
- `pia_operator_p0a1_cnext_score_diagnostics.csv`
- `pia_operator_p0a1_cnext_response_diagnostics.csv`
- `pia_operator_p0a1_cnext_anchor_diagnostics.csv`
- `pia_operator_p0a1_cnext_conclusion.md`

---

## 十一、必须回答的问题
1. 阶段 C 第一版失败，是否确实主要来自：
   - `\Lambda` 太平
   - 而不是 WLS 路线本身无效

2. `median_min_weighted` 是否比 `mean_dist_weighted` 更能压低：
   - `effective_sample_ratio`
   - 同时不把 `min_proto_effective_sample_size` 压塌

3. `C2` 是否能显著提升：
   - `template_mean_direction_cosine`

4. 方向改善后，`C2` 是否至少不再明显伤害：
   - `test_macro_f1`
   - `response_vs_margin_correlation`

5. 若 `C2` 成立，下一步更像：
   - 进入阶段 B 连续几何力场
   - 而不是回退阶段 C

---

## 十二、成功标准

### 弱成立
- `C2` 相比 `C1`
  - `effective_sample_ratio` 明显下降
  - `template_mean_direction_cosine` 明显上升

### 中等成立
- `C2` 相比 `C0`
  - `template_mean_direction_cosine` 明显提升
  - `test_macro_f1` 不明显退化
  - `min_proto_effective_sample_size` 未塌到极低水平
  - `proto_weight_scale_min` 未显示明显局部尺度塌缩

### 强成立
同时满足：
- `C2` 在方向指标上显著优于 `C0/C1`
- `C2` 在终端分数上不低于当前 `A2r`
- `C2` 仍保持：
  - `response_vs_margin_correlation >= 0`
  - `activation_coverage_ratio` 处于健康低值
- 可以合理进入阶段 B

---

## 十三、若结果不成立，如何解释

### 情形 A
`C2` 的 `effective_sample_ratio` 仍接近 `1`

说明：
- `median-min` 仍不够陡
- 当前阶段 C 的核心问题仍是 `Λ` 分层不足

### 情形 B
`C2` 的 `min_proto_effective_sample_size` 过低

说明：
- 当前候选核过陡
- 已经从 soft-weighted 退化成伪硬截断

若同时伴随：

- `proto_weight_scale_min` 极低

则更进一步说明：

- 问题主要来自某些极紧 prototype-local admitted 集合上的局部尺度塌缩

### 情形 C
`C2` 的方向指标升了，但分数仍明显差

说明：
- 模板方向改善尚未转成终端可读收益
- 下一步是否进入阶段 B 需要更谨慎

### 情形 D
`C0 / C1 / C2` 都站不住

说明：
- 当前阶段 C 仍未把 `\Lambda` 修成有效几何分层器
- 但这并不自动证明：
  - 广义逆路线无效
  - `r=1` 已被最终否定

更稳的结论是：
- 当前单模板 Stage C 权重几何仍未立住

---

## 十四、一句话执行目标
**在保留 `A2r` 前端、`r=1`、广义逆闭式解和当前系统边界不变的前提下，只把阶段 C 的样本度量矩阵 `\Lambda` 从“近似常数”修成“prototype-local、分布驱动、真正具有几何分层能力的权重矩阵”，并验证这种分层是否能把模板方向推向局部主判别轴。**
