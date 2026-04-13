# PIA Operator P0a.1 C3 Discriminative Closed-Form Probe Promote

## 阅读提示

这份文档是当前 PIA 主线里最接近现实现链的 promote：

- `A2r`
- `C3` 判别闭式解
- 与 `same-only` 路线的可归因对照

如果只想先看“现在框架总体做到了哪一步”，请先看：

- [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md)

## 任务名称
`SCP-Branch P0a.1-C3: Hetero-Associative Discriminative Closed-Form Probe`

## 零、战略定位
`P0a.1` 的阶段 A 已经证明：

- 单模板 PIA 的响应器可以被修到数值健康且部分可工作
- `A2r = local median-centering + frozen IQR scale`
- 在 `NATOPS + dense_dynamic_gru` 上：
  - `test_macro_f1 = 0.8231`
  - `response_vs_margin_correlation = +0.1770`
  - 与 `no_shaping baseline = 0.8287` 的差距已明显缩小

`P0a.1` 的阶段 C 和 `C_next` 进一步证明：

- 当前同侧 `same-only` 样本度量矩阵 `Lambda` 确实可以从“近似常数”被修成有分层的权重矩阵
- 但即便 `Lambda` 已经不再近似 `cI`
- 模板方向仍没有被稳定推成局部主判别轴

这暴露出一个更深层的计算问题：

> 当前 `TELM2` 在这条调用链里拟合的仍是自联想/重构型目标，而不是显式判别目标。

因此，这一步不是继续把 `C_next` 往下补，而是单独开一个新 probe：

> 在保留 `A2r`、`r=1` 和广义逆闭式解主线不变的前提下，把模板拟合从 auto-associative 改成 hetero-associative discriminative closed-form，检查单模板算子是否因此真正学到局部判别法向。

一句话说：

> `C3` 不是“纯 Stage C 的补丁”，而是一个新的判别目标闭式解探针。

---

## 一、唯一核心问题
**在固定 `A2r` 前端、固定 `r=1`、固定广义逆闭式解、固定参考几何的前提下，若把模板拟合目标从自联想重构改成双极判别目标，单模板 PIA 是否能从 same/opp 对比样本中学出更接近局部主判别轴的方向。**

这一步只问：

- 当前单模板 PIA 的更深瓶颈是否来自拟合目标错位
- 同样的闭式解路线，在判别目标下是否会显著改善模板方向
- 这种方向改善是否能转成终端可读收益

当前不问：

- `r=1` 是否已经足够作为最终容量
- `C3` 是否就是最终框架形态
- slow refresh 是否应该并入
- 阶段 B 连续几何力场是否已经需要并入

---

## 二、与当前工程的关系
这一步必须明确区分于 `C_next`：

- `C_next` 修的是样本度量矩阵 `Lambda`
- `C3` 修的是拟合目标 `Y` 和样本池构成

因此：

> 若 `C3` 成功，它不能被表述成“单靠修 `Lambda` 就成功了”。

它最多说明：

- 目标函数需要判别化
- 样本池需要 bipolar 化
- 广义逆闭式解在判别目标下有潜力学出法向

这不是坏事，但归因必须诚实。

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

> `r=1` 仍作为当前“PIA 本体验证”和“判别目标 probe”的最小载体。

不允许在本轮主结论里写成：

- `r=1` 已证明足够解决最终容量问题

### 3. 不改阶段 A，不混阶段 B
这一轮固定：

- 前端固定为 `A2r`
- 不再改 clipping / local median / frozen scale
- 不并入 continuous geometric force field
- 不做 slow refresh

### 4. `C3` 作为新 probe，不能覆盖 `C_next`
这一轮不回写“`C_next` 已被 `C3` 取代”。

更准确的口径是：

- `C_next` 继续代表“same-only + 修 `Lambda`”
- `C3` 代表“bipolar discriminative closed-form”

---

## 四、当前我们在整体框架里的位置
当前大系统位置可以写成：

1. `P0` 已证明：
   - 单模板 PIA 不是完全无效
   - 但还没站住独立必要性

2. `P0a.1` 阶段 A 已证明：
   - 响应器可以被修到“数值健康且部分可工作”

3. `P0a.1` 阶段 C / `C_next` 已证明：
   - `Lambda` 平不平是问题
   - 但只修 same-only 的 `Lambda` 还不够

4. 现在进入 `P0a.1-C3`
   - 目标不是继续修响应器
   - 也不是继续只修 same-only 的 `Lambda`
   - 而是测试：判别目标重写是否是决定性变量

---

## 五、数学定位
当前 `TELM2` 在这条链路里，本质上是：

\[
W=(P^TP+\lambda I)^{-1}P^TY
\]

但当前 `Y` 不是外部判别目标，而是来自输入坐标的反激活重构目标。

`C3` 的唯一新动作是：

> 保持广义逆闭式解形式不变，只把 `Y` 从 auto-associative 重构目标改成 hetero-associative discriminative target，并把拟合样本池从 same-only 改成 same/opp bipolar pool。

也就是从：

- “最好地拟合输入形状”

转向：

- “最好地把 same 侧和 opp 侧沿一维轴拉开”

---

## 六、主比较对象
主场仍只做：

- dataset: `natops`
- terminal: `dense_dynamic_gru`
- seed: `1`

并且只比较三条线：

### `C0`
`A2r + unweighted same-only`

作用：

- 当前最优 Stage A 前端下的无权 self-template 参考

### `C2`
`A2r + weighted same-only`

作用：

- 当前 `C_next` 的 strongest same-only 参考
- 用于回答“只修 `Lambda` 到哪一步”

说明：

- 第一版默认取当前已实现的最强 same-only 线
- 若当前结果显示 `C0 > C2`
- 则报告中仍必须同时列出二者

### `C3`
`A2r + bipolar discriminative weighted`

作用：

- 检查“目标函数判别化 + bipolar pooling + 加权闭式解”是否能真正把模板方向推向法向

当前不进入：

- 阶段 B
- slow refresh
- `Arm C`
- multi-template

---

## 七、`C3` 的唯一实现对象
这一步只改三件事：

- 拟合目标 `Y`
- 拟合样本池构成
- 由 bipolar pool 诱导的权重矩阵 `Lambda`

不改：

- `A2r` 响应器
- `r=1`
- 广义逆求解路线
- operator budget matching
- `epsilon`
- 平滑系数
- 阶段 B 力场

---

## 八、三条线的精确定义

### `C0`: `A2r + unweighted same-only`
- 样本池：当前 same-only admitted fit windows
- 目标：当前 `TELM2` 默认 auto-associative target
- 权重：`Lambda = I`

### `C2`: `A2r + weighted same-only`
- 样本池：当前 same-only admitted fit windows
- 目标：当前 `TELM2` 默认 auto-associative target
- 权重：沿用当前 strongest same-only weighted 版本

### `C3`: `A2r + bipolar discriminative weighted`

#### 1. Bipolar pooling
对每个 `(class_id, prototype_id)` 的 same prototype：

- 取其 same-side admitted fit windows
- 再为该 prototype 指定一个固定的 opposite prototype
- 从该 opposite prototype 对应的 admitted windows 中取 opposite pool
- 在样本维度上拼成一个 bipolar fit pool

#### 2. Pairing rule 必须写死
为避免归因污染，本轮固定：

> 每个 same prototype 对应一个**固定的最近 opposite prototype**。

定义：

- 在 frozen geometry 上
- 用 same prototype center 到所有 opposite prototype centers 的欧氏距离
- 选最近的一个 opposite prototype

本轮不允许：

- 每个窗口动态找最近 opp prototype
- 全局拼所有 opp admitted windows
- 按 test-time routing 改写 opposite 选择

#### 3. Pool mass handling
本轮不做硬性的 same/opp 样本数截断到完全相等。

第一版默认：

- 两侧样本都保留
- 通过权重归一化控制两侧总质量

要求：

- summary 中必须输出 same/opp 两侧原始样本数
- summary 中必须输出 same/opp 两侧权重总和
- conclusion 中必须显式审计：
  - same/opp 是否存在系统性样本数失衡
  - same/opp 是否存在系统性权重总质量失衡
- 若存在明显失衡，则必须在主结论里说明：
  - 当前结果同时受目标函数与 pool 结构共同影响
  - 不允许把增益简单归因为“目标改对了”

#### 4. Discriminative target override
不再使用 `TELM2` 内部自动构造的重构目标。

而是在外部显式构造一维判别目标 `Y_disc`：

- same-side 样本目标为正
- opp-side 样本目标为负

为了兼容当前激活反函数与数值稳定，第一版写死：

- 若使用 `sigmoid` 反域：
  - same target = `0.95`
  - opp target = `0.05`
- 再由 `inv_act` 转到拟合域

说明：

- 这里的 `0.95 / 0.05` 是数值保护，不是为性能调参
- 不允许使用精确 `1 / 0`
- 它仍然属于数值域设计选择，因此本轮叙事必须保持：
  - `hyperparameter-light / distribution-aware`
  - 而不是 `parameter-free`
- 若后续结果很好，正确表述只能是：
  - 判别目标重写在当前数值保护设定下有效
  - 而不是“完全无参地证明了什么”

#### 4.1 Target scaling bypass
这一轮必须防止 `TELM2` 内部默认的 target 构造链再次改写外部判别目标。

要求：

- `C3` 不允许继续沿用当前 `TELM2` 内部 auto-associative target 生成路径
- 不允许让外部传入的 `0.95 / 0.05` 再被内部 min-max 缩放到新的端点

第一版推荐实现：

- 在外部先把 `0.95 / 0.05` 通过当前激活的 `inv_act` 转成目标拟合域
- 再把该 `Y_disc_logits` 直接送入加权闭式解

也就是：

> 保留广义逆求解，但绕过当前 `TELM2` 内部的默认 target 构造链。

原因：

- 这样最能避免内部缩放/反演路径重新污染判别目标
- 也最符合当前 probe 的可归因性

#### 5. Bipolar weighting
对 same 和 opp 两侧分别计算 prototype-local 权重。

对 same 样本：

\[
w_i^{same}=\exp\left(-\frac{d_i^{same}-d_{min}^{same}}{s_{proto}^{same}}\right)
\]

其中：

- `d_i^{same} = ||z_i - p_same||`
- `s_proto^{same} = max(median(d^{same}) - min(d^{same}), 1e-8)`

对 opp 样本：

\[
w_j^{opp}=\exp\left(-\frac{d_j^{opp}-d_{min}^{opp}}{s_{proto}^{opp}}\right)
\]

其中：

- `d_j^{opp} = ||z_j - p_opp||`
- `s_proto^{opp} = max(median(d^{opp}) - min(d^{opp}), 1e-8)`

两侧权重在拼接前必须显式做总质量归一化：

\[
\tilde w^{same} = \frac{w^{same}}{\sum w^{same} + 1e-8}
\]

\[
\tilde w^{opp} = \frac{w^{opp}}{\sum w^{opp} + 1e-8}
\]

然后再把两侧权重拼成完整 `Lambda`。

解释：

- 这一步不是性能调参，而是强制 same/opp 两侧的总引力保持可比较
- 否则样本更多的一侧会仅凭总量优势拉偏闭式解
- 当前 probe 的目标是比较“方向对抗”，不是比较“谁的样本更多”

极小样本簇保护：

- 若任一侧局部池的样本数 `N <= 3`
- 则该侧直接返回全 `1` 权重

#### 6. Weighted generalized inverse
最后仍走同一条加权闭式解：

\[
W=(P^T\Lambda P+\lambda I)^{-1}P^T\Lambda Y_{disc}
\]

这一步必须明确：

> `C3` 不是换求解器，而是在保留广义逆主线的前提下，重写目标和样本度量。

---

## 九、实验执行流程

### Step 0：冻结当前已知最优前端
固定：

- `A2r = sigmoid_clip_tanh_local_median_scaled_iqr`

不再重跑整套阶段 A。

### Step 1：主场只跑 `NATOPS seed=1`
固定：

- dataset: `natops`
- terminal: `dense_dynamic_gru`
- seed: `1`

### Step 2：只比较 `C0 / C2 / C3`
必须同时产出：

- `C0 = A2r + unweighted same-only`
- `C2 = A2r + strongest same-only weighted`
- `C3 = A2r + bipolar discriminative weighted`

### Step 3：只有 `C3` 至少达到“中等成立”，才允许讨论后续扩展
若不满足：

- 本轮停在 `C3`
- 不进入阶段 B
- 不进入 slow refresh
- 不放开 `r`

---

## 十、必须新增的诊断指标

### 已有并继续保留
- `template_mean_direction_cosine`
- `response_vs_margin_correlation`
- `activation_coverage_ratio`
- `margin_gain_per_unit_distortion`

### `C3` 必须新增
- `fit_target_mode`
  - `auto_associative`
  - `hetero_associative_discriminative`

- `pool_mode`
  - `same_only`
  - `bipolar_same_opp`

- `opp_pair_rule`
  - 固定记录当前采用的 opposite pairing rule

- `same_pool_count`
  - 当前 prototype 汇总下 same-side 样本数

- `opp_pool_count`
  - 当前 prototype 汇总下 opp-side 样本数

- `same_weight_mass`
  - same-side 权重总和

- `opp_weight_mass`
  - opp-side 权重总和

- `same_opp_count_ratio`
  - `same_pool_count / max(1, opp_pool_count)`

- `same_opp_weight_mass_ratio`
  - `same_weight_mass / max(1e-8, opp_weight_mass)`

- `same_proto_effective_sample_size`
  - same-side 的 `N_eff`

- `opp_proto_effective_sample_size`
  - opp-side 的 `N_eff`

- `discriminative_target_gap`
  - same 与 opp 目标值的间隔

---

## 十一、必须输出的文件
- `pia_operator_p0a1_c3_config_table.csv`
- `pia_operator_p0a1_c3_per_seed.csv`
- `pia_operator_p0a1_c3_structure_diagnostics.csv`
- `pia_operator_p0a1_c3_score_diagnostics.csv`
- `pia_operator_p0a1_c3_response_diagnostics.csv`
- `pia_operator_p0a1_c3_anchor_diagnostics.csv`
- `pia_operator_p0a1_c3_conclusion.md`

---

## 十二、必须回答的问题
1. 当前单模板 PIA 更深层的瓶颈，是否确实来自：
   - auto-associative target 错位
   - 而不仅仅是 same-only `Lambda` 太平

2. `C3` 是否相对 `C0 / C2` 显著提升：
   - `template_mean_direction_cosine`

3. `C3` 若方向改善，是否至少不明显伤害：
   - `test_macro_f1`
   - `response_vs_margin_correlation`

4. 若 `C3` 成立，是否说明：
   - 判别目标重写是决定性变量
   - 而不是单靠 same-only `Lambda` 修复即可

5. 若 `C3` 成立，下一步更像：
   - 进入阶段 B 连续几何力场
   - 还是先做更完整的 discriminative-target formal

6. `C3` 的 same/opp pooling 是否存在系统性失衡：
   - 样本数失衡
   - 权重总质量失衡
   - 若存在，是否足以污染主归因

---

## 十三、成功标准

### 弱成立
- `C3` 相比 `C0 / C2`
  - `template_mean_direction_cosine` 明显提升

### 中等成立
- `C3` 相比 `C0`
  - `template_mean_direction_cosine` 明显提升
  - `test_macro_f1` 不明显退化
  - `response_vs_margin_correlation` 不明显变坏

### 强成立
同时满足：
- `C3` 在方向指标上显著优于 `C0 / C2`
- `C3` 在终端分数上不低于当前 `A2r`
- `C3` 仍保持：
  - `response_vs_margin_correlation >= 0`
  - `activation_coverage_ratio` 处于健康低值
- 可以合理讨论后续进入阶段 B

---

## 十四、若结果不成立，如何解释

### 情形 A
`C3` 比 `C0 / C2` 方向提升不明显

说明：

- 当前问题不只是目标函数错位
- 或当前 bipolar pool pairing 仍不足以给单模板提供稳定判别法向

### 情形 B
`C3` 方向指标升了，但分数仍明显差

说明：

- 判别目标确实改变了模板方向
- 但这种方向变化尚未转成终端可读收益

### 情形 B2
`C3` 看起来提升明显，但 same/opp pool 明显失衡

说明：

- 当前结果可能同时受 pairing rule 与 pool 统计结构影响
- 不能直接把增益完整归因为“判别目标改对了”

### 情形 C
`C3` 成功，但 `C0 / C2` 仍失败

说明：

- 当前 same-only 的 Stage C 不能独立解决问题
- 判别目标重写是决定性变量之一

### 情形 D
`C0 / C2 / C3` 都站不住

说明：

- 当前单模板 PIA 在现边界下仍未立住
- 但这并不自动证明：
  - 广义逆路线无效
  - `r=1` 已被最终否定

更稳的结论是：

- 当前单模板 probe 仍未找到足以稳定学出局部判别法向的目标与样本构造方式

---

## 十五、一句话执行目标
**在保留 `A2r` 前端、`r=1`、广义逆闭式解和当前系统边界不变的前提下，单独测试“判别目标重写 + bipolar pooling + 加权闭式解”是否能把单模板 PIA 从自联想模板推进为更接近局部判别法向的单轴算子，并与 same-only 路线做诚实可归因的对照。**
