# PIA Operator P0a.1 Computational Repair Promote

## 阅读提示

这份文档是 `P0a.1` 的母文档，仍然保留最高层的理论约束与阶段边界。

如果你现在只想知道“当前代码到底已经实现到哪一步”，请先看：

- [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md)

如果你要看当前最接近现实现链的 promote，请优先看：

- [PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md)

## 任务名称
`SCP-Branch P0a.1: Single-Template PIA Computational Repair Probe`

## 零、战略定位
`P0` 已经回答了一个重要但不完整的问题：

- `single-template PIA` 不是完全无效
- 它在 `SCP1` 和 `NATOPS` 上都能稳定地比 `mean/prototype-centered update` 更平滑
- 但它仍未稳定转化成正向 margin 与分数收益

当前最关键的收束不是继续扩系统，而是先从**计算链本身**修复单模板 PIA。

这一轮的总原则明确改成：

> **Hyperparameter-light / Distribution-driven**

也就是：
- 尽量避免人工硬阈值和手工幅值常数
- 优先使用训练期分布统计量与局部几何量
- 更关心“算子结构是否成立”，而不是“是否能调出一组幸运超参”

现有 smoke 已暴露出三类信号：

1. `SCP1`
   - `template_mean_direction_cosine = 0.4333`
   - `response_vs_margin_correlation = 0.2769`
   - `activation_coverage_ratio = 0.4420`
   - 说明：有一定区域选择性，但太弱，尚未转成稳定收益

2. `NATOPS + dense_dynamic_gru`
   - `template_mean_direction_cosine = 0.0580`
   - `response_vs_margin_correlation = -0.2277`
   - `activation_coverage_ratio = 0.9958`
   - 并伴随 `sigmoid overflow warning`
   - 说明：当前单模板 PIA 很可能已经退化成**接近全域激活的全局轴门控器**

因此，这一步不是 `P0b delayed refresh`，而是：

> 先修单模板 PIA 的数值域、局部几何耦合方式和拟合窗口来源，再判断它是否值得进入一次 delayed refresh。

一句话说：

> `P0a.1` 的目标是先回答：当前单模板 PIA 的问题主要来自“模板算子本体无效”，还是来自“当前计算实现把它锁残了”。

---

## 一、唯一核心问题
**在固定 backbone、固定对象定义、固定终端的前提下，若只修复单模板 PIA 的计算链，是否能把它从“全局平滑游走器”修回“局部判别响应器”。**

这一步只问：

- 当前 `Arm B` 的主要问题是否来自计算实现
- 哪一段计算链最值得修

当前不问：

- slow refresh 是否成立
- delayed refresh 是否最终转正
- class-conditioned PIA 是否值得上升
- 多轮闭环是否成立

---

## 二、为什么这一步必须先于 `P0b`

### 1. 当前 `Arm B` 的坏相位更像计算问题，不像系统问题
现有实现里至少存在三处高风险：

1. `TELM2 + sigmoid` 的目标域与响应域可能错配
2. `Arm B` 的实际更新方向与局部 `same/opp` 几何脱钩
3. 当前拟合窗口来源偏向 `tight_margin`，可能把模板学成边界噪声轴

如果这些问题不先修，直接进入 `P0b`，很容易把：

- 本体没立住
- 数值已饱和
- 局部几何未耦合

的责任推迟给“下一轮 refresh”。

### 2. `P0b` 必须建立在“当前算子不是纯游走”的前提上
只有当单模板 PIA 已经表现出：

- 非饱和的模板响应
- 与局部 margin 非负相关
- 与局部判别方向存在明确耦合

才值得测试“一次 delayed refresh”是否能把它转成正收益。

### 3. 这是当前最小、最可审计的修复层
与直接引入：

- multi-template
- class-conditioned templates
- slow refresh
- closed-loop trigger

相比，`P0a.1` 只修三段计算链：

- 数值域
- 方向/门控
- 拟合样本源

所以可归因性最高。

---

## 三、当前全部冻结

### Backbone 固定
- `raw -> dense z_seq -> terminal`

### 对象定义固定
- `prototype-memory`
- `local representative states`
- `v1b tight anchors`
- 当前 `train-only` 局部对象定义

### slow layer 固定
- 参考几何固定
- 不做 refresh
- 不做 candidate generation
- 不做 rollback

### 训练/评估口径固定
- train-only 拟合 operator
- operator 拟合后冻结，并以同一参数作用于 `train/val/test`
- 不做 test-time adaptation
- 不做 replay
- 不做 class-conditioned routing

### 终端固定
主评估优先：
- `NATOPS + dense_dynamic_gru`

辅助交叉检查：
- `SCP1 + dynamic_minirocket`

说明：
- `NATOPS + dense_dynamic_gru` 是当前最强 backbone，也是最容易暴露 `Arm B > Arm A but < baseline` 的地方
- `SCP1` 用来检查修复是否只是对单一强数据集过拟合

---

## 四、修复实验的唯一主线
这一轮仍然只比较：

- `Baseline 0: same_backbone_no_shaping`
- `Arm A: mean/prototype-centered update`
- `Arm B*: repaired single-template PIA variants`

不进入：
- `Arm C`
- `PIA-Geom`
- `PIA-Disc`
- delayed refresh

---

## 五、三段修复顺序

### 阶段 A：数值域修复
目标：
- 先判断当前问题是否主要来自 `sigmoid` 饱和和响应域错配

当前控制组：
- `B0 current_sigmoid_minimal`

修复组：

1. `A1 sigmoid_clip_tanh`
   - 保留当前单模板 `sigmoid` 主线
   - 但不再使用手工固定响应域
   - 在 train-only 拟合完成后，统计拟合样本 pre-activation 的分位区间
   - 第一版默认使用轻量分布驱动截断：
     - `C_lower = q01(preactivation_train)`
     - `C_upper = q99(preactivation_train)`
   - 推理响应写成：

\[
a_{resp}(z)=tanh(clip(zW+b,C_{lower},C_{upper}))
\]

   - 目标：
     - 消掉 `sigmoid overflow`
     - 把全域饱和拉回健康区间
     - 尽量保留绝对幅值语义

2. `A2 sigmoid_clip_tanh_local_median`
   - 与 `A1` 使用同一 train-only 分位截断
   - 但把 trial/sequence 内 clipped pre-activation 的**中位数**作为局部基线
   - 响应写成：

\[
a_{resp}(z)=tanh(clip(zW+b,C_{lower},C_{upper})-\mu_{local})
\]

   - 其中：
     - `mu_local` 被严格定义为：**当前 trial/sequence 在 frozen operator 下，经过分位截断后的 pre-activation 响应中位数**
   - 当前定位：
     - 诊断分支
   - 只用于对照“保留绝对幅值”与“只看局部相对波动”的差异
   - 不作为默认主推

3. `A1s sigmoid_clip_tanh_scaled`
   - 在 `A1` 的基础上，额外引入 train-only 的响应尺度归一化
   - 定义：

\[
a_{resp}(z)=tanh\left(\frac{clip(zW+b,C_{lower},C_{upper})}{\sigma_{resp}}\right)
\]

   - 其中：
     - `sigma_resp` 明确取自 train-only 拟合样本在分位截断后的 pre-activation 标准差
   - 目标：
     - 检查阶段 A 的核心问题是否不是“有没有截断”，而是“截断后量级仍然过大，导致 `tanh` 继续全域饱和”

4. `A2s sigmoid_clip_tanh_local_median_scaled`
   - 在 `A2` 的基础上，额外引入同一 `sigma_resp`
   - 定义：

\[
a_{resp}(z)=tanh\left(\frac{clip(zW+b,C_{lower},C_{upper})-\mu_{local}}{\sigma_{resp}}\right)
\]

   - 当前定位：
     - 诊断分支
     - 用于对照“局部去基线”与“分布尺度归一化”是否必须同时存在

5. `A1r sigmoid_clip_tanh_scaled_iqr`
   - 与 `A1s` 相同，但冻结尺度改为稳健统计量
   - 定义：

\[
S_{train}^{iqr} = \frac{P_{75}(R_{train}) - P_{25}(R_{train})}{1.349}
\]

\[
a_{resp}(z)=tanh\left(\frac{clip(zW+b,C_{lower},C_{upper})}{S_{train}^{iqr}}\right)
\]

   - 目的：
     - 检查常规 `std(R_train)` 是否被极端噪声点拉大
     - 验证稳健 frozen scale 是否更适合 NATOPS 这条主场

6. `A2r sigmoid_clip_tanh_local_median_scaled_iqr`
   - 与 `A2s` 相同，但分母改为 `S_{train}^{iqr}`
   - 当前定位：
     - 诊断分支
     - 用于验证“局部中位数去基线 + 稳健冻结尺度”是否更合适

阶段 A 只允许修：
- `activation`
- `distribution-driven response domain`
- `response centering`
- `pre-activation clipping`

阶段 A 不允许修：
- 方向定义
- 几何门控
- anchor 选择口径

### 阶段 C：拟合窗口来源修复
进入条件：
- 阶段 A 至少有一个变体让
  - `activation_coverage_ratio` 明显下降
  - `response_vs_margin_correlation` 不再明显为负

目标：
- 判断当前模板是否被 `tight_margin` 窗口源带偏
- 在不引入硬截断 `k` 和固定 margin 门槛的前提下，让类簇拓扑自己决定谁主导模板方向

当前控制组：
- `tight_margin`

修复组：

3. `C1 soft_weighted_fit`
   - 不再硬选前 `k` 个样本
   - 仍限定在当前 prototype 的 admitted windows 统计域内
   - 对每个拟合窗口 `z_i` 计算：

\[
w_i = exp\left(- \frac{\|z_i-p_{same}\|_2}{mean\_dist_{same} + 1e-8}\right)
\]

   - 其中：
     - `mean_dist_same` 明确只在当前 prototype 的 admitted windows 内统计
   - 用带权最小二乘替代当前无权求解：

\[
W=(Z^T \Lambda Z + \lambda I)^{-1} Z^T \Lambda Y
\]

   - 工程实现采用：
     - `sqrt_w * P`
     - `sqrt_w * Y`
     - 再复用当前 `np.linalg.solve`
   - 目的：
     - 让靠近 prototype 的典型状态自动获得更高权重
     - 降低边界纠缠态对 `r=1` 模板方向的污染

4. `C2 soft_weighted_fit + local_median_diagnostic`
   - 仅用于验证 `A2` 与软加权是否耦合得更好
   - 不是默认主推

阶段 C 只允许修：
- `fit window source`
- `soft weighting rule`
- `weighted least squares fit`

阶段 C 不允许修：
- 模板数
- delayed refresh

### 阶段 B：局部几何耦合修复
进入条件：
- 阶段 C 至少有一个变体表现出
  - 非负 `response_vs_margin_correlation`
  - 高于当前 `B0` 的 `margin_gain_per_unit_distortion`

目标：
- 修复当前 `sign(centered_response) * direction` 的全局轴漂移问题
- 让模板推力与真实局部判别边界连续耦合

当前控制组：
- 阶段 C 最优变体的 `current_direction_rule`

修复组：

5. `B3 continuous_geometric_force_field`
   - 局部真实几何轴定义为：

\[
u_{geom}(z)=normalize(p_{same}-p_{opp})
\]

   - 连续对齐度定义为：

\[
g_{geom}(z)=\langle u_{template}, u_{geom}(z) \rangle
\]

   - 不额外引入手工阈值，不做 `sign()` 硬裁决
   - 最终更新写成：

\[
d_{B3}(z)=local\_step(z)\cdot a_{resp}(z)\cdot g_{geom}(z)\cdot u_{geom}(z)
\]

   - 解释：
     - 同向时自然增强
     - 正交时自然衰减
     - 背离时自然产生反向推力

阶段 B 只允许修：
- 连续几何耦合方式
- 力场方向定义
- 响应项与局部几何项的乘法耦合

阶段 B 不允许修：
- 模板数
- class-conditioned routing
- slow refresh

---

## 六、实验执行流程

### Step 0：固定当前对照口径
在现有 runner 基础上保留：
- `Baseline 0`
- `Arm A`
- `B0 current_sigmoid_minimal`

作为所有修复实验的统一参照。

### Step 1：主场数据集先只跑 `NATOPS seed=1`
口径：
- dataset: `natops`
- terminal: `dense_dynamic_gru`

原因：
- 这是当前最强 backbone
- 也是最能暴露 `Arm B` 失效形态的主场

### Step 2：阶段 A 小扫
只跑：
- `B0`
- `A1`
- `A2`
- `A1s`
- `A2s`
- `A1r`
- `A2r`

必须回答：
- 饱和问题是不是主因
- 分布驱动截断是否能让模板响应脱离全域激活
- `A1/A2`、`A1s/A2s` 与 `A1r/A2r` 中，哪一类更适合作为后续主线

### Step 3：阶段 C 软加权拟合
只对阶段 A 最优变体继续跑：
- `current_unweighted_fit`
- `soft_weighted_fit`

必须回答：
- 当前模板是不是被 `tight_margin` 边界样本喂坏
- 软加权是否能提升模板方向质量而不退化成最近邻硬截断

### Step 4：阶段 B 连续几何力场
只对阶段 C 最优变体继续跑：
- `current_direction_rule`
- `continuous_geometric_force_field`

必须回答：
- 当前问题是不是主要来自“模板方向已经够好，但局部作用规则不对”

### Step 5：辅助检查 `SCP1 seed=1`
只把阶段 B 最优方案带到：
- `selfregulationscp1`
- `dynamic_minirocket`

目的：
- 检查修复是否只是在 `NATOPS` 上偶然成立

### Step 6：只有在以下条件满足时，才扩到 formal
至少满足：
- `NATOPS seed=1` 上 best repaired `Arm B`
  - 明显优于当前 `B0`
  - 明显缩小与 `no_shaping baseline` 的差距
  - 且 `response_vs_margin_correlation >= 0`
  - 且 `activation_coverage_ratio` 明显低于当前 `0.9958`

若不满足：
- 本轮停止在 smoke
- 不进入 `P0b`

---

## 七、必须新增的诊断指标

### 继续保留
- `template_mean_direction_cosine`
- `response_vs_margin_correlation`
- `activation_coverage_ratio`
- `margin_gain_per_unit_distortion`

### 必须新增
- `preactivation_clip_rate`
  - 看当前激活是否仍频繁进入数值极端区

- `response_centering_std_after_fix`
  - 看响应中心化后是否仍近似塌缩

- `geometry_alignment_cosine_mean`
  - 模板方向与局部 `same/opp` 判别轴的平均对齐程度

- `gate_saturation_ratio`
  - `response_gate >= 0.95` 的比例

- `fit_anchor_margin_mean`
  - 拟合窗口本身的平均初始 margin

- `fit_anchor_same_dist_mean`
  - 拟合窗口与其 prototype 的平均距离

- `effective_sample_size`
  - 软加权拟合中的有效样本数
  - 定义为：

\[
N_{eff} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
\]

  - 用于检查软权重是否退化成“只有极少数样本在说了算”

---

## 八、必须输出的文件
- `pia_operator_p0a_repair_config_table.csv`
- `pia_operator_p0a_repair_per_seed.csv`
- `pia_operator_p0a_repair_dataset_summary.csv`
- `pia_operator_p0a_repair_structure_diagnostics.csv`
- `pia_operator_p0a_repair_score_diagnostics.csv`
- `pia_operator_p0a_repair_response_diagnostics.csv`
- `pia_operator_p0a_repair_anchor_diagnostics.csv`
- `pia_operator_p0a_repair_conclusion.md`

---

## 九、必须回答的问题
1. 当前 `Arm B` 的主要问题是否首先来自：
   - `sigmoid` 数值域/目标域错配
   - 还是 `tight_margin` 拟合窗口源过于贴边界
   - 还是局部几何未耦合

2. 修复后，`activation_coverage_ratio` 是否从接近全域激活回落

3. 修复后，`response_vs_margin_correlation` 是否由负转零或转正

4. 修复后，`margin_gain_per_unit_distortion` 是否优于当前 `B0`

5. 修复后，best `Arm B` 是否：
   - 优于当前 `Arm A`
   - 优于当前 `B0`
   - 并明显缩小与 `no_shaping baseline` 的差距

6. 若上述问题都成立，是否值得进入：
   - `P0b: One-Step Delayed Effect Probe`

---

## 十、成功标准

### 弱成立
- 至少一个 repaired `Arm B` 变体让
  - `activation_coverage_ratio` 明显下降
  - `response_vs_margin_correlation` 不再为负

### 中等成立
- best repaired `Arm B` 优于当前 `B0`
- 且 `margin_gain_per_unit_distortion` 提升
- 且与 `no_shaping baseline` 的差距显著缩小

### 强成立
同时满足：
- best repaired `Arm B` 在 `NATOPS` 上优于 `Arm A` 和当前 `B0`
- `response_vs_margin_correlation >= 0`
- `activation_coverage_ratio` 不再接近 1
- `geometry_alignment_cosine_mean` 明显提升
- 可以合理进入一次受控的 `P0b delayed refresh`

---

## 十一、若结果不成立，如何解释

### 情形 A
阶段 A 就失败：
- 数值域修复后仍接近全域激活

说明：
- 单模板 PIA 当前不只是“参数保守”
- 更可能是 `r=1` 模型本身太弱

### 情形 B
阶段 A 成立，但阶段 C 失败：
- 响应脱饱和了，但模板方向仍没改善

说明：
- 当前核心问题更像不是“纯数值域”，而是 `r=1` 模板容量或当前软加权定义仍不足

### 情形 C
阶段 C 成立，但阶段 B 失败：
- 模板方向更健康了，但连续几何力场仍无法把它转成有效局部判别作用

说明：
- 当前核心问题更像局部几何耦合规则不对

### 情形 D
三个阶段都修不好

说明：
- 当前单模板 PIA 不是框架内核候选
- 至多作为弱几何先验保留
- 不应进入 `P0b`

---

## 十二、与后续路线的关系
只有 `P0a.1` 至少达到中等成立，后续才值得进入：

- `P0b: One-Step Delayed Effect Probe`

如果 `P0a.1` 站不住，则当前结论应收成：

- 单模板 PIA 仍未立住
- 不能靠 delayed refresh 继续延后解释

---

## 十三、一句话执行目标
**先把当前单模板 PIA 的问题压回到三个最小计算层面：数值域、局部几何耦合和拟合窗口来源；在固定参考几何和固定 backbone 下，判断它到底是“模板算子本体不行”，还是“当前实现把它锁成了一个全局平滑游走器”。**

补充执行契约：
- 当前实现优先只推进阶段 A
- 后续若继续推进，严格按 `A -> C -> B` 串行
- 阶段 B 和阶段 C 必须沿用阶段 A 的同一扰动预算匹配
- 不允许通过重新调大 `epsilon` 或平滑系数来人为制造后续阶段的“改进”
