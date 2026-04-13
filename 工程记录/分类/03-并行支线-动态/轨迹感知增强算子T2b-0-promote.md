# 轨迹感知增强算子 T2b-0 Promote

更新时间：2026-03-29

## 零、战略定位

当前不再继续把 `T2a` 当作长期主战场做局部调参。

当前工程决策是：

- 冻结 `T2a` 的默认实现与默认点
- 以它作为已成立的全局轨迹算子基线
- 直接推进下一层框架：`T2b`

但必须把方法学口径写清楚：

> `T2b` 第一版不是完整局部动力学算子，而是一个 **fixed-rule local saliency probe**，用来测试“局部时间感知是否值得进入主线”。

也就是说，本轮不是去证明一个完整的动力学系统已经被构造出来，而是要判断：

- 从“全局统一轨迹增强”推进到“局部时间感知轨迹增强”
- 是否已经构成下一层真正值得推进的框架升级

---

## 一、任务目标

本轮只回答一个核心问题：

**在冻结后的 `T2a` 基础上，引入固定规则的局部时间感知强度分配后，是否比全局统一强度更有效，尤其是否更适合 `SelfRegulationSCP1`。**

本轮不是：

- `T2a` 第二轮调参
- 完整动力学建模
- 双流融合
- bridge 回流
- feedback / re-basis
- 多 operator zoo
- controller / scheduler 搜索
- `DTW / Fréchet` 大系统

本轮只做：

1. 固定 `T2a` 默认点
2. 在共享全局 basis 上增加局部时间感知强度分配
3. 加入随机局部时变强度对照
4. 比较：
   - `trajectory baseline`
   - `T2a default`
   - `T2b saliency-aware`
   - `T2b randomized local-varying control`

---

## 二、当前固定前提

以下内容全部冻结：

- 当前窗口策略
- 当前 `trajectory_representation.py`
- 当前 pooled-train-windows -> shared basis 的学习方式
- `axis_count = 1`
- 当前 `dynamic_gru`
- 训练轮数 / 优化器 / batch / 学习率等训练超参
- 训练构造方式：
  - `train_final = orig_z_seq_train + aug_z_seq_train`
- `val/test` 永远保持原始 trajectory 表示
- `T2a` 默认点冻结为：
  - `gamma_main = 0.05`
  - `smooth_lambda = 0.50`

一句话说：

> 本轮不再调 `T2a`，而是把冻结后的 `T2a` 当成默认参照，直接验证 `T2b-0` 这个新算子层。

---

## 三、T2b-0 的最小方法定义

### 1. basis 保持不变

- 继续复用 `T2a` 的 shared basis
- 不改 `TELM2`
- 不改 basis 学习对象
- 不改 `axis_count`

### 2. 局部 saliency 定义

对每条轨迹的每个时间步，定义：

\[
s_t = \|z_t - z_{t-1}\|_2 + \|z_{t+1} - 2z_t + z_{t-1}\|_2
\]

边界继续采用最小合理近似：

- edge padding / 边缘复制
- 即：
  - `z_{-1} = z_0`
  - `z_K = z_{K-1}`

### 3. 轨迹内三档离散映射

在每条轨迹内部，对 `s_t` 按分位数分三档：

- low
- mid
- high

固定倍率映射：

- low -> `0.5`
- mid -> `1.0`
- high -> `1.5`

固定：

- `gamma_base = 0.05`
- `smooth_lambda = 0.50`

于是：

\[
\gamma_t = \gamma_{base} \cdot m_t,\quad m_t \in \{0.5, 1.0, 1.5\}
\]

### 4. 局部增强作用方式

仍沿共享 basis 做投影：

\[
\delta_t = \gamma_t \cdot \mathrm{comp}(z_t)
\]

再做与 `T2a` 相同的最小平滑：

\[
\tilde{\delta}_t = (1-\lambda)\delta_t + \frac{\lambda}{2}(\delta_{t-1} + \delta_{t+1})
\]

其中：

- `smooth_lambda = 0.50`

最终：

\[
z_t^{aug} = z_t + \tilde{\delta}_t
\]

### 5. 方法学口径必须写清楚

本轮不把下面这句话当前提：

- 高变化段一定更值得增强

而是把它当成需要测试的假设：

> `T2b-0` 第一版的目标，不是假定高变化段一定更优，而是测试“基于局部变化强弱进行强度分配”是否比全局统一强度更有效；若无收益，不应继续默认高变化段更该增强。

换句话说：

- 这轮测试的是这个假设是否成立
- 不是把它写成先验真理

---

## 四、关键新增对照

本轮主比较对象不能只有：

- `trajectory baseline`
- `T2a default`
- `T2b saliency-aware`

还必须增加一个关键对照：

- `T2b randomized local-varying control`

其定义必须写死：

- 保持每条轨迹内 `low/mid/high` 档位的数量不变
- 保持与 `T2b saliency-aware` 相同的平均强度
- 但将这些档位在时间轴上随机打乱分配

这个对照的作用是隔离：

- 是“局部时间感知”真正有效
- 还是“任何时变扰动多样性”都可能带来类似收益

因此，如果本轮 `T2b saliency-aware` 赢了，但没有赢过随机对照，则结论只能写成：

- fixed-rule local-varying operator 有价值

不能上升成：

- 基于 saliency 的局部增强已经被证明

---

## 五、任务边界

允许做：

- 复用当前 `trajectory_pia_operator.py`
- 在其上新增 `T2b-0` 所需的局部强度分配逻辑
- 复用当前 `trajectory_pia_evaluator.py`
- 复用当前 `dynamic_gru` 终端
- 复用当前结构诊断口径
- 在同一 runner 中比较：
  - baseline
  - `T2a default`
  - `T2b saliency-aware`
  - `T2b randomized local-varying control`

不允许做：

- 改 basis 学习方式
- 改窗口策略
- 改 `axis_count`
- 改分类头
- 引入 bridge / raw 回流
- 引入 feedback / re-basis
- 引入双流
- 开局部 saliency 网格搜索
- 比较多种倍率表
- 比较多种 smoothing 规则
- 引入可学习 gating 网络

---

## 六、数据集与执行规模

只做：

- `NATOPS`
- `SelfRegulationSCP1`

只做：

- `3 seeds`
- 小规模正式小矩阵

当前不做参数网格。  
当前只做系统级比较。

---

## 七、工程实现要求

建议新增：

- `route_b_unified/trajectory_pia_operator_t2b.py`
- `scripts/route_b/run_route_b_trajectory_pia_t2b0.py`

### 1. `trajectory_pia_operator_t2b.py`

职责：

- 复用 `T2a` 的 shared basis 学习逻辑
- 计算每条轨迹的局部 saliency
- 生成：
  - `saliency-aware` 局部强度分配
  - `randomized local-varying control`
- 输出：
  - `z_seq_aug`
  - `delta_seq`
  - `gamma_seq`
  - `operator_meta`

### 2. `run_route_b_trajectory_pia_t2b0.py`

职责：

- 组织：
  - `trajectory baseline`
  - `T2a default`
  - `T2b saliency-aware`
  - `T2b randomized control`
- 在：
  - `NATOPS + SCP1`
  - `3 seeds`
  上跑完整小矩阵

---

## 八、短轨迹实现边界

实现分位数时，允许采用最小、鲁棒、可复现处理：

- 用 `numpy.percentile`
- 对极短轨迹自然退化处理

例如当某条轨迹极短时：

- 允许三档分位数退化成少于三档的有效分配
- 但不为此额外引入复杂特殊规则

这属于实现细节，不改变 promote 主体。

---

## 九、必须回答的问题

1. `T2b saliency-aware` 是否优于 `trajectory baseline`
2. `T2b saliency-aware` 是否优于冻结后的 `T2a default`
3. `T2b saliency-aware` 是否优于 `randomized local-varying control`
4. `SCP1` 是否比 `NATOPS` 更受益于局部时间感知增强
5. 若 `T2b` 没有赢过随机对照，是否说明当前问题还没抓住真正动力学对象

---

## 十、结构诊断要求

除了分类指标，必须输出结构诊断。

至少输出：

- `step_change_mean`
- `local_curvature_proxy`
- `classwise_dispersion`
- `transition_separation_proxy`
- `continuity_distortion_ratio`
- `saliency_low_ratio`
- `saliency_mid_ratio`
- `saliency_high_ratio`
- `gamma_effective_mean`

第一轮这些 proxy 允许采用最小、简单、可复现定义，不要求理论最优。  
重点是保证：

- `baseline`
- `T2a default`
- `T2b saliency-aware`
- `T2b randomized control`

四者之间口径一致。

---

## 十一、输出文件要求

必须输出：

1. `trajectory_pia_t2b0_config_table.csv`
2. `trajectory_pia_t2b0_per_seed.csv`
3. `trajectory_pia_t2b0_dataset_summary.csv`
4. `trajectory_pia_t2b0_diagnostics_summary.csv`
5. `trajectory_pia_t2b0_conclusion.md`

建议字段：

### `trajectory_pia_t2b0_per_seed.csv`

- `dataset`
- `seed`
- `operator_mode`
- `gamma_base`
- `smooth_lambda`
- `test_macro_f1`
- `delta_vs_baseline`
- `delta_vs_t2a_default`

### `trajectory_pia_t2b0_dataset_summary.csv`

- `dataset`
- `baseline_macro_f1_mean/std`
- `t2a_default_macro_f1_mean/std`
- `t2b_saliency_macro_f1_mean/std`
- `t2b_random_macro_f1_mean/std`
- `best_mode`

### `trajectory_pia_t2b0_diagnostics_summary.csv`

- `dataset`
- `seed`
- `operator_mode`
- `step_change_mean`
- `local_curvature_proxy`
- `classwise_dispersion`
- `transition_separation_proxy`
- `continuity_distortion_ratio`
- `saliency_low_ratio`
- `saliency_mid_ratio`
- `saliency_high_ratio`
- `gamma_effective_mean`

### `trajectory_pia_t2b0_conclusion.md`

必须明确回答：

1. `T2b saliency-aware` 是否优于：
   - `trajectory baseline`
   - `T2a default`
   - `randomized control`
2. `SCP1` 是否比 `NATOPS` 更受益
3. 当前是否值得继续沿 `T2b` 推进，而不是回头继续调 `T2a`

---

## 十二、成功标准

满足至少两条即可视为 `T2b-0` 成立：

1. `T2b saliency-aware > T2a default` 至少在一个主数据集上成立
2. `T2b saliency-aware > randomized control`
3. `SCP1` 上出现比 `T2a default` 更稳定的改善
4. 结构诊断显示：
   - `transition_separation_proxy` 更好
   - 且 `continuity_distortion_ratio` 未明显恶化

---

## 十三、当前明确不做的事

这一轮明确不扩成：

- `T2a` 新一轮调参
- 多种 saliency 定义比较
- 多种倍率表比较
- 多种 smoothing 规则比较
- 可学习 gating 网络
- 双流
- bridge 回流
- feedback / re-basis

---

## 十四、建议执行顺序

### Step 1

冻结当前 `T2a default`，不再继续局部调参。

### Step 2

实现 `T2b-0` 的局部 saliency 计算与三档固定规则映射。

### Step 3

实现 `randomized local-varying control`。

### Step 4

运行：

- `trajectory baseline`
- `T2a default`
- `T2b saliency-aware`
- `T2b randomized control`

在：

- `NATOPS + SCP1`
- `3 seeds`

上的正式小矩阵。

### Step 5

输出正式结果与结论文档，明确判断：

- 局部时间感知是否真有价值
- 是否比随机时变扰动更好
- 是否值得继续推进 `T2b`

---

## 十五、一句话执行目标

**冻结已成立的 `T2a` 默认点，不再继续局部调参；先把 `T2b` 落成一个 fixed-rule local saliency probe，并加入随机局部时变强度对照，判断“局部时间感知强度分配”是否真比全局统一强度和随机时变扰动更有效。**
