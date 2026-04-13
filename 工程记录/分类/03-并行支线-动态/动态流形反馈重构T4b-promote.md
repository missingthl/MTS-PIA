# 动态流形反馈重构 T4b Promote

更新时间：2026-03-29

## 零、战略定位

当前动态主线已经推进到：

- 动态流形表示成立（`T0`）
- 全局 trajectory operator 成立（`T2a`）
- 局部时间感知方向值得保留（`T2b-0`）
- shared rebasis 在部分数据集上开始成立（`T3`）
- 最小 class-conditioned basis family 在 `SCP1` 上未被真正触发（`T4a smoke`）

因此，当前下一步不再优先推进：

- `T2a/T2b` 参数细磨
- `T4a` basis family 扩张
- 双流 / 回桥
- 多轮 feedback 闭环

当前更合理的入口是：

**T4b：Window-Conditioned Rebasis-Informative Feedback Pool Probe**

一句话说：

> `T4b` 不是继续改 generator 或 basis，而是测试“谁回流、凭什么回流”是否才是当前 `SCP1` 骨架不动的主瓶颈。

---

## 一、唯一核心问题

**把 feedback pool 的回流对象从“整条安全增强轨迹”升级为“安全且更有骨架信息密度的窗口对象”后，`SCP1` 的 rebasis 是否终于会动起来，并优于当前 `T3 shared rebasis`。**

---

## 二、为什么这是当前最合理入口

1. 不是继续磨 `T2a/T2b`  
   当前已经进入边际收益区。

2. 不是继续扩 `T4a`  
   `SCP1` 上最小 class-conditioned family 几乎塌回 shared-like，更像说明当前回流对象信息密度不足。

3. 不是直接上二轮回流  
   当前连第一轮“什么对象值得回流”都还没打通。

4. 不是回静态线或上双流  
   当前动态单流主线内部的骨架瓶颈仍未解决。

---

## 三、当前全部冻结

以下内容在 `T4b` 第一版中全部冻结：

- 当前 `trajectory_representation.py`
- 当前窗口策略
- 当前 `dynamic_gru`
- 当前训练超参
- `T2a default = gamma_main 0.05, smooth_lambda 0.50`
- 当前 `T3` 的 shared single-axis constrained rebasis
- `center_new` 定义固定为：
  - `orig-train windows ∪ admitted feedback windows` 的全局 pooled center
- 所有 `kNN purity` 的 reference set 固定为：
  - `orig-train-only windows`

当前不改：

- generator
- basis family
- `T2b` saliency
- 双流 / 回桥
- 多轮 feedback

---

## 四、T4b 第一版定义

### 1. 主改动：window-level feedback object

feedback pool 的基本对象从：

- `trajectory-level augmented sequence`

升级为：

- `window-level object`
  - `z_t^orig`
  - `z_t^aug`
  - `trial_id`
  - `window_index`
  - `class_id`

### 2. safety gate 冻结，只下沉到 window level

第一版仍保留 `T3` 的 safety 逻辑，但统计下沉到窗口层级。

录取前必须先满足：

1. `majority_aug_window == true class`
2. `purity_drop_window <= 0.10`
3. `local_continuity_ratio <= classwise q75`

其中必须写死：

> `local_continuity_ratio` 统一定义为：该窗口增强后与其相邻窗口构成的局部 step-change，相对原始轨迹同一局部邻域 step-change 的比例；其计算方式必须与当前 `T3/T4a` 的 continuity 口径兼容，只是下沉到 window-centered local neighborhood。

也就是说：

- continuity 仍依赖局部邻域
- 不是单窗口孤立统计
- 边界窗口统一采用 edge padding / 一侧近似

### 3. informative gate 作为最小对照层

在 `safety gate` 通过之后，第一版只比较三种 admission 模式：

1. `window_safety_only`
2. `window_radial_gate`
3. `window_margin_gate`

#### `radial_gain_window`

定义：

- `radial_gain_window = ||z_t^aug - center_old|| - ||z_t^orig - center_old||`

它的含义是：

- 当前窗口是否把样本往经验分布外缘推开

必须写清：

- 它只是 `outward-expansion proxy`
- 不是最终正确的 rebasis signal

#### `margin_gain_window`

定义：

- `margin(z) = d_neg(z) - d_pos(z)`
- `d_pos(z) = ||z - c_y||`
- `d_neg(z) = min_{c != y} ||z - c_c||`
- `margin_gain_window = margin(z_t^aug) - margin(z_t^orig)`

其中类中心必须写死：

> 所有类中心统一由 `orig-train-only window class centers` 计算；在每个 `dataset × seed` 中只计算一次，并在整轮 `T4b` sweep 中冻结复用；不允许因为不同 gate、不同 admitted pool 或 rebasis 结果而重算类中心。

#### gate 规则

第一版固定规则：

- 先过 `safety gate`
- `window_safety_only`：保留全部 safe windows
- `window_radial_gate`：在 safe windows 中按类保留 `radial_gain_window >= classwise median`
- `window_margin_gate`：在 safe windows 中按类保留 `margin_gain_window >= classwise median`

当前不做：

- 混合 gate
- 动态权重
- gate zoo

### 4. rebasis 口径不变

为了隔离变量，`T4b` 第一版继续保持：

- global pooled `center_new`
- shared single-axis constrained rebasis

admitted windows 进入 rebasis 时，允许按长度为 `1` 的伪序列送入当前 rebasis 模块。

但必须明确：

> `T4b` 第一版测的是 **window-conditioned rebasis signal**，不是完整的 segment-aware / trajectory-aware rebasis geometry。

### 5. 最终训练口径不变

feedback windows 只用于：

- `re-center`
- `re-basis`

最终训练仍然是：

- `orig trajectories + new_aug_on_original_trajectories`

而不是：

- `orig + feedback windows + new_aug`

---

## 五、数据集权重

### 1. `SCP1`：主矩阵

`SelfRegulationSCP1` 是 `T4b` 的主矩阵，跑完整对照：

- `baseline`
- `T2a default`
- `T3 shared rebasis`
- `window_safety_only`
- `window_radial_gate`
- `window_margin_gate`

### 2. `NATOPS`：锚点 / stability reference

`NATOPS` 不与 `SCP1` 同权。

第一版作为稳定性参考，至少保留：

- `T3 shared rebasis`
- `window_safety_only`
- `window_radial_gate`
- `window_margin_gate`

`baseline / T2a default` 可作为背景参考保留，但不构成本轮核心判定。

---

## 六、允许做 / 不允许做

### 允许做

- 新增 window-level feedback pool
- 冻结 safety gate
- 新增 radial / margin 两个固定 informative gate
- 复用 `T3` 的 re-center / rebasis / evaluator 主链

### 不允许做

- 改 generator
- 改 basis family
- 改窗口策略
- 改 `dynamic_gru`
- 改 `T2a/T2b` 参数
- 引入双流 / 回桥
- 引入多轮 feedback
- 把 `T4b` 扩成新的大 zoo

---

## 七、主比较对象

### `SCP1` 主矩阵

1. `trajectory baseline`
2. `T2a default`
3. `T3 shared rebasis`
4. `T4b window_safety_only`
5. `T4b window_radial_gate`
6. `T4b window_margin_gate`

### `NATOPS` 锚点

1. `T3 shared rebasis`
2. `T4b window_safety_only`
3. `T4b window_radial_gate`
4. `T4b window_margin_gate`

---

## 八、建议新增模块

- `route_b_unified/trajectory_feedback_pool_windows.py`
- `scripts/run_route_b_dynamic_feedback_rebasis_t4b.py`

---

## 九、必须输出的文件

1. `dynamic_feedback_rebasis_t4b_config_table.csv`
2. `dynamic_feedback_rebasis_t4b_per_seed.csv`
3. `dynamic_feedback_rebasis_t4b_dataset_summary.csv`
4. `dynamic_feedback_rebasis_t4b_window_pool_summary.csv`
5. `dynamic_feedback_rebasis_t4b_basis_shift_summary.csv`
6. `dynamic_feedback_rebasis_t4b_diagnostics_summary.csv`
7. `dynamic_feedback_rebasis_t4b_conclusion.md`

`window_pool_summary` 至少记录：

- `candidate_window_count`
- `safe_window_count`
- `accepted_window_count`
- `accept_rate`
- `source_trial_coverage`
- `class_balance_proxy`
- `mean_radial_gain_accepted`
- `mean_margin_gain_accepted`

---

## 十、必须回答的问题

1. `window_safety_only` 是否已经优于 `T3 shared rebasis`
2. `window_radial_gate` 或 `window_margin_gate` 是否进一步优于 `window_safety_only`
3. `SCP1` 是否终于出现比 `T3` 更明确的改善
4. `margin_gate` 是否比 `radial_gate` 更能推动 `SCP1` 的 basis 变化
5. 若三种 window-pool 版本都不成立，问题是否已不主要在 feedback object，而更可能在 shared rebasis 范式本身

---

## 十一、成功标准（分层）

### 弱成立

- `window_safety_only > T3 shared rebasis`

含义：

- object upgrade 本身有价值

### 中等成立

- `window_margin_gate` 或 `window_radial_gate > window_safety_only`

含义：

- informative gate 确有增量价值

### 强成立

同时满足：

- `SCP1` 上性能改善
- basis shift 从“几乎不动”变成“非平凡且可解释”

必须强调：

- 仅仅 basis 变化变大不构成成功
- 仅仅 accepted windows 变多不构成成功
- 只有当性能、骨架变化、pool 稳定性方向一致时，才可视为强成立

---

## 十二、一句话执行目标

**冻结当前动态主线已有组件，不再继续局部调参；直接把 feedback pool 对象从 whole-trajectory 升级为 window-level object，并在冻结的 safety gate 上对比 radial / margin 两个固定 informative gate，测试 `SCP1` 的当前瓶颈是否主要来自回流对象与录取逻辑过粗。**
