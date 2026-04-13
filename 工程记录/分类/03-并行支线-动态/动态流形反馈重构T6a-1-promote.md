# 动态流形反馈重构 T6a-1 Promote

更新时间：2026-03-30

## 零、战略定位

当前动态主线已经从：

- 对象成立
- 算子成立
- 骨架开始可塑

推进到：

**样本角色定义之前的局部判别 gate 校准阶段**

当前下一步不再优先推进：

- `T2a / T2b` 参数细磨
- `T4a` basis family 扩张
- 双流 / 回桥
- 多轮 feedback

而是先问：

> 更局部、更任务相关的 `local kNN margin`，在 unified role 口径下，是否比当前 `T3 / T4b` 更能同时推动 `SCP1` 的骨架变化与终点收益。

---

## 一、唯一核心问题

**当同一批通过 `safety + local kNN margin` 的窗口同时承担 rebasis 与最终训练增强角色时，`SCP1` 是否比当前 `T3 / T4b` 更好。**

---

## 二、为什么这是当前最合理入口

1. 不是继续磨 `T4b` 单一 gate  
当前更深的矛盾已经不是 `radial` 还是 `global margin`，而是更局部的判别 signal 是否才是更合适的对象。

2. 不是继续扩 `T4a`  
`SCP1` 上最小 class-conditioned family 还没有被真正触发。

3. 不是直接做 `T6a-2`  
如果现在把 `local gate + class-conditioned basis` 一起开，因果会再次变脏。

4. 不是直接做 dual-role  
`T5` formal 已经说明角色分离框架值得保留，但当前更该先把 **局部 gate 本身** 判清楚。

---

## 三、当前全部冻结

以下全部冻结：

- 当前 `trajectory_representation.py`
- 当前窗口策略
- 当前 `dynamic_gru`
- 当前训练超参
- `T2a default = gamma_main 0.05, smooth_lambda 0.50`
- `shared single-axis constrained rebasis`
- 当前 `safety gate`
- `center_new = orig-train windows ∪ admitted feedback windows` 的全局 pooled center
- 所有 purity / local margin 的 reference set 都固定为：
  - `orig-train-only windows`
- 所有类中心与局部邻域统计都在每个 `dataset × seed` 中只计算一次，并固定复用

当前不改：

- generator
- basis family
- 窗口策略
- 双流 / 回桥
- 多轮 feedback

---

## 四、T6a-1 第一版定义

### 1. 生成器固定

仍使用冻结后的 `T2a default`，只在 `train trajectories` 上生成增强。

### 2. feedback object 固定为 window-level

本轮不再回到 whole-trajectory object。  
每个候选对象是：

- `z_t^orig`
- `z_t^aug`
- `trial_id`
- `window_index`
- `class_id`

### 3. gate 结构固定为：`safety + local_kNN_margin_gain`

#### safety gate

完全冻结现有 `T4b` 口径，只是继续在 window-level 上执行。

#### local kNN margin

第一版定义：

\[
\mathrm{local\_margin}(z)
=
\mathrm{mean}\; d(z, k\text{-NN}_{\text{diff}})
-
\mathrm{mean}\; d(z, k\text{-NN}_{\text{same}})
\]

其中：

- `same-class / different-class` 近邻都统一相对于 `orig-train-only windows` 计算
- `same-class` 近邻统计时必须 **排除当前窗口自身**
- 不允许把增强窗口、admitted pool、rebasis 后样本混入 reference set

于是：

\[
\mathrm{local\_knn\_margin\_gain}(z_t)
=
\mathrm{local\_margin}(z_t^{aug})
-
\mathrm{local\_margin}(z_t^{orig})
\]

### 4. admission 规则

第一版固定为：

1. 先过 `safety gate`
2. 再在 `safe windows` 中，按类保留：
   - `local_knn_margin_gain >= classwise median`

### 5. coverage guard

为避免“极少数偶然窗口”的假成功，第一版加入硬约束：

对于每个 `dataset × seed × class`：

\[
\mathrm{coverage\_threshold}
=
\max(8,\lceil 0.05 \cdot \mathrm{safe\_window\_count\_class} \rceil)
\]

若：

- `admitted_window_count_class < coverage_threshold`

则该类记为：

- `low_coverage_flag = 1`
- `effective_trigger = 0`

当前不因为 low-coverage 直接否掉整个实验，但必须在 summary 与 conclusion 中显式记录。

### 6. unified role

`T6a-1` 当前采用 **unified role**，不是 dual-role。

也就是说，这批 admitted windows 同时用于：

- `feedback / rebasis`
- 最终 augmented trajectory 的窗口回写位置

### 7. re-center / re-basis

沿用当前 `T3 / T4b` 口径：

- `re-center`
- `shared single-axis constrained rebasis`

当前不引入 class-conditioned basis。

### 8. 最终 augmented trajectory 的构造

最终 augmented trajectory 只在 `z_seq` 表示层构造：

- 通过 gate 的窗口位置，用 `rebased augmented window` 回写
- 其余窗口保持原始值
- 若一条轨迹没有任何窗口通过 gate，则不生成其 augmented twin

硬约束：

- 所有回写都只发生在 `z_seq` 表示层
- 不允许回写 raw trial
- 不允许引入 raw-level stitching

### 9. stitching / local continuity 诊断

由于最终 augmented trajectory 是窗口替换式构造，因此必须强制输出：

- `masked_window_ratio`
- `stitch_boundary_count`
- `stitch_boundary_jump_ratio_mean`
- `stitched_continuity_distortion_ratio`

收益不能建立在明显的局部撕裂之上。

---

## 五、方法学口径必须写清

`T6a-1` 当前测的是：

**local kNN margin unified policy 的整体效果**

当前不宣称：

- `local kNN margin` 已被严格证明是最终正确 gate
- `shared basis` 已被严格证明足够或不够

它只是一个更局部、更任务相关的 gate probe，用来判断是否值得进入：

- `T6a-2: Local kNN Margin + Class-Conditioned Basis`

---

## 六、主比较对象

### `SCP1` 主矩阵

- `trajectory baseline`
- `T2a default`
- `T3 shared rebasis`
- `T4b window_radial_gate`
- `T4b window_margin_gate`
- `T6a-1 local_knn_margin_unified`

### `NATOPS` 锚点

- `T3 shared rebasis`
- `T4b window_margin_gate`
- `T6a-1 local_knn_margin_unified`

`NATOPS` 当前只作为 stability reference，不与 `SCP1` 同权。

---

## 七、允许做 / 不允许做

### 允许做

- 只升级 gate 为 `local kNN margin`
- 保持 unified role
- 保持 shared basis
- 保持 window-level object
- 保持当前 generator

### 不允许做

- 不引入 class-conditioned basis
- 不引入 dual-role
- 不引入新 PIA
- 不改窗口策略
- 不做双流 / 回桥
- 不做多轮 feedback
- 不把 `T6a` 扩成新的 gate zoo

---

## 八、建议新增模块

- `route_b_unified/trajectory_feedback_pool_windows.py`
  - 继续扩展，新增 `local_knn_margin`
- `route_b_unified/trajectory_unified_window_policy.py`
- `scripts/route_b/run_route_b_dynamic_local_knn_margin_t6a1.py`

---

## 九、必须输出的文件

1. `dynamic_local_knn_margin_t6a1_config_table.csv`
2. `dynamic_local_knn_margin_t6a1_per_seed.csv`
3. `dynamic_local_knn_margin_t6a1_dataset_summary.csv`
4. `dynamic_local_knn_margin_t6a1_window_pool_summary.csv`
5. `dynamic_local_knn_margin_t6a1_class_coverage_summary.csv`
6. `dynamic_local_knn_margin_t6a1_basis_shift_summary.csv`
7. `dynamic_local_knn_margin_t6a1_stitching_summary.csv`
8. `dynamic_local_knn_margin_t6a1_diagnostics_summary.csv`
9. `dynamic_local_knn_margin_t6a1_conclusion.md`

---

## 十、必须回答的问题

1. `T6a-1` 是否优于 `T3`
2. `T6a-1` 是否优于 `T4b window_margin_gate`
3. `T6a-1` 是否优于 `T4b window_radial_gate`
4. `SCP1` 上是否终于同时出现：
   - 更明显的 basis 变化
   - 更好的终点收益
5. 若 `T6a-1` 成立，是否值得进入：
   - `T6a-2: Local kNN Margin + Class-Conditioned Basis`

---

## 十一、成功标准

### 弱成立

- `T6a-1 > T3`
- 且 stitching 没有明显恶化

### 中等成立

- `T6a-1 > T4b window_margin_gate`
- 说明更局部的判别 gate 比当前 global margin 更合适

### 强成立

同时满足：

- `SCP1` 上出现终点提升
- `SCP1` 上出现非平凡、可解释的 basis shift
- 主要类别不是 `low_coverage`
- `NATOPS` 不明显回退

---

## 十二、T6a-2 的触发条件

只有当 `T6a-1` 在 `SCP1` 上出现明确正信号时，才进入：

- `T6a-2: Local kNN Margin + Class-Conditioned Basis`

也就是说，`T6a-2` 不是默认下一步，而是 `T6a-1` 成功后的升级路线。

---

## 十三、一句话执行目标

**冻结当前动态主线已有组件，只把 gate 从当前全局 `radial / margin` 升级为更局部的 `local kNN margin`，并在 unified role 口径下测试它是否终于能让 `SCP1` 同时获得更好的骨架变化与终点收益。**
