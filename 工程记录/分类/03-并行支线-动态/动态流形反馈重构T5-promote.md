# 动态流形反馈重构 T5 Promote

更新时间：2026-03-29

## 零、战略定位

当前动态主线已经从：

- 对象成立（`T0`）
- 算子成立（`T2a / T2b-0`）
- 骨架开始可塑（`T3 / T4b`）

推进到：

**样本角色定义阶段**

当前下一步不再优先推进：

- `T2a/T2b` 参数细磨
- `T4a` basis family 扩张
- 双流 / 回桥
- 多轮 feedback 闭环

当前主推进方向明确为：

**T5：Dual-role Sample Policy Probe**

一句话说：

> `T5` 不再问“单一 gate 里谁更好”，而是问“constructive role 和 discriminative role 是否本就该由不同样本承担”。

---

## 一、唯一核心问题

**当 constructive samples 和 discriminative samples 被正式拆开使用后，`SCP1` 是否终于能同时出现更好的骨架变化与更好的终点收益。**

---

## 二、为什么这是当前最合理入口

1. 不是继续磨 `T4b` 单一 gate  
   当前更深的矛盾已经不是 `radial` 还是 `margin`，而是同一套 gate 在同时承担两种样本角色。

2. 不是继续扩 `T4a`  
   `SCP1` 上最小 class-conditioned family 已显示 basis 组织方式不是第一落点。

3. 不是立刻做不同 PIA  
   在样本角色没拆开前，换 PIA 只会把问题继续混在一起。

4. 不是直接上多轮闭环  
   当前连第一轮“谁该回流、谁该进最终训练”都还没定义清楚。

---

## 三、当前全部冻结

以下内容全部冻结：

- 当前 `trajectory_representation.py`
- 当前窗口策略
- 当前 `dynamic_gru`
- 当前训练超参
- `T2a default = gamma_main 0.05, smooth_lambda 0.50`
- 当前 `shared single-axis constrained rebasis`
- `center_new` 固定定义为：
  - `orig-train windows ∪ admitted constructive windows` 的全局 pooled center
- 所有 `kNN purity` reference set 固定为：
  - `orig-train-only windows`
- 所有 `margin` 类中心固定为：
  - `orig-train-only window class centers`
- 每个 `dataset × seed` 的类中心只计算一次，并在整轮 `T5` 中冻结复用

当前不改：

- generator
- basis family
- 窗口策略
- 双流 / 回桥
- 多轮 feedback

---

## 四、T5 第一版定义

### 1. 冻结生成器

仍使用冻结后的 `T2a default` 在 `train trajectories` 上生成增强轨迹。

### 2. Constructive pool

- 对象：`window-level`
- gate：`frozen safety gate + radial_gain_window`
- 作用：**只进入 feedback / rebasis**

必须明确：

> `T5` 第一版采用 `radial_gain` 作为 constructive side 的最小 proxy，是因为它在 `T4b` 中已表现出更强的骨架推动能力；这不等于证明 constructive role 的最优 gate 必然是 `radial`，而只是当前第一版 dual-role policy 的固定实现选择。

### 3. Re-center / Re-basis

用 admitted constructive windows 做：

- `re-center`
- `shared single-axis constrained rebasis`

得到：

- `center_new`
- `W_new`

### 4. Rebased augmentation regeneration

在 `center_new + W_new` 下，对**原始 train trajectories**重新生成完整的 rebased augmented trajectories。

### 5. Discriminative pool

- 对象：仍是 `window-level`
- gate：`frozen safety gate + margin_gain_window`
- 作用：**只决定哪些增强窗口写回最终训练用 augmented trajectory**

### 6. 最终 augmented trajectory 的构造

最终 augmented trajectory 的构造**仅发生在 `z_seq` 表示层**：

- 对每条原始 trajectory
- 只有通过 discriminative gate 的窗口位置，使用 `rebased augmented windows`
- 其余窗口保持原始值
- 若一条轨迹没有任何窗口通过 discriminative gate，则不生成它的 augmented twin

必须写死：

> `T5` 第一版中，`discriminative-masked augmented trajectory` 的构造仅发生在 `z_seq` 表示层，所有拼接与 continuity 诊断也仅针对该表示层执行；不允许回写 raw trial，不允许引入 raw-level stitching。

### 7. 拼接连续性诊断

因为最终轨迹是“原始窗口 + 增强窗口”的替换式拼接，所以必须强制输出专门诊断：

- `masked_window_ratio`
- `stitch_boundary_count`
- `stitch_boundary_jump_ratio_mean`
- `stitched_continuity_distortion_ratio`

收益不能建立在明显局部撕裂之上。

### 8. Role overlap 作为核心判读项

`constructive pool` 与 `discriminative pool` 的 overlap 不是附带统计，而是核心判读项。

必须在 `(trial_id, window_index)` 层统计：

- `constructive_count`
- `discriminative_count`
- `overlap_count`
- `role_overlap_jaccard`
- `constructive_only_ratio`
- `discriminative_only_ratio`

若两者高度重合，则不能把结果直接解释为“样本角色分离成功”。

### 9. 方法学口径

`T5` 第一版测的是：

**dual-role sample policy 作为整体对象** 是否优于单一 gate policy。

当前不追求严格证明：

- `radial` 只该用于 constructive
- `margin` 只该用于 discriminative

---

## 五、允许做 / 不允许做

### 允许做

- 复用当前 `T4b` 的 window-level object
- constructive 侧固定用 `radial_gain`
- discriminative 侧固定用 `margin_gain`
- 复用当前 `T3/T4b` 的 shared rebasis 主链
- 新增 stitching diagnostics
- 新增 role-overlap diagnostics

### 不允许做

- 改 generator
- 改 basis family
- 改窗口策略
- 改 `dynamic_gru`
- 改 `T2a/T2b` 参数
- 引入双流 / 回桥
- 引入多轮 feedback
- 把 `T5` 扩成 gate zoo
- 第一版测试 swapped assignment
  - 即不做 `margin -> constructive / radial -> discriminative`

---

## 六、主比较对象

### `SCP1` 主矩阵

- `T3 shared rebasis`
- `T4b window_radial_gate`
- `T4b window_margin_gate`
- `T5 dual-role policy`

背景参考：

- `trajectory baseline`
- `T2a default`

### `NATOPS` 锚点

- `T3 shared rebasis`
- `T4b window_radial_gate`
- `T4b window_margin_gate`
- `T5 dual-role policy`

---

## 七、建议新增模块

- `route_b_unified/trajectory_dual_role_policy.py`
- `scripts/run_route_b_dynamic_dual_role_policy_t5.py`

---

## 八、必须输出的文件

1. `dynamic_dual_role_policy_t5_config_table.csv`
2. `dynamic_dual_role_policy_t5_per_seed.csv`
3. `dynamic_dual_role_policy_t5_dataset_summary.csv`
4. `dynamic_dual_role_policy_t5_constructive_pool_summary.csv`
5. `dynamic_dual_role_policy_t5_discriminative_pool_summary.csv`
6. `dynamic_dual_role_policy_t5_role_overlap_summary.csv`
7. `dynamic_dual_role_policy_t5_basis_shift_summary.csv`
8. `dynamic_dual_role_policy_t5_stitching_summary.csv`
9. `dynamic_dual_role_policy_t5_diagnostics_summary.csv`
10. `dynamic_dual_role_policy_t5_conclusion.md`

`role_overlap_summary` 至少记录：

- `constructive_count`
- `discriminative_count`
- `overlap_count`
- `role_overlap_jaccard`
- `constructive_only_ratio`
- `discriminative_only_ratio`

`stitching_summary` 至少记录：

- `masked_window_ratio`
- `stitch_boundary_count`
- `stitch_boundary_jump_ratio_mean`
- `stitched_continuity_distortion_ratio`

---

## 九、必须回答的问题

1. `T5 dual-role` 是否优于 `T3 shared rebasis`
2. `T5 dual-role` 是否优于 `T4b window_radial_gate` 与 `T4b window_margin_gate`
3. `SCP1` 是否终于同时出现：
   - 更明显的 basis 变化
   - 更好的终点收益
4. constructive 与 discriminative 两类样本是否真的分开了
5. 如果 `T5` 仍不成立，问题是否才真正上移到：
   - basis / shared rebasis 范式
   - 而不只是 sample policy

---

## 十、成功标准

### 弱成立

- `T5 dual-role > T3 shared rebasis`
- 且 stitching diagnostics 没有显示明显局部撕裂

### 中等成立

- `T5 dual-role > max(T4b radial, T4b margin)`
- 且 `role_overlap_jaccard` 不接近完全重合

### 强成立

同时满足：

- `SCP1` 上性能改善
- `SCP1` 上 basis shift 变得非平凡且可解释
- stitching continuity 仍健康
- role overlap 显示 constructive / discriminative 确实承担了不同角色

---

## 十一、一句话执行目标

**冻结当前动态主线已有组件，不再继续单独磨某个 gate 或 basis family；直接把样本角色正式拆开，用 `radial` 负责构造骨架、用 `margin` 负责最终训练筛选，测试 `SCP1` 是否终于能同时获得更好的骨架变化与终点收益。**
