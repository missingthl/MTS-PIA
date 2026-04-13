# 动态流形反馈重构 T6b Promote

更新时间：2026-03-30

## 零、战略定位

当前动态主线已经从：

- 对象成立
- 算子成立
- 骨架开始可塑

推进到：

**样本角色定义阶段**

当前下一步不再优先推进：

- `T2a / T2b` 参数细磨
- `T4a` basis family 扩张
- `T6a-1` unified gate 继续打磨
- 双流 / 回桥 / 多轮 feedback

当前主推进方向明确为：

- `Constructive = window_safety_only`
- `Discriminative = local_kNN_margin`

一句话说：

> `T6b` 不再问“单一 unified gate 谁更好”，而是问“更稳的骨架脚手架样本 + 更局部的判别样本，是否能一起把 SCP1 打开”。

---

## 一、唯一核心问题

**当 constructive side 固定为 `window_safety_only`、discriminative side 固定为 `local_kNN_margin` 后，`SCP1` 是否终于能同时出现更好的骨架变化与更好的终点收益。**

---

## 二、为什么这是当前最合理入口

1. 不是继续磨 `T4b`  
`T4b` 已经说明 window-level object 有价值，但单一 global gate 无法同时兼顾“推骨架”和“提分数”。

2. 不是继续磨 `T6a-1`  
`T6a-1` 已经说明 `local_kNN_margin` 作为 unified gate 偏负，更像不适合同时承担 constructive role。

3. 不是继续扩 `T4a`  
`SCP1` 上 class-conditioned family 还没有被真正触发，现在继续扩 basis family 会把因果再次做脏。

4. 不是立刻做不同 PIA 或多轮闭环  
当前第一轮样本角色策略还没站稳，直接上更大系统只会放大偏差。

---

## 三、当前全部冻结

以下全部冻结：

- 当前 `trajectory representation`
- 当前窗口策略
- 当前 `dynamic_gru`
- 当前训练超参
- `T2a default = gamma_main 0.05, smooth_lambda 0.50`
- 当前 `shared single-axis constrained rebasis`
- 当前 `window-level object`
- `center_new = orig-train windows ∪ admitted constructive windows` 的全局 pooled center
- `kNN purity` reference set 固定为 `orig-train-only windows`
- `local_kNN_margin` 的 reference set 固定为 `orig-train-only windows`
- same-class 近邻统计必须排除当前窗口自身
- 所有 reference stats 在每个 `dataset × seed` 中只计算一次并冻结复用

当前不改：

- generator
- basis family
- 窗口策略
- 双流 / 回桥
- 多轮 feedback

---

## 四、T6b 第一版定义

### 1. 冻结生成器

仍使用冻结后的 `T2a default` 在 `train trajectories` 上生成增强轨迹。

### 2. Constructive pool

- 对象：`window-level`
- gate：`window_safety_only`
- 作用：**只进入 feedback / rebasis**

当前必须明确：

- `window_safety_only` 只是 constructive side 的第一版稳妥脚手架实现
- 不等于它已被证明是最终最优 constructive gate

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

- 对象：`window-level`
- gate：`safety gate + local_kNN_margin_gain`
- 作用：**只决定哪些 rebased augmented windows 被写回最终 augmented trajectory**

`local_kNN_margin` 第一版定义为：

\[
\mathrm{local\_margin}(z)
=
\mathrm{mean\,dist\ to\ }k\text{-NN}_{\mathrm{diff}}
-
\mathrm{mean\,dist\ to\ }k\text{-NN}_{\mathrm{same}}
\]

并且写死：

- 所有近邻查询统一相对于 `orig-train-only windows`
- same-class 近邻必须排除当前窗口自身
- 不允许把增强窗口、constructive pool、rebasis 后样本混入 reference set

于是：

\[
\mathrm{local\_kNN\_margin\_gain}
=
\mathrm{local\_margin}(z^{rebased\_aug})
-
\mathrm{local\_margin}(z^{orig})
\]

第一版 admission 规则固定为：

1. 先过 `safety gate`
2. 再在 safe windows 中按类保留 `local_kNN_margin_gain >= classwise median`

### 6. Coverage guard

对每个 `dataset × seed × class`，固定：

\[
\mathrm{coverage\_threshold} = \max(8,\lceil 0.05 \cdot \mathrm{safe\_window\_count\_class}\rceil)
\]

若：

- `admitted_window_count_class < coverage_threshold`

则记为：

- `low_coverage_flag = 1`
- `effective_trigger = 0`

这条是硬约束，不做阈值搜索。

必须明确：

- `low_coverage` 的判读是 **按类判定、按数据集汇总**
- 结论中不能只看数据集平均 coverage
- 若关键类别长期 `low_coverage`，则不能把该数据集解释为“discriminative gate 已被有效触发”

### 7. 最终 augmented trajectory 的构造

最终 augmented trajectory 只在 `z_seq` 表示层构造：

- 只有通过 discriminative gate 的窗口位置，使用 `rebased augmented windows`
- 其余窗口保持原始值
- 若一条轨迹没有任何窗口通过 discriminative gate，则不生成 augmented twin

硬约束：

- 所有拼接只发生在 `z_seq` 层
- 不允许回写 raw
- 不允许引入 raw-level stitching

### 8. Stitching 诊断

必须额外输出：

- `masked_window_ratio`
- `stitch_boundary_count`
- `stitch_boundary_jump_ratio_mean`
- `stitched_continuity_distortion_ratio`

收益不能建立在明显局部撕裂之上。

### 9. Role overlap

`constructive pool` 与 `discriminative pool` 的 overlap 必须作为核心判读项，至少输出：

- `constructive_count`
- `discriminative_count`
- `overlap_count`
- `role_overlap_jaccard`
- `constructive_only_ratio`
- `discriminative_only_ratio`

如果 overlap 接近完全重合，就不能把结果解释成“角色分离成功”。

### 10. 方法学口径

`T6b` 第一版测的是：

**dual-role sample policy 作为整体对象** 是否优于当前单一 gate policy。

当前不追求严格证明：

- `window_safety_only` 天然只该用于 constructive
- `local_kNN_margin` 天然只该用于 discriminative

---

## 五、允许做 / 不允许做

### 允许做

- constructive side 固定用 `window_safety_only`
- discriminative side 固定用 `local_kNN_margin`
- 复用当前 `T3 / T4b / T6a-1` 的 window-level object、shared rebasis、evaluator 主链
- 新增 role-overlap 与 stitching 诊断

### 不允许做

- 改 generator
- 改 basis family
- 改窗口策略
- 改 `dynamic_gru`
- 改 `T2a / T2b` 参数
- 引入双流 / 回桥
- 引入多轮 feedback
- 把 `T6b` 扩成 gate zoo
- 第一版不做 swapped assignment
  - 不测试 `constructive=local_kNN`
  - 不测试 `discriminative=safety_only`

---

## 六、主比较对象

### `SCP1` 主矩阵

- `trajectory baseline`
- `T2a default`
- `T3 shared rebasis`
- `T4b window_radial_gate`
- `T4b window_margin_gate`
- `T5 dual-role`
- `T6a-1 local_kNN_margin_unified`
- `T6b safety_constructive + local_kNN_discriminative`

### `NATOPS` 锚点

- `T3 shared rebasis`
- `T4b window_radial_gate`
- `T4b window_margin_gate`
- `T6a-1 local_kNN_margin_unified`
- `T6b safety_constructive + local_kNN_discriminative`

其中：

- `T4b window_radial_gate` 必须保留
- 因为它是 NATOPS 上当前较强参考之一
- 若 `T6b` 最后不如它，更容易判断 `T6b` 是否只是 `SCP1` 型任务特化策略

---

## 七、建议新增模块

- `route_b_unified/trajectory_dual_role_policy_t6b.py`
- `scripts/run_route_b_dynamic_dual_role_policy_t6b.py`

若实现上最小改动，也允许直接复用现有 dual-role builder，但 runner 与输出命名必须独立，不要把 `T5` 与 `T6b` 混成同一入口。

---

## 八、必须输出的文件

1. `dynamic_dual_role_policy_t6b_config_table.csv`
2. `dynamic_dual_role_policy_t6b_per_seed.csv`
3. `dynamic_dual_role_policy_t6b_dataset_summary.csv`
4. `dynamic_dual_role_policy_t6b_constructive_pool_summary.csv`
5. `dynamic_dual_role_policy_t6b_discriminative_pool_summary.csv`
6. `dynamic_dual_role_policy_t6b_role_overlap_summary.csv`
7. `dynamic_dual_role_policy_t6b_class_coverage_summary.csv`
8. `dynamic_dual_role_policy_t6b_basis_shift_summary.csv`
9. `dynamic_dual_role_policy_t6b_stitching_summary.csv`
10. `dynamic_dual_role_policy_t6b_diagnostics_summary.csv`
11. `dynamic_dual_role_policy_t6b_conclusion.md`

---

## 九、必须回答的问题

1. `T6b` 是否优于 `T3`
2. `T6b` 是否优于 `T4b window_margin_gate`
3. `T6b` 是否优于 `T6a-1 local_kNN_margin_unified`
4. `T6b` 相对 `T4b window_radial_gate` 处在什么位置
5. `SCP1` 是否终于同时出现：
   - 更好的 basis 变化
   - 更好的终点收益
6. constructive / discriminative 两类样本是否真的分开了
7. 若 `T6b` 仍不成立，问题是否才真正上移到：
   - basis / shared rebasis 范式
   - 或 generator 层

---

## 十、成功标准

### 弱成立

- `T6b > T3`
- 且 stitching 没有明显恶化

### 中等成立

- `T6b > max(T4b window_margin_gate, T6a-1 local_kNN_margin_unified)`
- 且 `role_overlap_jaccard` 不接近完全重合

### 强成立

同时满足：

- `SCP1` 上终点提升
- `SCP1` 上 basis shift 非平凡且可解释
- stitching continuity 仍健康
- 主要类别不是 `low_coverage`
- `NATOPS` 不出现明显回退

---

## 十一、一句话执行目标

**冻结当前动态主线已有组件，不再继续单独磨某个 unified gate 或 basis family；直接让 constructive side 退回更稳的 `window_safety_only`，让 discriminative side 升级为 `local_kNN_margin`，测试 `SCP1` 是否终于能同时获得更好的骨架变化与终点收益。**
