# 动态流形反馈重构 T7a Promote

更新时间：2026-03-30

## 零、战略定位

当前动态主线已经完成了一轮系统排查：

- `object` 已从 whole-trajectory 升级到 window-level
- `radial / margin / local-kNN` 都已探过
- `unified / dual-role` 都已探过
- `coverage / stitching` 已排除工程伪效应

因此，当前下一步不再优先继续磨 gate，而是直接测试：

> **shared rebasis 容器本身是否太粗，接不住 SCP1 已经存在的样本价值。**

`T7a` 的角色是：

- 一个最干净的 **容器升级实验**
- 不是 gate zoo
- 不是多轮闭环
- 不是新的 dual-role 实验

---

## 一、唯一核心问题

**在 window-level constructive pool、冻结生成器、冻结安全脚手架都不变的前提下，把 rebasis 容器从 `shared single-axis` 升级为 `class-conditioned single-axis family` 后，`SCP1` 是否终于能把已有样本价值接住。**

---

## 二、为什么现在优先做这个

1. 不是继续磨 `T4b / T6a-1`
- 这些已经说明：window-level object 有价值，但 gate 本身不是唯一主瓶颈。

2. 不是继续扩 `T4a`
- 当前不是要扩更多 family 变体，而是先做一个最干净的容器升级实验。

3. 不是立刻做不同 PIA
- 样本对象、回流方式、容器三件事还没彻底拆清，现在换 PIA 会把因果继续混在一起。

4. 不是直接上多轮
- 单轮里的“容器是否接得住”都还没判清，多轮只会放大偏差。

---

## 三、当前全部冻结

以下全部冻结：

- `trajectory_representation`
- 当前窗口策略
- `dynamic_gru`
- 当前训练超参
- `T2a default = gamma_main 0.05, smooth_lambda 0.50`
- `window-level object`
- `constructive pool = window_safety_only`
- 所有 purity / kNN reference 继续只来自 `orig-train-only windows`
- `val/test` 继续只用原始 trajectory
- 不做双流
- 不做回桥
- 不做多轮
- 不改 raw 侧

---

## 四、T7a 第一版定义

### 1. 生成器冻结

仍使用冻结后的 `T2a default` 在 `train trajectories` 上生成增强。

### 2. Constructive side 固定

第一版 constructive pool 固定为：

- `window-level object`
- `window_safety_only`

它只负责：

- `feedback / rebasis`

当前不要求它最判别，只要求：

- 安全
- 覆盖支撑域
- 不给容器喂脏东西

### 3. Re-center 继续全局

为隔离变量，`center_new` 继续沿用当前 `T3/T4b` 口径：

- `orig-train windows ∪ admitted constructive windows`

上的 **global pooled center**

第一版不做 per-class center。

### 4. Re-basis 升级为 class-conditioned family

把：

- `shared single-axis rebasis`

改成：

- `class-conditioned single-axis rebasis family`

即：

- 对每个类 `c`
- 用该类的 `orig-train windows + admitted constructive windows` 作为正类
- 用所有其它类 `k != c` 的 `orig-train windows + admitted constructive windows` 作为负类
- 在同一个 `center_new` 下，按 **One-vs-Rest (OvR)** 逻辑求解该类轴 `W_c_new`

硬约束：

- 仍然是 `single-axis`
- 仍然是显式、受限、可审计的线性 family
- 不进入 cluster-wise / local-basis zoo

必须明确：

> 如果只把类别 `c` 的窗口单独喂给容器求轴，它会退化成无监督类内主方向。  
> 因此 `T7a` 第一版中，`W_c_new` 必须强制使用 train-only 的 OvR 判别逻辑求解，而不是类内无监督拟合。

### 5. 最终训练口径

第一版 **先不引入 dual-role**。  
直接做最干净的容器 probe：

- 用 `center_new + W_c_new`
- 对每条原始 train trajectory，按其真类 `c` 选择对应容器
- 生成 **完整 rebased augmented trajectory**
- 最终训练仍是：
  - `orig trajectories + full rebased augmented trajectories`

也就是说：

- 当前不做 window mask 回写
- 当前不做 stitching policy
- 当前不做 discriminative gate

硬约束：

> 按真类选择 `W_c_new` 只发生在训练阶段的增强生成。  
> `val/test` 仍然只输入原始 trajectory，不做 class-conditioned test-time routing。

### 6. 方法学口径

`T7a` 第一版测的是：

- **class-conditioned rebasis container** 作为整体训练时增强容器的效果

当前不追求严格证明：

- class-conditioned container 已被唯一证明正确
- 或 true-label conditioned generator 的贡献已被完全剥离

---

## 五、允许做 / 不允许做

### 允许做

- 复用当前 `window-level constructive pool`
- 复用当前 `window_safety_only`
- 保持 `shared global center`
- 只把 axis 容器升级为 `class-conditioned single-axis family`
- 用完整 rebased augmented trajectory 做统一训练

### 不允许做

- 改 generator
- 改窗口策略
- 改 `dynamic_gru`
- 改 `T2a/T2b` 参数
- 同时引入 dual-role
- 同时引入 local-kNN / margin 作为最终训练 gate
- 同时引入 class-conditioned center
- 引入双流 / 回桥 / 多轮

---

## 六、主比较对象

### `SCP1` 主矩阵

- `trajectory baseline`
- `T2a default`
- `T3 shared rebasis`
- `T4b window_radial_gate`
- `T6b dual-role policy`
- `T7a class-conditioned rebasis`

### `NATOPS` 锚点

- `T3 shared rebasis`
- `T4b window_radial_gate`
- `T7a class-conditioned rebasis`

`NATOPS` 当前只作 stability reference，不与 `SCP1` 同权。

---

## 七、建议新增模块

- `route_b_unified/trajectory_feedback_rebasis_t7.py`
- `scripts/route_b/run_route_b_dynamic_feedback_rebasis_t7a.py`

---

## 八、必须输出的文件

- `dynamic_feedback_rebasis_t7a_config_table.csv`
- `dynamic_feedback_rebasis_t7a_per_seed.csv`
- `dynamic_feedback_rebasis_t7a_dataset_summary.csv`
- `dynamic_feedback_rebasis_t7a_constructive_pool_summary.csv`
- `dynamic_feedback_rebasis_t7a_class_coverage_summary.csv`
- `dynamic_feedback_rebasis_t7a_basis_family_summary.csv`
- `dynamic_feedback_rebasis_t7a_diagnostics_summary.csv`
- `dynamic_feedback_rebasis_t7a_conclusion.md`

---

## 九、必须强制输出的三层信息

### 1. 容器层

至少输出：

- `class_id`
- `pooled_window_count_orig`
- `pooled_window_count_feedback`
- `pooled_window_count_total`
- `ovr_negative_window_count`
- `basis_cosine_to_old_shared`
- `basis_angle_to_old_shared`
- `inter_basis_cosine_mean`
- `inter_basis_cosine_min`
- `inter_basis_cosine_max`

必须明确：

> 不能只看 `inter_basis_cosine_mean`。  
> 第一版必须同时看 `inter_basis_cosine_min / max`，否则会掩盖“某些类分开、某些类仍塌在一起”的情况。

### 2. pool 层

至少输出：

- `class_id`
- `safe_window_count_class`
- `admitted_window_count_class`
- `coverage_ratio_class`
- `coverage_threshold_class`
- `low_coverage_flag`

### 3. 终点层

至少输出：

- `test_macro_f1`
- `delta_vs_t3`
- `delta_vs_t4b_radial`
- `delta_vs_t6b`
- `delta_vs_t2a_default`

---

## 十、必须回答的问题

1. `T7a` 是否优于 `T3 shared rebasis`
2. `T7a` 是否优于当前 `SCP1` 最强参考
3. `SCP1` 上 class-conditioned basis family 是否终于不再塌回 shared-like
4. `SCP1` 上是否同时出现：
   - 更好的分数
   - 更清晰的 basis family 分化
5. 若 `T7a` 成立，是否值得进入：
   - `T7b: class-conditioned container + dual-role policy`
6. 若 `T7a` 失败，是否说明问题已开始上移到：
   - 显式线性 rebasis 容器本身

---

## 十一、成功标准

### 弱成立

- `T7a > T3` 于 `SCP1`

### 中等成立

- `T7a >` 当前 `SCP1` 最强参考
- 且 basis family 不再塌成 shared-like

### 强成立

同时满足：

- `SCP1` 上分数稳定提升
- 各类 basis 真正分开
- class coverage 健康
- `NATOPS` 不明显被打坏

---

## 十二、后续分叉

### 如果 `T7a` 成功

进入：

- `T7b`

也就是在 **class-conditioned 容器** 上，再把 dual-role sample policy 接回来。

### 如果 `T7a` 失败

更稳的下一步是：

- `T8: Dual-Stream Nonlinear Fusion Probe`

因为这时更可能说明：

- 问题不只是 shared 容器太粗
- 而是显式线性 rebasis 容器本身开始到顶

---

## 十三、一句话执行目标

**先做一个最干净的容器升级实验：冻结生成器、冻结 window-level 对象、冻结安全脚手架，只把 shared rebasis 改成 class-conditioned rebasis family，优先在 `SCP1` 上验证“容器是否终于能接住已有样本价值”。**
