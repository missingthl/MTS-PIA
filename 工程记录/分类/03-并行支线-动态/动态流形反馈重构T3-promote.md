# 动态流形反馈重构 T3 Promote

更新时间：2026-03-29

## 零、战略定位

当前已经不适合继续把主精力放在：

- `T2a` 的局部倍率收口
- 或 `T2b-0` 的固定规则细磨

当前更值得推进的是：

**T3：Dynamic Manifold Feedback Re-basis Probe**

它研究的不是：

- 轨迹增强还能不能再多涨一点

而是：

**增强结果能否反过来改变动态分支的参考中心与共享 basis，从而开始真正重构骨架。**

一句话说：

> `T3` 不是“再调 operator”，而是“测试增强是否已经能反馈并重塑动态流形骨架”。

---

## 一、为什么现在适合进入 T3

当前已经满足三个前提：

1. `T0` 已成立  
   - 轨迹对象已正式成立

2. `T2a` 已成立，`T2b-0` 也已有明确信号  
   - 轨迹增强算子已不再只是空想

3. 已有可冻结的稳定生成器  
   第一版可直接冻结：
   - 当前 `trajectory representation`
   - 当前窗口策略
   - `dynamic_gru`
   - `T2a default = gamma_main 0.05, smooth_lambda 0.50`

所以现在完全可以停止继续围绕局部倍率打磨，直接问：

- 增强是否开始改骨架

---

## 二、任务目标

`T3` 第一版只回答一个问题：

**冻结当前动态主线默认生成器后，经过过滤的增强轨迹回流，能否重估更好的参考中心与共享 basis，并让同一条增强链在新骨架上变得更有效。**

本轮不是：

- `T2a` 新一轮调参
- `T2b` 新规则搜索
- 双流融合
- bridge 回流
- trajectory 多轮 feedback
- metric learning 大系统

本轮只做：

1. 冻结当前默认生成器
2. 构造 `filtered trajectory feedback pool`
3. `re-center`
4. `re-basis`
5. 在**原始 trajectory** 上重新增强并评估

---

## 三、第一版生成器固定

第一版生成器写死为：

- 当前 `trajectory_representation.py`
- 当前窗口策略
- `T2a default`
  - `gamma_main = 0.05`
  - `smooth_lambda = 0.50`
- `dynamic_gru`

当前**不**用 `T2b-0` 当主生成器。  
原因：

- `T2a default` 更稳、更干净、更容易归因
- `T2b-0` 在 `SelfRegulationSCP1` 上还不足以当默认骨架生成器

---

## 四、T3 第一版主逻辑

### Step 1：冻结生成器生成 feedback candidates

- 只在 `train trajectories` 上生成增强轨迹
- `val/test` 永远不参与生成，也不参与 feedback

### Step 2：构造 filtered feedback pool

硬约束：

- `Train Pool ≠ Feedback Pool`
- 不是所有增强轨迹都能回流

第一版 feedback pool 当前只定义为：

- **safety-filtered pool**

也就是说：

- 它的职责是避免明显坏样本回流
- 当前不宣称它已经是最优的 `re-basis pool`

第一版建议的 first-pass safety gate：

对每个增强轨迹，至少计算：

- `continuity_distortion_ratio`
- `traj_mean_orig`
- `traj_mean_aug`
- `kNN purity orig`
- `kNN purity aug`

并采用最小硬规则：

1. `aug` 的近邻多数类必须仍等于原类
2. `purity_drop = purity_orig - purity_aug <= 0.10`
3. `continuity_distortion_ratio` 不超过该类候选分布的 `q75`

### Step 3：re-center

第一版不必一上来宣称严格黎曼中心。  
工程上先定义为：

- 在当前 trajectory 表示空间里
- 用 `orig train windows + admitted feedback windows`
- 重估一个新的全局 pooled center

记为：

- `center_new`

硬约束：

> `T3` 第一版的 `center_new` 统一定义为：在 `orig-train windows ∪ admitted feedback windows` 的联合集合上计算的全局 pooled center；当前不做 per-trajectory / per-class center 版本。

### Step 4：re-basis

在 densified pool 上重新学习 shared basis：

- 输入：
  - `orig train trajectories`
  - `filtered feedback trajectories`
- 输出：
  - `W_new`

第一版保持：

- shared basis
- single-axis
- 不进入 classwise / local basis family

当前必须明确：

> `T3` 第一版采用的是 **受限 shared-basis re-fit** 口径，而不是完全自由重学。

也就是说：

- 仍保持单轴
- 仍保持全局共享 basis
- 不引入 classwise/local family
- 必须强制输出：
  - `basis_cosine_to_old`
  - `basis_angle_proxy`

这些指标的意义不是装饰，而是：

- 防止 basis 暴走被误判为成功

### Step 5：在原始 trajectory 上重新增强并评估

这是第一版最重要的口径约束：

**不要把 feedback pool 本身直接并进最终训练集当答案。**

而要这样评估：

1. 用冻结生成器生成 feedback pool  
2. 得到 `center_new + W_new`  
3. 用 `center_new + W_new` 在**原始 train trajectories** 上重新生成增强  
4. 再走同一条 `dynamic_gru` 训练链

也就是：

- 最终训练仍然是 `orig + new_aug`
- 不是 `orig + feedback_pool + new_aug`

这样测到的是：

- 新骨架是否更适合当前增强链

而不是：

- 样本变多了所以涨分

---

## 五、方法学口径必须写清

`T3` 第一版不是完整动态闭环系统。  
它的本质是：

**frozen generator + safety-filtered feedback pool + global pooled re-centering + constrained shared-basis re-fit probe**

也就是说：

- 它是动态分支上的 feedback re-basis probe
- 不是多轮自进化系统
- 不是完整动态流形学习终态

---

## 六、主比较对象

主比较固定为：

1. `trajectory baseline`
2. `T2a default`
3. `T3 filtered rebasis`

可选负对照：

4. `T3 all-aug replay`

说明：

- `T3 all-aug replay` 不是第一版必须项
- 若实现成本低，可以保留
- 若会拖慢主线，可以先不做

---

## 七、第一版必须冻结的内容

全部冻结：

- 当前窗口策略
- 当前 `trajectory_representation.py`
- 当前 pooled-train-windows -> shared basis 方式
- `axis_count = 1`
- 当前 `dynamic_gru`
- 所有训练超参
- `val/test` 永远只用原始 trajectory
- `T2a default = gamma 0.05 / smooth 0.50`

一句话：

- 本轮不调参数
- 本轮只测 feedback re-basis 是否有骨架价值

---

## 八、kNN purity 的 reference set 硬约束

所有 `kNN purity` 统计必须统一相对于：

- **orig-train-only reference set**

计算。

硬约束：

- 不允许将增强候选混入 reference set
- 不允许将 admitted feedback pool 混入 reference set
- 不允许将 rebasis 后样本混入 reference set

这条约束的目的，是防止 purity 被 feedback 样本自我支撑，从而失真。

---

## 九、建议新增模块

建议新增：

- `route_b_unified/trajectory_feedback_pool.py`
- `route_b_unified/trajectory_feedback_rebasis.py`
- `scripts/route_b/run_route_b_dynamic_feedback_rebasis_t3.py`

职责建议：

### `trajectory_feedback_pool.py`

- 输入：
  - orig train trajectories
  - `T2a default` 生成的增强 trajectories
- 输出：
  - admitted feedback pool
  - pool summary

### `trajectory_feedback_rebasis.py`

- 输入：
  - orig trajectories
  - feedback pool
- 输出：
  - `center_new`
  - `W_new`
  - basis drift summary

### `run_route_b_dynamic_feedback_rebasis_t3.py`

- 跑：
  - `NATOPS + SelfRegulationSCP1`
  - `3 seeds`
- 输出正式对照表和结论

---

## 十、必须输出的文件

1. `dynamic_feedback_rebasis_t3_config_table.csv`
2. `dynamic_feedback_rebasis_t3_per_seed.csv`
3. `dynamic_feedback_rebasis_t3_dataset_summary.csv`
4. `dynamic_feedback_rebasis_t3_pool_summary.csv`
5. `dynamic_feedback_rebasis_t3_basis_shift_summary.csv`
6. `dynamic_feedback_rebasis_t3_diagnostics_summary.csv`
7. `dynamic_feedback_rebasis_t3_conclusion.md`

建议字段：

### `dynamic_feedback_rebasis_t3_pool_summary.csv`

- `dataset`
- `seed`
- `candidate_count`
- `accepted_count`
- `accept_rate`
- `class_balance_proxy`

### `dynamic_feedback_rebasis_t3_basis_shift_summary.csv`

- `dataset`
- `seed`
- `center_shift_norm`
- `basis_cosine_to_old`
- `basis_angle_proxy`
- `pooled_window_count_orig`
- `pooled_window_count_feedback`

### `dynamic_feedback_rebasis_t3_dataset_summary.csv`

- `dataset`
- `baseline_macro_f1_mean/std`
- `t2a_default_macro_f1_mean/std`
- `t3_rebasis_macro_f1_mean/std`
- `best_mode`

---

## 十一、必须回答的问题

1. `T3 filtered rebasis` 是否优于 `T2a default`
2. `SelfRegulationSCP1` 是否比 `NATOPS` 更受益于 feedback re-basis
3. feedback pool 是否稳定，不是极少数偶然点
4. 新中心与新 basis 是否发生了可解释变化
5. 如果失败，更像说明：
   - 生成器信息量不足
   - feedback pool 对象不对
   - 还是 shared basis 已经过粗

---

## 十二、成功标准

`center_shift_norm / basis_shift` 本身**不能单独算成功**。

骨架变化不是目标。  
**骨架变化 + 更好结果 + 更稳结构** 才是目标。

因此，满足以下主条件之一：

1. `T3 rebasis > T2a default` 至少在一个主数据集上成立
2. `SelfRegulationSCP1` 上出现比 `T2a default` 更明确的改善

并且同时满足至少一条辅助条件：

- feedback pool 录取稳定，不是偶然小样本
- `center_shift_norm / basis_shift` 显示骨架发生了非平凡但不过激的变化
- 结构诊断显示新骨架下的增强链更健康

失败标准：

- `T3` 不优于 `T2a default`
- feedback pool 极不稳定
- basis 几乎不变或暴走
- 结果更像“样本变多”而不是“骨架变好”

---

## 十三、当前明确不做的事

- 不做多轮 feedback
- 不做 `T2a/T2b` 新一轮调参
- 不做 dual-stream
- 不做 bridge/raw 回流
- 不做 classwise/local rebasis
- 不做 metric learning 大系统

---

## 十四、建议执行顺序

### Step 1

冻结当前 `T2a default` 生成器与整个 trajectory 主干。

### Step 2

生成 feedback candidates，并在 `orig-train-only reference set` 上计算 purity / continuity 指标。

### Step 3

构造 `safety-filtered feedback pool`。

### Step 4

在 `orig-train windows ∪ admitted feedback windows` 上重估：

- `center_new`
- `W_new`

### Step 5

用 `center_new + W_new` 在**原始 train trajectories** 上重新生成增强，并走同一条 `dynamic_gru` 训练链。

### Step 6

输出正式结果与结论文档，明确判断：

- feedback rebasis 是否成立
- `SCP1` 是否更受益
- 是否值得继续沿 `T3` 推进

---

## 十五、一句话执行目标

**冻结当前动态主线默认生成器，用 filtered trajectory feedback pool 做一次动态分支上的全局 pooled re-center + constrained shared-basis re-fit probe，并在原始 trajectory 上重新增强和评估，判断增强结果是否已经能真正反过来改变共享 basis 和参考几何。**
