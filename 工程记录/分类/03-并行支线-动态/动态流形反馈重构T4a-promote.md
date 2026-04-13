# 动态流形反馈重构 T4a Promote

更新时间：2026-03-29

## 零、战略定位

当前动态主线已经走到一个新的分化点：

- `NATOPS` 上，`T3 shared rebasis` 已开始成立
- `SelfRegulationSCP1` 上，feedback pool 存在且不小，但 shared basis 基本不动

因此，当前更值得推进的，不再是：

- `T2a/T2b` 参数小修
- 静态线补丁
- 双流/回桥扩张

而是测试：

**shared single basis 是否已经成为 `SCP1` 上的主瓶颈。**

`T4a` 的角色是：

- 一个最小、可归因的 **basis 组织方式升级 probe**
- 不是在宣称 class-wise basis 已被唯一证明

一句话说：

> `T4a` 研究的不是“再调 trajectory operator”，而是“把 shared basis 升级为 class-conditioned basis family 后，`SCP1` 的动态骨架是否终于开始动起来”。

---

## 一、唯一核心问题

**把 `T3` 的 shared single basis 升级为最小的 class-conditioned basis family 后，是否能让 `SCP1` 的 feedback rebasis 真正动起来，并优于当前 `T3 shared rebasis`。**

---

## 二、为什么现在优先做这个

- 不是继续磨 `T2a/T2b`，因为那已经进入边际收益区。
- 不是回静态点主线，因为静态线已退居参考分支。
- 不是先改 feedback pool，因为 `T3` 已说明 `SCP1` 的回流量并不低。
- 不是先上双流/回桥，因为动态单流骨架瓶颈还没打通。

当前最合理的解释是：

**`SCP1` 不是没有 feedback，而是 shared basis 太粗，承载不了它的类条件局部结构。**

---

## 三、当前全部冻结

以下全部冻结：

- 当前 `trajectory_representation.py`
- 当前窗口策略
- 当前 `dynamic_gru`
- 当前训练超参
- 当前 `T2a default = gamma_main 0.05, smooth_lambda 0.50`
- 当前 `T3` 的 safety-filtered feedback pool 规则
- `center_new` 的定义：
  - `orig-train windows ∪ admitted feedback windows` 上的全局 pooled center
- `kNN purity` 的 reference set：
  - `orig-train-only`

当前不改：

- feedback pool 对象
- operator 强度
- T2b saliency 定义
- 终端分类器

---

## 四、第一版方法定义

### 1. 生成器不变

仍用冻结后的 `T2a default` 生成 feedback candidates。

### 2. feedback pool 不变

仍复用 `T3` 的 safety-filtered pool。  
第一版不宣称它是最优 rebasis pool，只把它当作稳定回流入口。

### 3. re-center 不变

仍使用全局 pooled `center_new`。  
第一版不做 per-class center。

### 4. re-basis 升级

不再只学一个 shared basis。  
第一版改成：

- 每个类各学一个 single-axis basis
- 组成最小的 `class-conditioned basis family`

### 5. 训练时的使用方式

对每条原始 `train trajectory`，按其**真类**选择对应类的 basis 重新生成增强。

但必须明确写清：

**`T4a` 第一版当前测的是“class-conditioned basis family 作为训练时增强骨架”的整体效果，不是纯粹隔离 basis 组织方式本身的唯一因果来源。**

也就是说：

- 若 `T4a` 成功，只能先说：
  - `class-conditioned generator` 作为整体对象成立
- 不能直接说：
  - `basis family` 的贡献已被完全单独证明

后续若要更细归因，还需继续区分：

- basis family 的贡献
- 按类增强选择的贡献

### 6. 评估口径

仍与 `T3` 一致：

- feedback pool 只用于重估骨架
- 最终训练仍是：
  - `orig train trajectories + new_aug_from_rebased_family`
- 不把 feedback pool 本身直接当最终训练集答案

---

## 五、任务边界

允许做：

- 复用 `T3` feedback pool
- 复用 `T3` global center_new
- 在 densified pool 上按类分别 fit single-axis basis
- 用 class-conditioned basis family 在原始 train trajectories 上重新增强
- 与 `baseline / T2a default / T3 shared rebasis` 直接对照

不允许做：

- 改 feedback pool 规则
- 改窗口策略
- 改 `dynamic_gru`
- 改 `T2a/T2b` 参数
- 引入 cluster-wise/local basis zoo
- 引入 per-window basis
- 引入多轮 feedback loop
- 引入双流/回桥

---

## 六、主比较对象

1. `trajectory baseline`
2. `T2a default`
3. `T3 shared rebasis`
4. `T4a class-conditioned rebasis`

---

## 七、建议新增模块

建议新增：

- `route_b_unified/trajectory_feedback_rebasis_t4.py`
- `scripts/route_b/run_route_b_dynamic_feedback_rebasis_t4a.py`

其中：

- `trajectory_feedback_rebasis_t4.py`
  - 负责 fit class-conditioned basis family
  - 负责按类生成增强
- `run_route_b_dynamic_feedback_rebasis_t4a.py`
  - 负责完整对照：
    - baseline
    - T2a default
    - T3 shared rebasis
    - T4a class-conditioned rebasis

---

## 八、必须输出的文件

1. `dynamic_feedback_rebasis_t4a_config_table.csv`
2. `dynamic_feedback_rebasis_t4a_per_seed.csv`
3. `dynamic_feedback_rebasis_t4a_dataset_summary.csv`
4. `dynamic_feedback_rebasis_t4a_pool_summary.csv`
5. `dynamic_feedback_rebasis_t4a_basis_family_summary.csv`
6. `dynamic_feedback_rebasis_t4a_diagnostics_summary.csv`
7. `dynamic_feedback_rebasis_t4a_conclusion.md`

`basis_family_summary` 至少记录：

- `dataset`
- `seed`
- `class_id`
- `pooled_window_count_orig`
- `pooled_window_count_feedback`
- `basis_cosine_to_old_shared`
- `basis_angle_to_old_shared`
- `inter_basis_cosine_mean`

---

## 九、必须回答的问题

1. `T4a` 是否优于 `T3 shared rebasis`
2. `SCP1` 是否终于出现比 `T3` 更明确的改善
3. `class-conditioned basis family` 是否在 `SCP1` 上产生了 shared basis 没有的非平凡骨架变化
4. 若 `T4a` 仍不成立，更可能说明问题在：
   - feedback pool 对象
   - 而不是 basis 组织方式

---

## 十、成功标准

满足至少两条才可视为 `T4a` 成立：

- `T4a > T3 shared rebasis` 至少在一个主数据集上成立
- `SCP1` 上出现比 `T3` 更明确的改善
- `SCP1` 上 basis family 出现了**非退化、非暴走、且和性能变化方向一致**的类间差异
- `NATOPS` 不出现明显回退

必须强调：

**仅仅 basis 数量变多，或角度差异增大，不构成成功。**  
成功必须同时满足：

- 变化可解释
- 变化不过激
- 变化与性能改善方向一致

---

## 十一、失败标准

- `T4a` 不优于 `T3`
- `SCP1` 仍没有改善
- basis family 只是形式上“分家”，但没有带来更好结果或更合理结构
- basis family 出现明显退化或暴走

---

## 十二、一句话执行目标

**冻结当前动态主线已有组件，不再继续局部调参；直接用一个最小的 class-conditioned single-axis basis family 替代 `T3` 的 shared basis，测试 `SCP1` 的当前瓶颈是否主要来自 basis 组织方式过粗。**
