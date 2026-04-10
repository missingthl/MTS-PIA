# SCP-Branch v2.1 Trial-based Manifold Anchoring Promote

## 任务名称
`SCP-Branch v2.1: Trial-based Manifold Anchoring`

## 零、战略定位
当前 `SCP` 分支已经完成并吸收了以下事实：

1. `dense z_seq + dynamic_minirocket` 作为 `SCP` backbone 成立
2. `prototype-memory v0` 已证明：
   - 它不是空对象
   - 它更像 `distribution-supported local anchors / local representative states`
   - 但它不是天然更强的类间分离原型
3. `v1b tight anchors + local shaping` 给出小幅正信号：
   - 说明“先收紧对象，再谈 shaping 强度”是对的
4. `v2 geometry refresh` 第一版 smoke 已说明：
   - `balanced acceptance` 机制本身是必要的
   - 但当前 candidate refresh 过于躁动，几乎全部死在 `margin / within` 两门

因此，`v2.1` 的目标不是再改 acceptance，也不是重新打开 replay，而是把 refresh 候选本身改得更稳：

> 从窗口级 anchor mean，升级为 **prototype-conditioned trial mean** 的候选刷新。

---

## 一、唯一核心问题
**在固定 `dense z_seq + dynamic_minirocket + prototype-memory + v1b tight anchors + balanced acceptance` 的前提下，把 prototype refresh 改成 trial-based manifold anchoring 后，`SCP1` 上的 geometry refresh 是否终于能产生可通过的候选更新。**

这一步只问：
- `final_accept_rate` 能不能明显高于当前 `0`
- `margin / within` 两门能否不再系统性失败

当前不问：
- fast local mapping 的最终 F1
- replay/curriculum
- online / test-time adaptation

---

## 二、为什么这是当前最合理入口
1. 不是继续加大 shaping 强度
   当前问题更像 candidate refresh 太躁，不是 local force 太弱。

2. 不是继续磨 accepted anchors
   `v1b` 已经完成了“对象收紧”的第一层任务；现在更像需要把慢层 refresh 本身改稳。

3. 不是改 balanced acceptance
   当前 acceptance 没有错，它只是把粗 refresh 全挡住了。

4. 不是 replay / curriculum
   慢层 refresh 还没站住，继续开快层闭环会再次把因果做脏。

---

## 三、当前全部冻结
以下全部冻结，并且未来 `new v3` 必须共用：

### Backbone 固定
- `raw -> dense z_seq -> dynamic_minirocket`

### 对象定义固定
- `prototype-memory`
- `local representative states`
- `v1b tight anchors`
- `local shaping` 的 admitted object 口径

### 训练/评估口径固定
- train-only
- offline
- single-round
- `val/test` 继续只输入原始 dense trajectory
- 不做 replay
- 不做 curriculum
- 不做 online streaming
- 不做 test-time routing / filtering / calibration
- 不做 NATOPS 兼容扩展
- 不做 dual-stream
- 不开新的 family/container zoo

### Acceptance 固定
- `accept_between`
- `accept_margin`
- `accept_within`
- `final_accept`

---

## 四、v2.1 第一版定义

### 1. 核心改动
`v2.1` 只改一件事：

**prototype candidate refresh 的生成规则**

从：
- `anchor-level mean`

改成：
- `prototype-conditioned trial mean`

### 2. prototype-conditioned trial mean
对每个类 `c`、prototype `p`、trial `τ`，定义：

\[
S_{p,\tau} = \{ z_t \mid z_t \text{ 属于 trial } \tau,\ 且当前被分配给 prototype\ p \}
\]

若：
\[
|S_{p,\tau}| \ge m_{min}
\]

则：
\[
m_{p,\tau} = \frac{1}{|S_{p,\tau}|}\sum_{z \in S_{p,\tau}} z
\]

然后聚合所有有效 trial：

\[
\bar m_p = \frac{1}{T_p}\sum_{\tau} m_{p,\tau}
\]

最后形成候选：

\[
P_p^{candidate} = (1-\alpha)P_p^{old}+\alpha \bar m_p
\]

### 3. 第一版硬参数
第一版不做搜索，写死：

- `alpha = 0.2`
- `m_min = 4`
- `T_p_min = 2`

其中：
- `m_min`
  - 单个 trial 内，某个 prototype 至少要有 4 个支持窗口，才允许贡献一个 `trial mean`
- `T_p_min`
  - 某个 prototype 至少要被 2 个有效 trial 支持，才允许生成 candidate

若：
\[
T_p < T_{p,min}
\]

则该 prototype 本轮：
- `skip_refresh = 1`
- 直接保留旧 reference

这条约束的目的，是防止少量 trial 恰好看起来“很稳”却把 prototype 带偏。

### 4. representative-state refresh
若该 prototype 的 candidate 最终通过 acceptance，则：
- 在该 prototype 的原 member 集合中
- 选取离 `P_p^{candidate}` 最近的真实成员
- 作为新的 representative state

### 5. local direction refresh
若该 prototype 最终通过 acceptance，则：
- 用 refreshed same-class representative
- 和最近的异类 representative
- 重算 local discriminative direction

---

## 五、trial-level 额外诊断
除了当前已有的：
- `trial_proto_support_count`
- `trial_proto_mean_shift_norm`

第一版必须新增：
- `trial_proto_within_dispersion`

定义为：
- 该 trial 内、该 prototype 支持窗口围绕 `m_{p,\tau}` 的平均离散程度

它的作用是区分：
- 该 trial 内确实形成了稳定局部状态
- 还是只是样本很少，均值碰巧不坏

---

## 六、balanced acceptance 继续保留
`v2.1` 不改 acceptance 门控，只改 candidate refresh 的生成。

第一版继续写死：
- 若 `delta_between_prototype_separation <= 0`，拒绝
- 若 `delta_nearest_prototype_margin < 0`，拒绝
- 若 `delta_within_prototype_compactness > 0`，拒绝

被拒绝的对象回退到 refresh 前版本。

---

## 七、允许做 / 不允许做

### 允许做
- prototype-conditioned trial mean
- `m_min` guard
- `T_p_min` guard
- medoid refresh
- local direction recompute
- balanced acceptance / rollback
- 结构指标 before/after 对照
- 代价与重训对照

### 不允许做
- replay
- curriculum
- online / test-time adaptation
- terminal 参数更新
- backbone 参数更新
- 双流
- NATOPS 兼容扩展
- 新 gate / role / container zoo

---

## 八、主比较对象
当前只在 `SCP1` 上做主比较：

1. `v1b shaped memory`
2. `v2 current anchor-mean refresh`
3. `v2.1 trial-based manifold anchoring`

背景参考可保留：
- `v0 prototype-memory`
- `same_backbone_no_shaping`

---

## 九、必须输出的文件
- `scp_branch_v21_config_table.csv`
- `scp_branch_v21_per_seed.csv`
- `scp_branch_v21_dataset_summary.csv`
- `scp_branch_v21_prototype_update_summary.csv`
- `scp_branch_v21_direction_summary.csv`
- `scp_branch_v21_acceptance_summary.csv`
- `scp_branch_v21_trial_anchor_summary.csv`
- `scp_branch_v21_conclusion.md`

## 九点五、数据集矩阵
当前 `v2.1` 的数据集矩阵分两层：

### 主矩阵
- `selfregulationscp1`

### EEG 扩展矩阵
- `seed1`
- `seediv`
- `seedv`

判读原则写死为：
- `SCP1` 仍是当前 `SCP` 分支的主判断对象
- `SEED family` 当前用于检验 trial-based anchoring 在 EEG 类数据上的兼容性与迁移潜力
- 第一轮 smoke 不要求 `SEED family` 直接给出与 `SCP1` 同级别的方法结论，但必须输出 acceptance / cost / direction 稳定性结果

其中 `scp_branch_v21_acceptance_summary.csv` 至少必须记录：
- `accept_between`
- `accept_margin`
- `accept_within`
- `final_accept`

其中 `scp_branch_v21_trial_anchor_summary.csv` 至少必须记录：
- `class_id`
- `prototype_id`
- `trial_id`
- `trial_proto_support_count`
- `trial_proto_mean_shift_norm`
- `trial_proto_within_dispersion`

---

## 十、必须回答的问题
1. `final_accept_rate` 是否从当前 `0` 明显上升
2. `accept_margin_rate` 是否明显高于当前 `0`
3. `accept_within_rate` 是否明显高于当前 `0`
4. `between_prototype_separation` 是否仍能改善
5. `update_to_retrain_ratio` 是否仍低于或接近 `1.0`
6. `trial_proto_within_dispersion` 是否能解释哪些 prototype 更容易通过 acceptance

---

## 十一、成功标准

### 弱成立
- `final_accept_rate > 0`
- 且 `accept_margin_rate`、`accept_within_rate` 明显高于当前 `0`

### 中等成立
- `final_accept_rate` 明显提升
- `between` 有改善
- `margin / within` 不恶化
- `update_to_retrain_ratio <= 1.0`

### 强成立
同时满足：
- `final_accept_rate` 达到稳定可观水平
- `between ↑`
- `margin ↑`
- `within` 保住
- `update_to_retrain_ratio < 1.0`
- 足以支撑进入 `new v3 fast local mapping`

---

## 十二、与下一阶段的关系
`v2.1` 的目标不是直接追 F1，而是把慢层几何刷新变得稳健可信。

一旦 `v2.1` 成立，下一步再进入：

## `new v3: Fast Local Sample Generation / Local Mapping`

也就是：
- 固定当前 `v2.1` 刷新后的局部几何参考
- 快速为当前窗口生成终端更容易读取的局部样本视图

所以：
- `v2.1` 是慢层强化
- `new v3` 是快层生成

---

## 十三、一句话执行目标
**把 `v2` 的 prototype refresh 从“窗口级 anchor mean 牵引”升级为“prototype-conditioned trial mean anchoring”，先让 geometry refresh 变得稳定可信，再进入后续 fast local mapping。**
