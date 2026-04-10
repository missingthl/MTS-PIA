# SCP-Branch v2 Geometry Refresh Promote

## 任务名称
`SCP-Branch v2: Global/Local Geometry Refresh`

## 零、战略定位
当前 `SCP` 分支已经完成三件事：

1. `dense z_seq + dynamic_minirocket` 作为 `SCP` backbone 成立  
2. `prototype-memory v0` 已证明：
   - 它不是空对象
   - 它更像 `distribution-supported local anchors / local representative states`
   - 但它还不是天然更强的类间分离原型
3. `v1b tight anchors + local shaping` 已给出小幅正信号：
   - 说明“先收紧对象，再谈 shaping 强度”是对的
   - 但当前提升仍弱，尚未把 `SCP1` 真正打通

与此同时，原来的 `v2 single replay` 已经给出负信号：
- 它把快层结果过早写回训练序列
- 引入 stitching / boundary artifact
- 更像是把“几何更新问题”误写成了“样本回放问题”

因此，从这一阶段开始，`SCP` 分支应正式分成两层：

- **慢层**：阶段性刷新局部流形参考
- **快层**：在刷新后的参考几何上快速生成局部样本映射

在这个重构下：

- 原 `v2 replay` 降级为过渡性工程探针
- 原 `v3 closed-form local update` 上升并重命名为 **新 `v2 geometry refresh`**

一句话说：

> 新 `v2` 不再负责 replay，而是负责在 train-only 条件下，对当前个体的局部状态空间做一次低成本、阶段性的几何刷新。

### 吸收自 T 系列的硬约束
这一步不是凭空起新线，而是明确吸收以下已成立判断：

1. `T0`
   - 样本对象必须是动态轨迹对象，不能退回静态点。
   - 因此 `v2` 继续以 `dense z_seq` 的 local windows 为基本工作对象。

2. `T2a / T2b`
   - PIA 的价值在于轨迹几何干预，而不是 raw 级拼接或点噪声式增强。
   - 因此 `v2` 只做 train-only 的局部几何刷新，不做 raw 回写，不做 replay，不做 test-time 适配。

3. `T3`
   - reference set 必须干净，不能让增强/更新对象自我支撑。
   - 因此 `v2` 的局部参考、prototype member 与异类 nearest reference 统一只来自 `orig-train-only dense windows` 的对象系谱。

4. `T4b`
   - 真正有价值的对象已下沉到 window/local-state 层。
   - 因此 `v2` 只刷新 local representative states / prototype / local direction，不回到 whole-trajectory 的粗对象。

5. `T5 / T6`
   - “谁构参考几何”和“谁生成样本映射”必须分层，不应再混成一个 unified 回路。
   - 因此 `v2` 只承担慢层 geometry refresh，快层 sample generation 留给后续 `new v3`。

6. `T7a`
   - 线性容器即使分家，也不自动转化成收益。
   - 因此 `v2` 不追求扩大 container zoo，而是先把当前局部参考几何刷新得更稳、更可分、更可读。

---

## 一、唯一核心问题
**在固定 `dense z_seq + dynamic_minirocket + prototype-memory + v1b tight anchors` 的前提下，能否用低成本、闭式或近闭式的局部更新，对当前 `SCP` 个体的局部流形参考做一次阶段性刷新，并得到更稳、更可分、且不过度破坏局部 margin/类内紧致性的几何。**

这一步只问：
- 当前局部参考几何能不能被刷新得更好

当前不问：
- replay 能不能吸收
- fast mapping 最终能否追到更高 F1
- online / test-time adaptation 是否成立

---

## 二、为什么这是当前最合理入口
### 1. 不是继续 replay
原 `v2 replay` 已经证明：
- 它会把“几何更新”和“训练样本回放”缠在一起
- 容易产生 stitching / boundary artifact
- 不符合当前逐渐清晰的慢层 / 快层分工

### 2. 不是继续磨 `v1x`
`v1b` 已经说明：
- 对象收紧方向是对的
- 但继续只磨 admitted anchors，收益在变小

### 3. 不是直接开 fast local mapping
fast local mapping 需要一个更可信的局部几何参考；
如果参考空间本身还没刷新，快层生成会继续建立在旧参考上。

### 4. 不是 online / test-time adaptation
当前 benchmark 仍是 fixed split；
现在引入 online/test-time 更新，会把变量再度炸开。

---

## 三、当前全部冻结
以下全部冻结，并且未来的 `new v3` 必须共用这套定义：

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

---

## 四、新 v2 第一版定义

### 1. v2 的职责
新 `v2` 只做：

**Global/Local Geometry Refresh**

即：
- 刷新当前个体的 prototype-memory
- 刷新 local representative states
- 刷新 local discriminative directions
- 在必要时刷新 classwise local reference

其目标不是生成更多训练样本，而是：

**得到一个更好的当前局部流形参考空间。**

### 2. 第一版允许更新的对象
第一版只允许更新以下三类对象：

1. `prototype`
2. `local representative states`
3. `local discriminative direction`

当前不允许更新：
- terminal 参数
- backbone 参数
- raw-domain 表示

### 3. 第一版更新规则
第一版坚持闭式或近闭式、低成本、可审计：

#### prototype refresh
- `prototype moving average`
- 输入：当前 prototype + admitted tight anchors 的均值
- 更新形式：固定 `alpha` 的 EMA / convex combination

#### representative-state refresh
- `medoid refresh`
- 将 prototype 刷新后，对应到类内最接近的新 medoid / representative member

#### local direction refresh
- `local margin direction recompute`
- 以刷新后的 same-class representative 和最近异类 representative 重算局部判别方向

当前不做：
- terminal 重训内环
- 梯度学习式几何更新
- 局部图传播

### 3.5 第一版必须加入 balanced acceptance
旧语义下的 `v3 closed-form update` 已暴露出一个关键风险：
- `between_prototype_separation` 可以上升
- 但 `nearest_prototype_margin` 与 `within_prototype_compactness` 会一起变坏

因此，`v2` 第一版不能把“分得更开”当成唯一目标，而必须加入 **balanced acceptance**。

对每个候选 refresh 结果，至少审计以下三个方向：
- `delta_between_prototype_separation`
- `delta_nearest_prototype_margin`
- `delta_within_prototype_compactness`

第一版 acceptance 写死为：
- 若 `delta_between_prototype_separation <= 0`，拒绝更新
- 若 `delta_nearest_prototype_margin < 0`，拒绝更新
- 若 `delta_within_prototype_compactness > 0`，拒绝更新

被拒绝的对象回退到 refresh 前版本。

这条约束的目的不是追求最终最优，而是避免再次出现“分离度上升，但局部判别性和类内紧致同时恶化”的粗更新。

### 4. v2 的代价口径
`v2` 的一个核心目标是：

**显著低于重新训练 terminal 的成本**

因此必须同时输出：
- `update_time_seconds`
- `retrain_reference_seconds`
- `update_to_retrain_ratio`

这里的 `retrain_reference` 必须统一写死为：
- 同一份 `dense z_seq`
- 同一份 `dynamic_minirocket`
- 同一 train/val/test split
- 同一 normalization
- 同一 terminal 训练协议

不允许临时更换参考重训口径。

---

## 五、方法学口径
`v2` 测的是：

**train-only, single-round, closed-form local geometry refresh** 的整体效果。

当前不宣称：
- 这已经是最终最优更新规则
- 这已经是完整 continual learning 方案
- 这一步单独就应该直接追到终点 F1 最优

`v2` 的主要任务是为下一阶段的 `new v3 fast local mapping` 提供更好的局部几何参考。

---

## 六、允许做 / 不允许做

### 允许做
- prototype moving average
- medoid refresh
- local discriminative direction recompute
- balanced acceptance / rollback
- 结构指标 before/after 对照
- 代价与重训对照

### 不允许做
- replay
- curriculum
- multi-round self-evolution
- online / test-time adaptation
- terminal 参数更新
- backbone 参数更新
- 双流
- NATOPS 兼容扩展
- 新 gate / role / container zoo

---

## 七、主比较对象
当前只在 `SCP1` 上做主比较：

1. `v0 prototype-memory`
2. `v1b shaped memory`
3. `v2 refreshed geometry`

当前不把：
- `raw_minirocket`
- `dense_dynamic_gru`
- `NATOPS`

放进这一轮主矩阵里当主判读对象。

它们可以保留为背景参考，但不是 `v2` 的直接核心问题。

---

## 八、建议复用/新增模块
建议复用：
- `route_b_unified/scp_prototype_memory.py`
- `route_b_unified/scp_local_shaping.py`

建议主模块：
- `route_b_unified/scp_closed_form_update.py`

建议 runner：
- `scripts/run_route_b_scp_branch_v3.py`

但从语义上，应将其升级解释为：
- **新 `v2 geometry refresh` runner**

也就是说：
- 代码可以暂时复用旧 `v3` 入口
- 但框架语义上，它现在属于 `v2`

---

## 九、必须输出的文件
- `scp_branch_v2_config_table.csv`
- `scp_branch_v2_per_seed.csv`
- `scp_branch_v2_dataset_summary.csv`
- `scp_branch_v2_prototype_update_summary.csv`
- `scp_branch_v2_direction_summary.csv`
- `scp_branch_v2_acceptance_summary.csv`
- `scp_branch_v2_conclusion.md`

其中 `scp_branch_v2_acceptance_summary.csv` 至少必须单独记录：
- `accept_between`
- `accept_margin`
- `accept_within`
- `final_accept`

如果暂时复用旧 runner 名称，也必须在结论文档中明确写清：
- 这是 **new v2 geometry refresh**
- 不是旧语义下的 `v3`

---

## 十、必须回答的问题
1. `v2` 是否改善了 `between_prototype_separation`
2. `v2` 是否改善了 `nearest_prototype_margin`
3. `v2` 是否没有明显破坏 `within_prototype_compactness`
4. `v2` 是否没有明显破坏 `temporal_assignment_stability`
5. balanced acceptance 是否有效避免了“只拉开 separation、却破坏 margin/within”的粗刷新
6. `v2` 的更新时间是否显著低于 terminal 重训
7. 若 `v2` 成立，是否值得进入：
   - **new `v3 fast local mapping`**
8. 若 `v2` 失败，更像问题在：
   - 更新规则过粗
   - 对象定义仍不够好
   - 或 `prototype-memory` 本身还不够可刷新

---

## 十一、成功标准

### 弱成立
- `between_prototype_separation` 有改善
- 且 `nearest_margin` 不下降
- 且 `within_compactness` 不明显恶化

### 中等成立
- `nearest_prototype_margin` 也有改善
- `temporal_stability` 不明显下降
- rollback 比例不高
- `update_to_retrain_ratio` 明显小于 `1.0`

### 强成立
同时满足：
- `between_separation ↑`
- `nearest_margin ↑`
- `within_compactness` 基本保住
- `temporal_stability` 基本保住
- balanced acceptance 不是靠大量拒绝才保住指标
- 更新成本显著低于完整重训

---

## 十二、与下一阶段的关系
`v2` 不是终点，而是为下一阶段准备参考几何。

一旦 `v2` 成立，下一步应该进入：

## `new v3: Fast Local Sample Generation / Local Mapping`

也就是：
- 固定当前 `v2` 刷新后的局部几何参考
- 快速为当前窗口生成终端更容易读取的局部样本视图

所以：
- `v2` 是慢层
- `new v3` 是快层

---

## 十三、一句话执行目标
**停止把 replay 当成下一阶段主线；先把原来的 closed-form local update 正式提升为新 `v2 geometry refresh`，在 train-only、single-round、offline 条件下刷新当前 `SCP` 的局部流形参考，并验证这种刷新是否值得成为后续 fast local mapping 的慢层基础。**
