# SCP-Branch v0 Prototype-Memory Promote

更新时间：2026-03-30

## 一、唯一核心问题

在冻结 `dense z_seq + dynamic_minirocket` backbone 的前提下，`prototype-memory` 这个对象本身，是否能在 `SCP1` 上形成稳定、非随机、具有类相关性的局部状态结构。

这里的 `prototype` 当前**不解释为真实生理稳态中心**，而解释为：

- 当前训练分布内反复出现的局部代表态
- distribution-supported local anchors
- 比随机窗口集合更紧致、更分离、更稳定的局部状态簇

## 二、为什么先做 v0

- 当前最缺的不是 replay，而是确认 `prototype-memory` 是否值得成为 `SCP` 分支的新对象。
- 当前不引入 `PIA-guided local geometry / replay / curriculum`，避免一次性改太多变量。
- 若 `v0` 不成立，后续把 replay 或 PIA 接进来也缺少坚实对象基础。
- 当前不要求训练集里存在“真实稳态中心”；`v0` 只验证训练分布中是否存在可复现的局部代表态。

## 三、当前全部冻结

- 冻结 `dense z_seq`
- 冻结 `dynamic_minirocket`
- 冻结当前 dense 表示协议
- 冻结 train/val/test split
- 不做 augmentation
- 不做 rebasis
- 不做 replay
- 不做 curriculum
- 不做 raw-level 操作
- 不做 bridge
- 不做双流
- 不做 test-time routing
- 不做 online/test-time learning

## 四、第一版定义

1. Backbone 固定为 `raw -> dense z_seq -> dynamic_minirocket`
2. Memory object 固定为 `window-level dense z_t`
3. 直接做 classwise prototype compression
4. 第一版 prototype 数固定为 `K_proto = 4`
5. 当前 cluster mode 锁定为 `kmeans_centroid`
6. 必须增加同规模 `random-memory control`

## 五、当前不做的事

- 不做 `PIA-guided local geometry construction`
- 不做 replay
- 不做 curriculum
- 不做 stitching
- 不做 dual-role
- 不做 local gate zoo
- 不做更重的 safety gate

## 六、主比较对象

1. `dense_dynamic_minirocket`
2. `prototype-memory`
3. `random-memory control`

其中 `dense_dynamic_minirocket` 主要作为 backbone 参考；`prototype-memory` 与 `random-memory` 的核心比较以结构诊断为主。
当前判读的重点不是“有没有全局中心”，而是：

- 是否存在稳定局部簇
- 这些局部簇是否比随机窗口集合更有结构

## 七、必须输出

1. `scp_branch_v0_config_table.csv`
2. `scp_branch_v0_per_seed.csv`
3. `scp_branch_v0_dataset_summary.csv`
4. `scp_branch_v0_memory_summary.csv`
5. `scp_branch_v0_prototype_summary.csv`
6. `scp_branch_v0_random_control_summary.csv`
7. `scp_branch_v0_structure_diagnostics.csv`
8. `scp_branch_v0_conclusion.md`

## 八、关键结构指标

- `candidate_window_count`
- `safe_window_count`
- `prototype_count`
- `coverage_ratio`
- `low_coverage_flag`
- `within_prototype_compactness`
- `between_prototype_separation`
- `nearest_prototype_margin`
- `temporal_assignment_stability`

并且必须和 `random-memory control` 对照。

## 九、一句话执行目标

先不做 replay，不做 curriculum，不做 PIA replay；只在 `dense z_seq + dynamic_minirocket` backbone 上验证 `prototype-memory` 这个对象本身是否能在当前任务分布里形成可复现的局部代表态，且这些局部代表态是否比随机窗口集合更有结构，值得成为 `SCP` 分支的新组件。
