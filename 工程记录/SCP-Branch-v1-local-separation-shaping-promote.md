# SCP-Branch v1 Local Separation Shaping Promote

更新时间：2026-03-30

## 一、唯一核心问题

在 prototype-memory 已成立的前提下，若只对其对应的局部代表窗口做 train-only 的 local separation shaping，`SCP1` 是否会出现更好的局部几何可分性，并进一步提升 `dynamic_minirocket` 的终端可利用性。

## 二、为什么这是当前最合理入口

- 当前先单独验证 local shaping 本身，不引入 replay。
- 当前不引入 curriculum，避免把对象与回放策略混在一起。
- 当前不引入 neighborhood propagation，避免一步变成半图传播系统。
- 当前不引入 online / test-time adaptation，保持 fixed-split 评估口径。
- 当前不回到全局单轴整条轨迹推点，而只做 representative-state-centered local shaping。

## 三、当前全部冻结

- 冻结 `dense z_seq`
- 冻结 `dynamic_minirocket`
- 冻结当前 dense 表示协议
- 冻结 train/val/test split
- 冻结 normalization
- 冻结 prototype-memory v0 对象定义
- `val/test` 继续只输入原始 dense trajectory
- 不做 replay
- 不做 curriculum
- 不做 neighborhood propagation
- 不做 dual-role
- 不做 test-time routing / filtering / update
- 不做 raw-level 操作
- 不做双流

## 四、v1 第一版定义

1. 作用对象只限于 admitted prototype-memory windows
2. admitted anchor 规则固定为：
   - 对每个类、每个 prototype
   - 取距离该 prototype centroid 最近的真实成员窗口前 `M=16`
3. `anchor_coverage_ratio` 统一定义为：
   - admitted anchor 数 / 该 prototype member 总数
   - 当前不将其解释为类覆盖率、trial 覆盖率或时间覆盖率
4. 对每个被选中的窗口 `z`：
   - `p_same` 为其本类 representative anchor centroid
   - `p_opp` 为最近的异类 representative anchor centroid
5. shaping 方向固定为一个受限 local margin direction：
   - `direction_local = normalize((p_same - z) + beta * (z - p_opp))`
6. 第一版固定：
   - `beta = 0.5`
   - `epsilon_scale = 0.10`
   - `epsilon_local = 0.10 * min(||z-p_same||, ||z-p_opp||)`
7. shaping 只发生在 train 阶段
8. baseline 必须是 same backbone / no shaping 对照：
   - 同一份 dense backbone
   - 同一 normalization
   - 同一 dynamic_minirocket 训练协议
   - 与 v1 的唯一差别是 baseline 不执行 shaping 写回

## 五、当前不做的事

- 不做 replay
- 不做 curriculum
- 不做 neighborhood influence / propagation
- 不做 test-time adaptation
- 不做 prototype-based inference routing
- 不做全局单轴 operator

## 六、必须输出

1. `scp_branch_v1_config_table.csv`
2. `scp_branch_v1_per_seed.csv`
3. `scp_branch_v1_dataset_summary.csv`
4. `scp_branch_v1_anchor_summary.csv`
5. `scp_branch_v1_shaping_diagnostics.csv`
6. `scp_branch_v1_structure_diagnostics.csv`
7. `scp_branch_v1_conclusion.md`

## 七、必须审计的结构指标

- `within_prototype_compactness`
- `between_prototype_separation`
- `nearest_prototype_margin`
- `temporal_assignment_stability`

理想情况是：
- `between_separation ↑`
- `nearest_margin ↑`
- `within_compactness` 不明显恶化
- `temporal_stability` 不明显崩坏

## 八、额外风险审计

- `local_step_distortion_ratio`
- `local_step_distortion_ratio_p95`
- `margin_to_score_conversion`

若 `nearest_prototype_margin` 上升但终点分数不动，不直接判为 shaping 失败；该情形优先解释为 local geometry gain 未被当前 `dynamic_minirocket` 终端充分读取。

## 九、一句话执行目标

冻结当前 `SCP` dense backbone 与 prototype-memory 对象，不进入 replay / curriculum / online adaptation；只让 PIA 围绕 admitted local representative windows 做受限、局部、训练专用的 separation shaping，测试它是否真能提升 `SCP1` 的局部几何可分性与终端可利用性。
