# Active Output Layer

更新时间：2026-03-29

这里现在只放“当前还在推进的分类主线结果”。

## 1. PIA Core current line

### Package 0：PIA Core Minimal Chain

- [route_b_pia_core_minimal_chain_20260327_formal](/home/THL/project/MTS-PIA/out/route_b_pia_core_minimal_chain_20260327_formal)

### Package 1：Augment Admission & Composition

- [route_b_pia_core_admission_control_20260327_formal](/home/THL/project/MTS-PIA/out/route_b_pia_core_admission_control_20260327_formal)

### Package 2：Operator Control Refinement

- [route_b_pia_core_axis_refine_20260327_formal](/home/THL/project/MTS-PIA/out/route_b_pia_core_axis_refine_20260327_formal)

### Package 2B：Axis-2 × Pullback Closure Round

- [route_b_pia_core_axis_pullback_refine_20260328_formal](/home/THL/project/MTS-PIA/out/route_b_pia_core_axis_pullback_refine_20260328_formal)

### Path A：Dynamic Risk-Aware Axis-2 Scaling

- [route_b_pia_core_risk_aware_axis2_20260328_formal](/home/THL/project/MTS-PIA/out/route_b_pia_core_risk_aware_axis2_20260328_formal)

## 2. 当前判断

- `PIA Core` 主链已经成立
- 第一包 `admission` 结论为：
  - `not yet useful`
- 第二包 `axis refine` 结论为：
  - 第二轴是主风险源之一
  - refined `k=2` 比粗粒度 `k=2` 更接近共同候选点
- 第二包收口轮 `axis-2 × pullback` 结论为：
  - 比上一轮更接近共同候选点
  - `NATOPS` 正区仍保住
  - `SCP1` 副作用继续减轻
  - 但统一强点仍未形成
- 路径 A `risk-aware axis-2` 结论为：
  - `SCP1` 相对 `base pia` 继续改善
  - 但没有优于当前静态第二轴挡位方案
  - `NATOPS` 也没有保住静态最优

## 2A. 并行 no-bridge 双流分类 smoke

- [verify_route_b_dual_stream_no_bridge_combined_20260329](/home/THL/project/MTS-PIA/out/_active/verify_route_b_dual_stream_no_bridge_combined_20260329)

当前判断：

- 分支已实现并跑通双站 smoke
- `NATOPS` 当前 smoke 里 `manifold_only` 最强
- `SelfRegulationSCP1` 当前 smoke 里 `spatial_only` 最强
- `dual_stream` 已闭环，但还没有超过最强单流

## 2B. 新表示路径 T0：dynamic manifold classification smoke

- [verify_route_b_dynamic_manifold_classification_20260329](/home/THL/project/MTS-PIA/out/_active/verify_route_b_dynamic_manifold_classification_20260329)
- [route_b_dynamic_manifold_classification_20260329_formal](/home/THL/project/MTS-PIA/out/route_b_dynamic_manifold_classification_20260329_formal)

当前判断：

- 这条线是静态 SPD 单点主线冻结后的新表示路径 T0
- 当前 `3 seeds` formal 已完成：
  - `static_linear`
  - `dynamic_meanpool`
  - `dynamic_gru`
- `NATOPS`
  - `dynamic_gru = 0.7940 +/- 0.0103`
  - 已优于 `static_linear = 0.7276 +/- 0.0231`
  - 也高于 `raw + MiniROCKET = 0.7329 +/- 0.0142`
- `SelfRegulationSCP1`
  - `dynamic_gru = 0.6604 +/- 0.0131`
  - 已优于 `static_linear = 0.5171 +/- 0.0120`
  - 但仍低于 `raw + MiniROCKET = 0.6872 +/- 0.0059`
- 两站共同正信号：
  - `GRU > mean-pool`
  - 轨迹顺序信息本身有价值

## 2C. trajectory-aware operator T2a formal

- [route_b_trajectory_pia_t2a_20260329_formal](/home/THL/project/MTS-PIA/out/route_b_trajectory_pia_t2a_20260329_formal)

当前判断：

- 这是 trajectory branch 上的第一版全局共享 basis operator
- 主比较：
  - `baseline`
  - `operator_unsmoothed`
  - `operator_smoothed`
- `NATOPS`
  - `operator_smoothed = 0.8118 +/- 0.0090`
  - 高于 `baseline = 0.7940 +/- 0.0103`
  - 当前 `T2a` 在该数据集上明确成立
- `SelfRegulationSCP1`
  - `operator_unsmoothed = 0.6615 +/- 0.0072`
  - 仅边际高于 `baseline = 0.6604 +/- 0.0131`
  - 平滑未显示必要性
- 当前总判断：
  - `T2a` 已值得保留
  - 但当前还不适合直接跳到 `T2b`

## 2D. trajectory-aware operator T2b-0

- [verify_route_b_trajectory_pia_t2b0_20260329](/home/THL/project/MTS-PIA/out/_active/verify_route_b_trajectory_pia_t2b0_20260329)
- [route_b_trajectory_pia_t2b0_20260329_formal](/home/THL/project/MTS-PIA/out/route_b_trajectory_pia_t2b0_20260329_formal)

当前判断：

- 这条线当前不是完整局部动力学系统，而是：
  - `fixed-rule local saliency probe`
- 主比较：
  - `baseline`
  - `t2a_default`
  - `t2b_saliency`
  - `t2b_randomized`
- `NATOPS`
  - `t2b_saliency = 0.8204 +/- 0.0014`
  - 已高于 `t2a_default = 0.8182 +/- 0.0041`
  - 也高于 `t2b_randomized = 0.8182 +/- 0.0041`
- `SelfRegulationSCP1`
  - `t2b_saliency = 0.6641 +/- 0.0086`
  - 尚未正式高于 `t2a_default = 0.6641 +/- 0.0086`

## 2E. dynamic feedback re-basis T3

- [verify_route_b_dynamic_feedback_rebasis_t3_20260329](/home/THL/project/MTS-PIA/out/_active/verify_route_b_dynamic_feedback_rebasis_t3_20260329)
- [route_b_dynamic_feedback_rebasis_t3_20260329_formal](/home/THL/project/MTS-PIA/out/route_b_dynamic_feedback_rebasis_t3_20260329_formal)

当前判断：

- 这条线当前不是完整动态闭环，而是：
  - `dynamic feedback re-basis probe`
- 主比较：
  - `baseline`
  - `t2a_default`
  - `t3_rebasis`
- `NATOPS`
  - `t3_rebasis = 0.8279 +/- 0.0061`
  - 已高于 `t2a_default = 0.8182 +/- 0.0041`
  - 且 basis 发生了非平凡旋转
- `SelfRegulationSCP1`
  - `t3_rebasis = 0.6628 +/- 0.0092`
  - 尚未高于 `t2a_default = 0.6641 +/- 0.0086`
  - feedback pool 虽稳定存在，但 basis 基本不动
  - 但已略高于 `t2b_randomized = 0.6629 +/- 0.0098`
- 当前最稳妥的口径：
  - `T2b-0` 已成立为下一阶段框架 probe
  - 但还不能把它写成完整局部动力学框架已正式站稳

## 3. 当前不再放在 active 的线

下面这些结果仍然有效，但不再属于“当前活跃主线”：

- `Route B legacy bridge`
- `LRAES + curriculum`
- `route_b_unified` shell
- `Freeze`

它们现在应从：

- [out/_formal/README.md](/home/THL/project/MTS-PIA/out/_formal/README.md)
- [out/README.md](/home/THL/project/MTS-PIA/out/README.md)

进入，而不是再从 active 层进入。
