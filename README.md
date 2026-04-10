# MTS-PIA Workspace

更新时间：2026-03-29

当前仓库最重要的事情，不是再区分一堆旧实验线，而是先分清：

1. 当前真正还在推进的分类主线是什么
2. 哪一条是并行保留的回归探针分支
3. 哪些脚本和结果是当前有效入口
4. 哪些内容只作为历史参考

当前统一运行约定：

- 只要是在本工程里执行当前主线任务，默认都走 `pia` 环境
- 统一入口是 [scripts/run_in_pia.sh](scripts/run_in_pia.sh)
- 当前主线完整自检入口是 [scripts/verify_current_stack_in_pia.sh](scripts/verify_current_stack_in_pia.sh)

## 1. 当前主线

当前活跃的分类主线是：

`representation -> TELM2 / PIA Core -> explicit affine operator -> bridge -> raw MiniROCKET`

当前阶段的推进顺序是：

1. `PIA Core minimal chain`
2. `Augment Admission & Composition`
3. `Operator Control Refinement`

当前最新状态：

- `PIA Core minimal chain`
  - 已正式跑通
  - `SCP1` 为正
  - `NATOPS` 为 `flat / 小幅正`
- 第一包 `admission`
  - 正式结论：`not yet useful`
- 第二包 `axis refine`
  - 已确认第二轴是主风险源之一
  - 比粗粒度 `k=2` 更接近共同候选点
  - 但还没有形成统一强点

### 并行保留：z-space regression probe

这条线当前只作为**并行验证分支**保留，不替代分类主线：

`raw trial -> z-space -> regressor`

当前定位：

- 代码与数据都已单独分层
- 第一阶段 `IEEEPPG` baseline 已完成
- 当前结论：`not yet justified for phase-2 augmentation`

### 并行保留：no-bridge dual-stream classification probe

这条线当前也只作为**并行验证分支**保留，不替代当前 `PIA Core + bridge` 主线：

`raw trial -> DCNet spatial stream`
`+`
`raw trial -> representation.py -> z-space manifold stream`
`-> fusion classifier`

当前定位：

- 不经过 `bridge`
- 不做 `PIA augmentation`
- 只验证“几何流 + raw 时序流”的双流终点是否比单流更自然
- 第一轮 smoke 已完成，当前入口见：
  - [run_route_b_dual_stream_no_bridge.py](scripts/run_route_b_dual_stream_no_bridge.py)
  - [verify_route_b_dual_stream_no_bridge_combined_20260329](out/_active/verify_route_b_dual_stream_no_bridge_combined_20260329)

### 新表示路径 T0：dynamic manifold classification

这条线当前不是旧静态流形的小补丁，而是**静态 SPD 单点主线阶段冻结后的新表示路径 T0**：

`raw trial -> sliding windows -> SPD/log-Euclidean trajectory z_seq -> minimal trajectory classifier`

当前定位：

- 先验证“样本对象从单点变轨迹”是否成立
- 不预设最终一定走：
  - 双流融合
  - 去 bridge
  - 或保留 bridge
- 当前第一轮只比较：
  - `static_linear`
  - `dynamic_meanpool`
  - `dynamic_gru`
  - `raw + MiniROCKET` 仅作外部参考
- 当前入口见：
  - [工程记录/动态流形分类T0-promote.md](工程记录/动态流形分类T0-promote.md)
  - [run_route_b_dynamic_manifold_classification.py](scripts/run_route_b_dynamic_manifold_classification.py)
  - [route_b_dynamic_manifold_classification_20260329_formal](out/route_b_dynamic_manifold_classification_20260329_formal)
  - [verify_route_b_dynamic_manifold_classification_20260329](out/_active/verify_route_b_dynamic_manifold_classification_20260329)

### 新表示路径 T2a：trajectory-aware operator

这条线当前是建立在 `T0 trajectory manifold branch` 之上的第一版轨迹增强入口：

`pooled train windows -> shared trajectory basis -> train-only trajectory augmentation -> dynamic_gru`

当前定位：

- 先验证“共享全局 basis + 最小时间连续性约束”的轨迹增强算子是否成立
- 第一版只做：
  - `baseline`
  - `operator_unsmoothed`
  - `operator_smoothed`
- 不引入：
  - bridge
  - 双流融合
  - feedback / re-basis
  - `T2b` 局部时间感知 operator
- 当前 formal 已完成，当前入口见：
  - [工程记录/轨迹感知增强算子T2a-promote.md](工程记录/轨迹感知增强算子T2a-promote.md)
  - [run_route_b_trajectory_pia_t2a.py](scripts/run_route_b_trajectory_pia_t2a.py)
  - [route_b_trajectory_pia_t2a_20260329_formal](out/route_b_trajectory_pia_t2a_20260329_formal)
  - [verify_route_b_trajectory_pia_t2a_20260329](out/_active/verify_route_b_trajectory_pia_t2a_20260329)

### 新表示路径 T2b-0：fixed-rule local saliency probe

这条线当前不是完整局部动力学系统，而是建立在冻结后的 `T2a default` 之上的下一阶段框架 probe：

`shared basis -> fixed-rule local saliency gating -> trajectory train augmentation`

当前定位：

- 不再继续细磨 `T2a` 参数
- 直接比较：
  - `baseline`
  - `t2a_default`
  - `t2b_saliency`
  - `t2b_randomized`
- 核心问题不是“是不是又涨了 0.003”，而是：
  - 局部时间感知强度分配
  - 是否比全局统一强度和随机时变扰动更值得推进
- 当前入口见：
  - [工程记录/轨迹感知增强算子T2b-0-promote.md](工程记录/轨迹感知增强算子T2b-0-promote.md)
  - [run_route_b_trajectory_pia_t2b0.py](scripts/run_route_b_trajectory_pia_t2b0.py)
  - [route_b_trajectory_pia_t2b0_20260329_formal](out/route_b_trajectory_pia_t2b0_20260329_formal)

### 新表示路径 T3：dynamic feedback re-basis probe

当前在冻结后的 `T2a default` 生成器上，已经进一步推进到动态分支上的反馈重构 probe：

`frozen generator -> safety-filtered feedback pool -> re-center -> constrained shared-basis re-fit -> re-augment orig trajectories`

当前定位：

- 这不是静态分支 `M0+` 的改名版
- 也不是完整动态闭环系统
- 它更像：
  - `dynamic manifold feedback re-basis probe`
- 核心问题不是“轨迹增强还能不能再多涨一点”，而是：
  - 增强结果能否反过来改变共享 basis 和参考几何
- 当前入口见：
  - [工程记录/动态流形反馈重构T3-promote.md](工程记录/动态流形反馈重构T3-promote.md)
  - [run_route_b_dynamic_feedback_rebasis_t3.py](scripts/run_route_b_dynamic_feedback_rebasis_t3.py)
  - [route_b_dynamic_feedback_rebasis_t3_20260329_formal](out/route_b_dynamic_feedback_rebasis_t3_20260329_formal)

## 2. 先看哪里

如果现在要接手工程，先看这 5 个入口：

- [docs/CURRENT_ENGINEERING_MAP.md](docs/CURRENT_ENGINEERING_MAP.md)
- [工程记录/分类工程现状.md](工程记录/分类工程现状.md)
- [工程记录/分类调试记录.md](工程记录/分类调试记录.md)
- [scripts/README.md](scripts/README.md)
- [docs/PROJECT_INDEX.md](docs/PROJECT_INDEX.md)

当前最重要的正式结果入口：

- [pia_core_minimal_chain_summary.csv](out/route_b_pia_core_minimal_chain_20260327_formal/pia_core_minimal_chain_summary.csv)
- [admission_conclusion.md](out/route_b_pia_core_admission_control_20260327_formal/admission_conclusion.md)
- [axis_refine_conclusion.md](out/route_b_pia_core_axis_refine_20260327_formal/axis_refine_conclusion.md)
- [axis_pullback_refine_conclusion.md](out/route_b_pia_core_axis_pullback_refine_20260328_formal/axis_pullback_refine_conclusion.md)
- [risk_aware_axis2_conclusion.md](out/route_b_pia_core_risk_aware_axis2_20260328_formal/risk_aware_axis2_conclusion.md)
- [zspace_regression_conclusion.md](out/regression/route_b_zspace_regression_baseline_20260328/zspace_regression_conclusion.md)

## 3. 根目录怎么理解

当前不要把根目录里所有一级目录都当成“现在要看的代码”。

### 当前主线白名单

- `PIA/`
- `route_b_unified/`
- `transforms/`
- `datasets/`
- `data/`
- `scripts/`
- `docs/`
- `工程记录/`
- `out/`

### 当前仍可能被旧脚本依赖的支持层

- `manifold_raw/`
- `models/`
- `runners/`
- `tools/`

这些目录当前不是主叙事中心，但还不能随意搬动。

### 已降级/已归档的资料或历史工作区

- `archive/`

参考资料目录已经移到：

- [archive/reference_materials](archive/reference_materials)

历史工作区已移到：

- [archive/legacy_workspace](archive/legacy_workspace)

legacy 代码层已移到：

- [archive/legacy_code](archive/legacy_code)

## 4. 当前最重要的代码

核心模块：

- [representation.py](route_b_unified/representation.py)
- [pia_core.py](route_b_unified/pia_core.py)
- [bridge.py](route_b_unified/bridge.py)
- [evaluator.py](route_b_unified/evaluator.py)
- [augmentation_admission.py](route_b_unified/augmentation_admission.py)

回归探针模块：

- [route_b_unified/regression](route_b_unified/regression)
- [datasets/regression](datasets/regression)
- [scripts/regression](scripts/regression)

当前活跃 runner：

- [run_route_b_pia_core_minimal_chain.py](scripts/run_route_b_pia_core_minimal_chain.py)
- [run_route_b_pia_core_admission_control.py](scripts/run_route_b_pia_core_admission_control.py)
- [run_route_b_pia_core_axis_refine.py](scripts/run_route_b_pia_core_axis_refine.py)
- [run_route_b_pia_core_axis_pullback_refine.py](scripts/run_route_b_pia_core_axis_pullback_refine.py)
- [run_route_b_pia_core_risk_aware_axis2.py](scripts/run_route_b_pia_core_risk_aware_axis2.py)
- [run_route_b_dual_stream_no_bridge.py](scripts/run_route_b_dual_stream_no_bridge.py)

统一示例：

```bash
scripts/run_in_pia.sh scripts/run_route_b_pia_core_minimal_chain.py --help
scripts/run_in_pia.sh bash scripts/verify_current_stack_in_pia.sh
```

回归探针 runner：

- [run_route_b_zspace_regression_baseline.py](scripts/regression/run_route_b_zspace_regression_baseline.py)

## 5. 当前只作参考的线

下面这些线仍然保留，但不再是当前主推进入口：

- `Route B legacy bridge / multiround`
- `LRAES + curriculum`
- `route_b_unified` policy shell
- `Freeze`

这些线的作用是：

- 提供对照
- 提供历史诊断
- 提供旧结果参考

而不是继续主导当前工程叙事。

## 6. 当前整理原则

从现在开始，优先按下面这个顺序理解仓库：

1. `PIA Core current line`
2. `当前 formal 结果`
3. `当前活跃 runner`
4. `历史参考线`

不要再从：

- `Freeze`
- `unified policy shell`
- 旧的大矩阵脚本

倒着理解整个工程。
