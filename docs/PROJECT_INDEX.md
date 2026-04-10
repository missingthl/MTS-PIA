# Project Index

更新时间：2026-03-27

## 1. 当前真正的主线

当前分类工程的活跃主线不是：

- `Freeze`
- `route_b_unified` policy shell
- 旧 `Route B legacy bridge`

当前活跃主线是：

`representation -> TELM2 / PIA Core -> explicit affine operator -> bridge -> raw MiniROCKET`

总入口：

- [CURRENT_ENGINEERING_MAP.md](CURRENT_ENGINEERING_MAP.md)

## 2. 当前阶段结果入口

### Package 0

- [pia_core_minimal_chain_summary.csv](../out/route_b_pia_core_minimal_chain_20260327_formal/pia_core_minimal_chain_summary.csv)

### Package 1

- [admission_conclusion.md](../out/route_b_pia_core_admission_control_20260327_formal/admission_conclusion.md)

### Package 2

- [axis_refine_dataset_summary.csv](../out/route_b_pia_core_axis_refine_20260327_formal/axis_refine_dataset_summary.csv)
- [axis_refine_mechanism_summary.csv](../out/route_b_pia_core_axis_refine_20260327_formal/axis_refine_mechanism_summary.csv)
- [axis_refine_conclusion.md](../out/route_b_pia_core_axis_refine_20260327_formal/axis_refine_conclusion.md)

### Package 2B

- [axis_pullback_refine_dataset_summary.csv](../out/route_b_pia_core_axis_pullback_refine_20260328_formal/axis_pullback_refine_dataset_summary.csv)
- [axis_pullback_refine_mechanism_summary.csv](../out/route_b_pia_core_axis_pullback_refine_20260328_formal/axis_pullback_refine_mechanism_summary.csv)
- [axis_pullback_refine_conclusion.md](../out/route_b_pia_core_axis_pullback_refine_20260328_formal/axis_pullback_refine_conclusion.md)

### Path A

- [risk_aware_axis2_dataset_summary.csv](../out/route_b_pia_core_risk_aware_axis2_20260328_formal/risk_aware_axis2_dataset_summary.csv)
- [risk_aware_axis2_mechanism_summary.csv](../out/route_b_pia_core_risk_aware_axis2_20260328_formal/risk_aware_axis2_mechanism_summary.csv)
- [risk_aware_axis2_conclusion.md](../out/route_b_pia_core_risk_aware_axis2_20260328_formal/risk_aware_axis2_conclusion.md)

### Parallel Regression Probe

- [zspace_regression_dataset_summary.csv](../out/regression/route_b_zspace_regression_baseline_20260328/zspace_regression_dataset_summary.csv)
- [zspace_regression_conclusion.md](../out/regression/route_b_zspace_regression_baseline_20260328/zspace_regression_conclusion.md)

## 3. 当前最重要的代码

### 核心模块

- [representation.py](../route_b_unified/representation.py)
- [pia_core.py](../route_b_unified/pia_core.py)
- [bridge.py](../route_b_unified/bridge.py)
- [evaluator.py](../route_b_unified/evaluator.py)
- [augmentation_admission.py](../route_b_unified/augmentation_admission.py)

### 当前活跃 runner

- [run_route_b_pia_core_minimal_chain.py](../scripts/run_route_b_pia_core_minimal_chain.py)
- [run_route_b_pia_core_admission_control.py](../scripts/run_route_b_pia_core_admission_control.py)
- [run_route_b_pia_core_axis_refine.py](../scripts/run_route_b_pia_core_axis_refine.py)
- [run_route_b_pia_core_axis_pullback_refine.py](../scripts/run_route_b_pia_core_axis_pullback_refine.py)
- [run_route_b_pia_core_risk_aware_axis2.py](../scripts/run_route_b_pia_core_risk_aware_axis2.py)

### Parallel Regression Probe

- [route_b_unified/regression](../route_b_unified/regression)
- [datasets/regression](../datasets/regression)
- [run_route_b_zspace_regression_baseline.py](../scripts/regression/run_route_b_zspace_regression_baseline.py)

## 4. 当前最重要的工程记录

- [分类工程现状.md](../工程记录/分类工程现状.md)
- [分类调试记录.md](../工程记录/分类调试记录.md)
- [分类框架全貌.md](../工程记录/分类框架全貌.md)

## 5. 根目录白名单

当前真正需要优先进入的目录只有：

- `PIA/`
- `route_b_unified/`
- `transforms/`
- `datasets/`
- `data/`
- `scripts/`
- `docs/`
- `工程记录/`
- `out/`

资料类目录已归档到：

- [archive/reference_materials](../archive/reference_materials)

历史工作区已归档到：

- [archive/legacy_workspace](../archive/legacy_workspace)

legacy 代码层已归档到：

- [archive/legacy_code](../archive/legacy_code)

## 6. 历史参考入口

这些内容仍然有效，但不是当前主线：

- [ROUTE_B_MAIN_BODY.md](ROUTE_B_MAIN_BODY.md)
- [ROUTE_B_UNIFIED_POLICY.md](ROUTE_B_UNIFIED_POLICY.md)
- [phase15_mainline_freeze.md](phase15_mainline_freeze.md)

它们的定位是：

- 历史路线说明
- 对照与诊断
- 旧结果索引
