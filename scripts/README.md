# Scripts Index

更新时间：2026-03-29

这份索引只保留“现在值得直接看”的脚本。

## 0. 统一运行方式

当前工程默认统一走 `pia` 环境。

- 统一入口：[run_in_pia.sh](/home/THL/project/MTS-PIA/scripts/run_in_pia.sh)
- 主线自检：[verify_current_stack_in_pia.sh](/home/THL/project/MTS-PIA/scripts/verify_current_stack_in_pia.sh)

推荐用法：

```bash
scripts/run_in_pia.sh scripts/run_route_b_pia_core_minimal_chain.py --help
scripts/run_in_pia.sh bash scripts/verify_current_stack_in_pia.sh
```

## 1. 当前活跃主线

### PIA Core current line

- [run_route_b_pia_core_minimal_chain.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_core_minimal_chain.py)
  - Package 0：最小主链贯通
- [run_route_b_pia_core_admission_control.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_core_admission_control.py)
  - Package 1：admission / composition
- [run_route_b_pia_core_axis_refine.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_core_axis_refine.py)
  - Package 2：第二轴最小细化控制
- [run_route_b_pia_core_axis_pullback_refine.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_core_axis_pullback_refine.py)
  - Package 2B：`axis-2 × pullback` 收口轮
- [run_route_b_pia_core_risk_aware_axis2.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_core_risk_aware_axis2.py)
  - Path A：`risk-aware` 第二轴离散选挡

### 协议与外部基线

- [protocol_split_utils.py](/home/THL/project/MTS-PIA/scripts/protocol_split_utils.py)
  - official protocol / split 统一入口
- [run_raw_minirocket_official_fixedsplit.py](/home/THL/project/MTS-PIA/scripts/run_raw_minirocket_official_fixedsplit.py)
  - fixed-split 官方口径 MiniROCKET
- [run_raw_minirocket_official_seed_family.py](/home/THL/project/MTS-PIA/scripts/run_raw_minirocket_official_seed_family.py)
  - SEED family 官方口径 MiniROCKET

### 回归探针分支

- [run_route_b_zspace_regression_baseline.py](/home/THL/project/MTS-PIA/scripts/regression/run_route_b_zspace_regression_baseline.py)
  - 并行回归 probe
  - 当前只跑 `IEEEPPG`
  - 不属于当前分类主线 Package 0/1/2

### 并行 no-bridge 双流分类 probe

- [run_route_b_dual_stream_no_bridge.py](/home/THL/project/MTS-PIA/scripts/run_route_b_dual_stream_no_bridge.py)
  - `raw DCNet stream + z-space manifold stream`
  - 不经过 `bridge`
  - 不做 `PIA augmentation`
  - 第一轮只比较：
    - `spatial_only`
    - `manifold_only`
    - `dual_stream`

### 新表示路径 T0：dynamic manifold classification

- [run_route_b_dynamic_manifold_classification.py](/home/THL/project/MTS-PIA/scripts/run_route_b_dynamic_manifold_classification.py)
  - 静态 SPD 单点主线阶段冻结后的新表示路径 T0
  - `raw trial -> sliding windows -> z_seq -> minimal trajectory classifier`
  - 第一轮只比较：
    - `static_linear`
    - `dynamic_meanpool`
    - `dynamic_gru`
  - `raw + MiniROCKET` 只作外部参考

### 轨迹感知增强算子 T2a

- [run_route_b_trajectory_pia_t2a.py](/home/THL/project/MTS-PIA/scripts/run_route_b_trajectory_pia_t2a.py)
  - `T2a: Global Trajectory Operator`
  - 在 pooled train windows 上学习单轴共享 basis
  - 只比较：
    - `baseline`
    - `operator_unsmoothed`
    - `operator_smoothed`
  - 终端固定：
    - `dynamic_gru`
  - 当前正式结果：
    - `NATOPS` 上 operator 明确成立，且平滑版最佳
    - `SCP1` 上仅边际优于 baseline，平滑当前未显示必要性

### 轨迹感知增强算子 T2a 收口轮

- [run_route_b_trajectory_pia_t2a_closure.py](/home/THL/project/MTS-PIA/scripts/run_route_b_trajectory_pia_t2a_closure.py)
  - 固定 `T2a` 其余一切
  - 只扫：
    - `gamma_main`
    - `smooth_lambda`
  - `SCP1` 为主矩阵
  - `NATOPS` 只做锚点
  - 当前正式结果：
    - `SCP1` 最稳收口点：
      - `gamma_main = 0.05`
      - `smooth_lambda = 0.50`
    - `NATOPS` 锚点未被打坏
    - 当前不需要急着进入 `T2b`

### 轨迹感知增强算子 T2b-0

- [run_route_b_trajectory_pia_t2b0.py](/home/THL/project/MTS-PIA/scripts/run_route_b_trajectory_pia_t2b0.py)
  - `T2b-0: fixed-rule local saliency probe`
  - 冻结 `T2a default = gamma 0.05 / smooth 0.50`
  - 主比较只看：
    - `baseline`
    - `t2a_default`
    - `t2b_saliency`
    - `t2b_randomized`
  - 不做参数网格
  - 当前正式结果：
    - `NATOPS` 上：
      - `t2b_saliency` 已优于 `t2a_default`
      - 也优于 `t2b_randomized`
    - `SCP1` 上：
      - `t2b_saliency` 与 `t2a_default` 持平
      - 仅略高于 `t2b_randomized`
  - 当前口径：
    - 这是值得继续推进的局部时间感知 probe
    - 还不是完整局部动力学框架

### 动态反馈重构 T3

- [run_route_b_dynamic_feedback_rebasis_t3.py](/home/THL/project/MTS-PIA/scripts/run_route_b_dynamic_feedback_rebasis_t3.py)
  - `T3: dynamic manifold feedback re-basis probe`
  - 冻结：
    - `trajectory representation`
    - 当前窗口策略
    - `dynamic_gru`
    - `T2a default = gamma 0.05 / smooth 0.50`
  - 主比较只看：
    - `baseline`
    - `t2a_default`
    - `t3_rebasis`
  - 核心步骤：
    - `feedback pool`
    - `re-center`
    - `re-basis`
    - 再在原始 trajectory 上重新增强和评估
  - 当前正式结果：
    - `NATOPS` 上：
      - `t3_rebasis` 已优于冻结后的 `t2a_default`
      - 且 shared basis 发生了明显非平凡旋转
    - `SCP1` 上：
      - `t3_rebasis` 仍未优于 `t2a_default`
      - feedback pool 虽稳定存在，但 basis 基本不动
  - 当前口径：
    - `T3` 已经开始回答“增强能否反过来改骨架”
    - 这条线值得继续推进

### 当前阅读建议

如果只是为了推进当前主线，优先只看：

1. `run_route_b_pia_core_minimal_chain.py`
2. `run_route_b_pia_core_admission_control.py`
3. `run_route_b_pia_core_axis_refine.py`
4. `run_route_b_pia_core_axis_pullback_refine.py`
5. `run_route_b_pia_core_risk_aware_axis2.py`
6. `protocol_split_utils.py`

如果看并行无桥双流验证，再单独看：

7. `run_route_b_dual_stream_no_bridge.py`

如果看新表示路径 T0，再单独看：

8. `run_route_b_dynamic_manifold_classification.py`

如果看 T2a，再单独看：

9. `run_route_b_trajectory_pia_t2a.py`

## 2. 当前仍有参考价值的旧线

- [run_bridge_curriculum_pilot.py](/home/THL/project/MTS-PIA/scripts/run_bridge_curriculum_pilot.py)
  - 旧 Route B bridge 参考线
- [run_phase15_multiround_curriculum_probe.py](/home/THL/project/MTS-PIA/scripts/run_phase15_multiround_curriculum_probe.py)
  - 旧 multiround target 升级线
- [run_phase15_lraes_curriculum_probe.py](/home/THL/project/MTS-PIA/scripts/run_phase15_lraes_curriculum_probe.py)
  - LRAES 上游方向线
- [run_route_b_unified_probe.py](/home/THL/project/MTS-PIA/scripts/run_route_b_unified_probe.py)
  - unified shell 历史原型

## 3. 当前不建议作为入口的脚本

下面这些脚本多数仍可运行，但不应再作为当前主线入口：

- `run_phase15_step0*`
- `run_phase15_step1*`
- 各类 `_probe.py / _scan.py / _diag.py / _smoke.py`
- 已被当前 `PIA Core current line` 覆盖的旧 batch 脚本

## 4. 当前正确的阅读顺序

如果先看脚本，再理解工程，最容易迷路。

更稳的顺序是：

1. [CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
2. [分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)
3. 再回到这里找 runner
