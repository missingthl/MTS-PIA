# MTS-PIA Workspace

更新时间：2026-04-14

这个仓库现在不要再按“所有历史实验都同权并列”的方式阅读。更准确的读法是：

1. 当前活跃主线是**分类二阶段宿主验证**
2. 一阶段 `PIA-Operator` 现在主要保留为**结构证据层**
3. 动态线、SCP、时间序列都保留，但不再是默认入口

## 当前最短入口

如果你想快速接住当前工程，只看下面 6 份：

1. [工程记录/分类/README.md](/home/THL/project/MTS-PIA/工程记录/分类/README.md)
2. [工程记录/分类/00-入口/分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类工程现状.md)
3. [工程记录/分类/00-入口/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md)
4. [工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
5. [docs/CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
6. [scripts/README.md](/home/THL/project/MTS-PIA/scripts/README.md)

## 当前工程怎么理解

### 分类二阶段：当前活跃层

当前二阶段已经不是单一宿主，而是双宿主结构：

- `Tensor-CSPNet`
  - 当前主要承担 EEG/SPD 外部宿主验证与端到端闭式残差实现验证
- `ResNet1D`
  - 当前主要承担通用 MTS 宿主验证与公开结果沉淀

当前框架矩阵是：

- `E0`
- `E1`
- `E2`

当前活跃协议分两条：

- `Tensor-CSPNet`：
  - `BCIC-IV-2a holdout`
- `ResNet1D`：
  - fixed-split 多数据集对比

当前最直接的结果总结有两块：

- `Tensor-CSPNet`：
  - [stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md)
- `ResNet1D`：
  - [verify_resnet1d_stage2_framework_compare_20260413](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_stage2_framework_compare_20260413)
  - [verify_resnet1d_stage2_framework_compare_ext_20260413](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_stage2_framework_compare_ext_20260413)
  - [verify_resnet1d_stage2_e2_pinv_compare_20260414](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_stage2_e2_pinv_compare_20260414)
  - [verify_resnet1d_e2_template_matrix_20260414](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_e2_template_matrix_20260414)

当前 `ResNet1D` 宿主上的阶段性读法是：

- `E2` 第一版并非统一赢家
- `E2 + pinv` 已经在 `FingerMovements / SelfRegulationSCP1 / Epilepsy / AtrialFibrillation` 上优于 `E0`
- 模板机制已经开始展现结构作用：
  - `4模板 + pooled` 更适合 `FingerMovements / SCP1`
  - `4模板 + committee_mean` 更适合 `NATOPS / UWaveGestureLibrary / Epilepsy`

### 分类一阶段：结构证据层

一阶段现在不再是默认对外方法形态，而主要负责回答：

- 为什么 `same/opp bipolar` 成立
- 为什么问题会从 `axis -> force` 转向 `operator scope`
- 为什么当前二阶段会长成 `local / residual / end-to-end`

当前入口：

- [工程记录/分类/00-入口/阶段一结构证据总览.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/阶段一结构证据总览.md)
- [工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md)

### 并行支线

仍保留，但当前不是默认入口：

- 动态分类线：
  [工程记录/分类/03-并行支线-动态](/home/THL/project/MTS-PIA/工程记录/分类/03-并行支线-动态)
- SCP 支线：
  [工程记录/分类/04-并行支线-SCP](/home/THL/project/MTS-PIA/工程记录/分类/04-并行支线-SCP)
- 时间序列线：
  [工程记录/时间序列/README.md](/home/THL/project/MTS-PIA/工程记录/时间序列/README.md)

## 目录怎么读

当前最重要的目录是：

- `models/`
  二阶段当前实现
- `scripts/`
  已按 `hosts / route_b / raw_baselines / support / analysis / data_prep / probes / forecast / manifold / seed_suites / devtools / legacy_phase` 分层
- `route_b_unified/`
  一阶段算子与几何核心实现
- `工程记录/`
  按阶段整理后的文档入口
- `out/_active/`
  当前有效结果
- `archive/reference_code/`
  外部宿主与参考实现

## 当前最该看的代码

二阶段：

- [models/local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py)
- [models/tensor_cspnet_adapter.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_adapter.py)
- [models/tensor_cspnet_residual_linear.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_residual_linear.py)
- [scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py](/home/THL/project/MTS-PIA/scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py)
- [models/resnet1d.py](/home/THL/project/MTS-PIA/models/resnet1d.py)
- [models/resnet1d_local_closed_form.py](/home/THL/project/MTS-PIA/models/resnet1d_local_closed_form.py)
- [scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py](/home/THL/project/MTS-PIA/scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py)

一阶段：

- [route_b_unified/pia_operator_value_probe.py](/home/THL/project/MTS-PIA/route_b_unified/pia_operator_value_probe.py)

## 当前不建议怎么读

请不要再把当前仓库理解成：

- 旧 `Route B / bridge / MiniROCKET` 仍是唯一活跃主线
- 所有 promote 文档都应平铺逐个阅读
- 动态线和时间序列线已经接管当前默认叙事

当前更准确的总读法是：

**先看分类二阶段，再用一阶段结构证据解释它；当前公开结果叙事已经不应再只由 Tensor-CSPNet 单独承担。**
