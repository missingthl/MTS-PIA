# MTS-PIA Workspace

更新时间：2026-04-13

这个仓库现在不要再按“所有历史实验都同权并列”的方式阅读。更准确的读法是：

1. 当前活跃主线是**分类二阶段**
2. 一阶段 `PIA-Operator` 现在主要保留为**结构证据层**
3. 动态线、SCP、时间序列都保留，但不再是默认入口

## 当前最短入口

如果你想快速接住当前工程，只看下面 5 份：

1. [工程记录/分类/README.md](/home/THL/project/MTS-PIA/工程记录/分类/README.md)
2. [工程记录/分类/00-入口/分类二阶段现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类二阶段现状.md)
3. [工程记录/分类/01-阶段二-Tensor-CSPNet/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/分类/01-阶段二-Tensor-CSPNet/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
4. [工程记录/分类/00-入口/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md)
5. [docs/CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)

## 当前工程怎么理解

### 分类二阶段：当前活跃层

当前默认主线是：

`Tensor-CSPNet backbone -> Temporal_Block latent -> residual head`

当前实验矩阵是：

- `E0`
- `E1`
- `E2`

当前默认协议是：

- `BCIC-IV-2a holdout`

当前最直接的结果总结：

- [stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md)

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
  训练与验证入口
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

- [models/tensor_cspnet_adapter.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_adapter.py)
- [models/tensor_cspnet_residual_linear.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_residual_linear.py)
- [models/local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py)
- [scripts/run_tensor_cspnet_local_closed_form_holdout.py](/home/THL/project/MTS-PIA/scripts/run_tensor_cspnet_local_closed_form_holdout.py)

一阶段：

- [route_b_unified/pia_operator_value_probe.py](/home/THL/project/MTS-PIA/route_b_unified/pia_operator_value_probe.py)

## 当前不建议怎么读

请不要再把当前仓库理解成：

- 旧 `Route B / bridge / MiniROCKET` 仍是唯一活跃主线
- 所有 promote 文档都应平铺逐个阅读
- 动态线和时间序列线已经接管当前默认叙事

当前更准确的总读法是：

**先看分类二阶段，再用一阶段结构证据解释它，最后按需回看支线和历史。**
