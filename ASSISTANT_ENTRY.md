# Assistant Entry

更新时间：2026-04-12

这份文件是给**外部语言助手 / ChatGPT / 代码分析助手**看的最短接手入口。

当前请先不要把这个仓库理解成“仍在继续扩一阶段 `PIA-Operator` 主线”。  
更准确的读法是：

- **阶段一**
  - 现在主要保留为内部几何与闭式算子的证据层
- **阶段二**
  - 现在是当前活跃实验层
  - 目标是在外部宿主 baseline 上验证端到端局部闭式残差层

## 1. 当前真实在做什么

当前分类工程的默认活跃路线是：

**借助外部宿主 baseline 的二阶段验证。**

更具体地说：

- 宿主：
  - `Tensor-CSPNet`
- 当前方法原型：
  - `Temporal_Block` 后 latent 上的 `local closed-form residual head`
- 当前协议：
  - `BCIC-IV-2a holdout`

当前最短总览：

- [工程记录/分类二阶段现状.md](/home/THL/project/MTS-PIA/工程记录/分类二阶段现状.md)
- [工程记录/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
- [工程记录/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类调试记录.md)

## 2. 一阶段现在是什么

一阶段当前仍然重要，但角色已经变化。

它现在主要提供：

- `same/opp bipolar` 为什么成立
- 为什么当前瓶颈在 `operator scope`
- 为什么值得从 `global fixed operator` 转向 `local operator`

当前最重要的一阶段来源：

- [工程记录/PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-当前主线总览.md)
- [工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)
- [工程记录/分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)

当前请把一阶段理解成：

**结构证据层，而不是当前默认对外方法形态。**

## 3. 当前已经完成到哪一步

### 3.1 宿主 baseline 已复现

`Tensor-CSPNet` 已在本机完整复现到：

- `BCIC-IV-2a holdout = 0.7238`

这说明它现在可以视为可信的外部宿主 baseline。

### 3.2 二阶段方法原型已完成最小 smoke

当前已经完成：

- `E0`
  - 原始 `Tensor-CSPNet`
- `E1`
  - `Tensor-CSPNet + residual linear adapter`
- `E2`
  - `Tensor-CSPNet + local closed-form residual head`

在 `subject 1 / 1 epoch / seed 1` 下的最小 smoke。

当前这一步的结论不是“方法已成立”，而是：

- wrapper 已通
- residual 融合已通
- 局部闭式残差头前向/反向已通

### 3.3 当前正在进行什么

当前正在执行：

- `BCIC holdout` 单 seed 全量比较

第一步先跑：

- `E0`

目的：

- 确认新 wrapper 不会把原始宿主改坏

## 4. 当前最该看的代码

### 二阶段当前实现

- [models/tensor_cspnet_adapter.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_adapter.py)
- [models/tensor_cspnet_residual_linear.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_residual_linear.py)
- [models/local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py)
- [scripts/run_tensor_cspnet_local_closed_form_holdout.py](/home/THL/project/MTS-PIA/scripts/run_tensor_cspnet_local_closed_form_holdout.py)

### 一阶段关键来源

- [route_b_unified/pia_operator_value_probe.py](/home/THL/project/MTS-PIA/route_b_unified/pia_operator_value_probe.py)
- [scripts/run_route_b_pia_operator_p0a1_b3.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_operator_p0a1_b3.py)
- [scripts/run_route_b_pia_operator_p1a_stage1_local_wls.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_operator_p1a_stage1_local_wls.py)

## 5. 当前比较口径

当前请把比较口径拆成两套：

### 一阶段口径

- `raw + MiniROCKET`

### 二阶段口径

- 原始宿主 `Tensor-CSPNet`
- `E1 residual linear`
- `E2 local closed-form residual`

也就是说，当前最优先回答的问题是：

1. `E2` 是否优于 `E1`
2. `E2` 是否优于宿主 `E0`

## 6. 不要误读的点

请不要把当前仓库误读成：

- 旧 `Route B / unified shell` 仍是默认主线
- `R0 / P0b-lite` 已经升格为默认下一代方法
- 动态分类线已经替代当前方法主叙事
- 二阶段只是“随便给 Tensor-CSPNet 加了个头”

当前更准确的总读法是：

**当前活跃路线是“外部宿主上的端到端局部闭式残差层验证”；一阶段 `PIA-Operator` 当前主要作为几何与算子证据层保留。**
