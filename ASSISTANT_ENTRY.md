# Assistant Entry

更新时间：2026-04-13

这份文件给外部语言助手一个最短、最不容易读偏的入口。

## 当前仓库应该怎么读

当前请先把仓库理解成两层：

- **分类二阶段**
  当前活跃实验层
- **分类一阶段**
  当前结构证据层

不要再把整个仓库默认理解成“仍在继续扩一阶段 `PIA-Operator` 主线”。

## 当前最短阅读路径

1. [工程记录/分类/README.md](/home/THL/project/MTS-PIA/工程记录/分类/README.md)
2. [工程记录/分类/00-入口/分类二阶段现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类二阶段现状.md)
3. [工程记录/分类/01-阶段二-Tensor-CSPNet/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/分类/01-阶段二-Tensor-CSPNet/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
4. [工程记录/分类/00-入口/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md)
5. [工程记录/分类/00-入口/阶段一结构证据总览.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/阶段一结构证据总览.md)

## 当前活跃方法

当前分类默认主线：

`Tensor-CSPNet backbone -> Temporal_Block latent -> residual head`

实验矩阵：

- `E0`
- `E1`
- `E2`

默认协议：

- `BCIC-IV-2a holdout`

当前最直接的结果摘要：

- [stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md)

## 当前一阶段怎么使用

一阶段当前主要提供：

- `same/opp bipolar` 的结构证据
- `operator scope` 问题的来源
- `local operator` 为什么值得进入二阶段

当前最重要的一阶段入口：

- [工程记录/分类/00-入口/阶段一结构证据总览.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/阶段一结构证据总览.md)
- [工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md)
- [工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)

## 当前已知结果

`fp64 mainline`：

- `E0 = 0.7103`
- `E1 = 0.7083`
- `E2 = 0.7018`

`GPU2 fp32`：

- `E0 = 0.7118`
- `E1 = 0.7114`
- `E2 = 0.6983`

当前最重要的读法：

- `SPD fp32` 已是可用候选数值策略
- `E2` 当前还没有稳定赢过 `E0 / E1`
