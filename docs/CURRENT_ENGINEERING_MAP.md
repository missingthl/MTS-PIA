# Current Engineering Map

更新时间：2026-04-13

这份文件现在优先回答两个问题：

1. 当前活跃工程层到底是哪一层
2. 仓库里哪些目录和文档应该先看

## 当前工程的四层结构

1. **分类二阶段外部宿主验证层**
2. **分类一阶段结构证据层**
3. **分类并行支线**
4. **时间序列与历史层**

其中当前活跃层已经切到：

- **分类二阶段外部宿主验证层**

## 当前最短入口

- [工程记录/分类/README.md](../工程记录/分类/README.md)
- [工程记录/分类/00-入口/分类二阶段现状.md](../工程记录/分类/00-入口/分类二阶段现状.md)
- [工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](../工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
- [scripts/README.md](../scripts/README.md)
- [models/README.md](../models/README.md)

## 当前活跃实验是什么

当前活跃路线固定为：

`Tensor-CSPNet backbone -> Temporal_Block latent -> residual head`

当前协议：

- `BCIC-IV-2a holdout`

当前代码入口：

- [models/tensor_cspnet_adapter.py](../models/tensor_cspnet_adapter.py)
- [models/tensor_cspnet_residual_linear.py](../models/tensor_cspnet_residual_linear.py)
- [models/local_closed_form_residual_head.py](../models/local_closed_form_residual_head.py)
- [scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py](../scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py)

## 一阶段现在是什么

一阶段当前不再是默认对外方法形态，而是结构证据层。

当前最重要的文档：

- [工程记录/分类/00-入口/阶段一结构证据总览.md](../工程记录/分类/00-入口/阶段一结构证据总览.md)
- [工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md](../工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md)
- [工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-R0-多数据集稳定性阶段小结.md](../工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-R0-多数据集稳定性阶段小结.md)
- [工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](../工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)

## 并行支线

当前仍保留，但不作为默认活跃主线：

- [工程记录/分类/03-并行支线-动态](../工程记录/分类/03-并行支线-动态)
- [工程记录/分类/04-并行支线-SCP](../工程记录/分类/04-并行支线-SCP)
- [工程记录/时间序列/README.md](../工程记录/时间序列/README.md)

## 当前结果入口

- [stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md](../out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md)

## 根目录怎么理解

当前优先进入的目录：

- `models/`
- `scripts/`
  已拆成 `hosts / route_b / raw_baselines / support / analysis / data_prep / probes / forecast / manifold / seed_suites / devtools / legacy_phase`
- `route_b_unified/`
- `工程记录/`
- `docs/`
- `out/`

`archive/` 仍保留，但默认只在需要上游参考或历史材料时再进入。
