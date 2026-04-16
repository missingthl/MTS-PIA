# Current Engineering Map

更新时间：2026-04-16

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
- [工程记录/分类/00-入口/分类工程现状.md](../工程记录/分类/00-入口/分类工程现状.md)
- [工程记录/分类/00-入口/分类调试记录.md](../工程记录/分类/00-入口/分类调试记录.md)
- [工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](../工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
- [scripts/README.md](../scripts/README.md)

## 当前活跃实验是什么

当前二阶段已经变成双宿主结构：

- `Tensor-CSPNet`
  - `BCIC-IV-2a holdout`
  - 当前主要承担 EEG/SPD 宿主验证与端到端闭式残差实现验证
- `ResNet1D`
  - fixed-split 多数据集对比
  - 当前主要承担通用 MTS 宿主验证与公开结果沉淀

当前代码入口：

- [models/local_closed_form_residual_head.py](../models/local_closed_form_residual_head.py)
- [models/tensor_cspnet_adapter.py](../models/tensor_cspnet_adapter.py)
- [models/tensor_cspnet_residual_linear.py](../models/tensor_cspnet_residual_linear.py)
- [scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py](../scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py)
- [models/resnet1d.py](../models/resnet1d.py)
- [models/resnet1d_adapter.py](../models/resnet1d_adapter.py)
- [models/resnet1d_local_closed_form.py](../models/resnet1d_local_closed_form.py)
- [scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py](../scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py)

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

- `Tensor-CSPNet`
  - [stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md](../out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md)
- `ResNet1D`
  - [verify_resnet1d_stage2_framework_compare_20260413](../out/_active/verify_resnet1d_stage2_framework_compare_20260413)
  - [verify_resnet1d_stage2_framework_compare_ext_20260413](../out/_active/verify_resnet1d_stage2_framework_compare_ext_20260413)
  - [verify_resnet1d_stage2_e2_pinv_compare_20260414](../out/_active/verify_resnet1d_stage2_e2_pinv_compare_20260414)
  - [verify_resnet1d_e2_template_matrix_20260414](../out/_active/verify_resnet1d_e2_template_matrix_20260414)

当前 `ResNet1D` 结果沉淀的最短读法：

- **阶段二验证完毕**：完成 21 个数据集在 `E2 tau=0.2 corrected` 协议下的全量扫表。
- **超越 SOTA**：在 `NATOPS` (+0.5%)、`FingerMovements` (+2.0%)、`JapaneseVowels`、`Libras` 和 `MotorImagery` (+4.0%) 等 5 个数据集上超越了 MiniRocket SOTA，另有两个数据集战平。
- **协同演化定论 (Synergy Verdict)**：通过 MiniRocket (Frozen Backbone) 挂载 DLCR 的实验对比（精度显著下降），确认了 DLCR 的威力源于**主干网在联合训练中针对闭式求解器进行的特征空间自适应（Synergy）**。
- **当前最优配置**：`solve_mode=pinv` + `prototype_geometry_mode=center_subproto` + `tau=0.2`。

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
