# Assistant Entry

更新时间：2026-04-10

这份文件是给**外部语言助手 / ChatGPT / 代码分析助手**看的最短接手入口。

目标不是完整覆盖全部历史，而是让你尽快理解：

1. 这个仓库当前在做什么
2. 当前现役主线是什么
3. 当前已经完成到哪一步
4. 哪些分支是并行研究线，哪些只是历史背景
5. 应该先看哪些代码和结果

## 1. 仓库当前在做什么

这个仓库当前的重点是：

**多变量时间序列分类中的几何表示与闭式增强算子研究。**

更准确地说，当前主叙事不是“做一个新的分类器”，而是：

- 先把原始时序映射成几何状态表示
- 再在冻结局部几何上构造闭式可解释 operator
- 再由终端分类器判断这种状态重写是否真的有用

当前请优先把仓库理解为：

**一个围绕 `PIA-Operator` 主线展开的分类研究工程。**

## 2. 当前现役主线

当前默认主线不是早期：

- `Freeze`
- `Route B curriculum`
- `unified shell`

当前现役主线是：

`raw -> dense z_seq -> frozen local geometry -> closed-form operator -> A2r response -> optional B3 coupling -> terminal`

最准确的一句话定位是：

**当前已经完成的不是完整快慢闭环，而是冻结慢层下的快层 operator family。**

当前主线文档：

- [工程记录/PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-当前主线总览.md)

当前工程状态总图：

- [工程记录/分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)
- [工程记录/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类调试记录.md)
- [工程记录/README.md](/home/THL/project/MTS-PIA/工程记录/README.md)

## 3. 当前主线已经完成到哪一步

当前最准确的位置：

- 仍在 `P0a.1` 尾部

已经真实跑通的关键部件：

- `A2r`
- `C3LR`
- `optional B3`
- `r > 1`

当前 formal 结果锚点：

- [out/_active/verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal/pia_operator_p0a1_b3_conclusion.md](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal/pia_operator_p0a1_b3_conclusion.md)
- [out/_active/verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2/pia_operator_p0a1_b3_conclusion.md](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2/pia_operator_p0a1_b3_conclusion.md)

当前最关键判断：

- `C3LR + r4` 已能在 `NATOPS` 上追平 backbone baseline
- `B3` 在不同数据集上的行为不同
- 当前核心问题已从“轴是否学出来”转成：
  - `axis -> force`
  - `global -> local execution`
  - `operator scope`

最关键的结构证据：

- [out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/geometry_visual_summary.md](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/geometry_visual_summary.md)

## 4. 当前主线后的分流

### 4.1 `R0 / P0b-lite`

定位：

- `R0` 是 post-fast frozen-identity refit probe
- `P0b-lite` 是 one-step delayed refresh probe

关键文档：

- [工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md)
- [工程记录/PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md)

当前结论：

- 这条线是真实信号，但不稳定
- 当前不能升格为统一主线

### 4.2 `P1a`

定位：

- 当前最新前沿不是继续修全局固定算子
- 而是测试：
  - `query-conditioned local operator`

关键文档：

- [工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)
- [out/_active/verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke/pia_operator_p1a_stage1_conclusion.md](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke/pia_operator_p1a_stage1_conclusion.md)

当前最关键结果：

- `FingerMovements`
  - `baseline_0 = 0.5088`
  - `f1_global_mainline = 0.4982`
  - `p1a_s1_offline_local_wls = 0.5288`

当前判断：

- `P1a` 是最新前沿
- 它强烈暗示当前多模态轨迹任务的主矛盾之一是：
  - **全局固定算子作用域不足**

## 5. 并行分类支线

### 5.1 动态分类线

当前动态分类线已经单独收束，不再和 `PIA-Operator` 混成同一主叙事。

总览：

- [工程记录/动态主线v1框架收束.md](/home/THL/project/MTS-PIA/工程记录/动态主线v1框架收束.md)

当前推荐读法：

- `raw -> dense z_seq -> terminal`

### 5.2 SCP branch

这是一条 prototype-memory / local geometry / closed-form local update 的独立探索线。

入口文档：

- [工程记录/SCP-Branch-v0-prototype-memory-promote.md](/home/THL/project/MTS-PIA/工程记录/SCP-Branch-v0-prototype-memory-promote.md)
- [工程记录/SCP-Branch-v3-closed-form-local-update-promote.md](/home/THL/project/MTS-PIA/工程记录/SCP-Branch-v3-closed-form-local-update-promote.md)

### 5.3 no-bridge dual-stream

这条线已经工程闭环，但仍是并行验证分支。

结果入口：

- [out/_active/verify_route_b_dual_stream_no_bridge_20260329/dual_stream_no_bridge_conclusion.md](/home/THL/project/MTS-PIA/out/_active/verify_route_b_dual_stream_no_bridge_20260329/dual_stream_no_bridge_conclusion.md)

## 6. 代码入口

如果你是代码分析助手，请优先看这些文件：

### 6.1 当前 `PIA-Operator` 主线

- [route_b_unified/pia_operator_value_probe.py](/home/THL/project/MTS-PIA/route_b_unified/pia_operator_value_probe.py)
- [scripts/run_route_b_pia_operator_p0a1_c3.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_operator_p0a1_c3.py)
- [scripts/run_route_b_pia_operator_p0a1_b3.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_operator_p0a1_b3.py)
- [scripts/run_route_b_pia_operator_p1a_stage1_local_wls.py](/home/THL/project/MTS-PIA/scripts/run_route_b_pia_operator_p1a_stage1_local_wls.py)

### 6.2 动态分类线

- [route_b_unified/trajectory_representation.py](/home/THL/project/MTS-PIA/route_b_unified/trajectory_representation.py)
- [route_b_unified/trajectory_minirocket_evaluator.py](/home/THL/project/MTS-PIA/route_b_unified/trajectory_minirocket_evaluator.py)

### 6.3 历史背景代码

只在需要回溯历史时再看：

- [route_b_unified/pia_core.py](/home/THL/project/MTS-PIA/route_b_unified/pia_core.py)
- [route_b_unified/bridge.py](/home/THL/project/MTS-PIA/route_b_unified/bridge.py)
- [docs/ROUTE_B_MAIN_BODY.md](/home/THL/project/MTS-PIA/docs/ROUTE_B_MAIN_BODY.md)

## 7. 对外比较口径

从当前阶段开始，外部主对比默认只认：

- `raw + MiniROCKET`

以下对象只保留为内部归因与消融对照：

- `baseline_0`
- `same_backbone_no_shaping`
- `F1`
- `R0`
- `P0b-lite`

所以分析结果时，请优先回答：

1. 相对 `raw + MiniROCKET` 还差多少
2. 相对内部 baseline 的增益来自哪里

## 8. 不要误读的点

请不要把当前仓库误读成：

- 旧 `Route B curriculum` 仍是默认主线
- `unified shell` 仍是方法学核心
- `R0 / P0b-lite` 已经升格成默认下一代主线
- 动态分类线已经替代 `PIA-Operator` 主叙事

当前更准确的总读法是：

**`PIA-Operator` 已在 `P0a.1` 尾部站住为现役主线，`R0/P0b-lite` 是机会分支，`P1a` 是最新前沿，动态分类线已独立成平行系统路径。**

## 9. 如果你要继续协助这个项目

优先建议做这三类事：

1. 帮助分析当前 `A2r / C3 / B3` 链的真正瓶颈
2. 判断 `P1a local operator` 是否值得继续扩成更正式路线
3. 在不误回旧 `Route B / unified shell` 叙事的前提下，整理当前代码与结果

如果你只读一个结果文件，请先读：

- [out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/geometry_visual_summary.md](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/geometry_visual_summary.md)
