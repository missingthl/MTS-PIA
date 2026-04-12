# Current Engineering Map

更新时间：2026-04-12

这份文件现在优先回答当前工程的**阶段地图**，而不是只回答一阶段 `PIA-Operator`。

## 1. 当前工程的默认读法

当前分类工程请按下面四层理解：

1. **二阶段外部宿主验证层**
2. **一阶段内部几何证据层**
3. **并行分类支线**
4. **历史背景层**

其中当前活跃实验层已经切到：

- 二阶段外部宿主验证层

最短入口：

- [工程记录/分类文档导航总表.md](../工程记录/分类文档导航总表.md)
- [工程记录/分类二阶段现状.md](../工程记录/分类二阶段现状.md)
- [工程记录/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](../工程记录/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)

## 2. 当前活跃实验是什么

当前活跃路线固定为：

`Tensor-CSPNet backbone -> Temporal_Block latent -> local closed-form residual head -> residual fusion`

当前协议：

- `BCIC-IV-2a holdout`

当前实验矩阵：

- `E0`
- `E1`
- `E2`

当前代码入口：

- [models/tensor_cspnet_adapter.py](../models/tensor_cspnet_adapter.py)
- [models/tensor_cspnet_residual_linear.py](../models/tensor_cspnet_residual_linear.py)
- [models/local_closed_form_residual_head.py](../models/local_closed_form_residual_head.py)
- [scripts/run_tensor_cspnet_local_closed_form_holdout.py](../scripts/run_tensor_cspnet_local_closed_form_holdout.py)

## 3. 一阶段现在是什么

一阶段当前不再是默认对外方法形态，而是结构证据层。

当前最重要的一阶段结论有三条：

1. `A2r + C3LR + r>1 (+ optional B3)` 已把快层 operator family 站到 `P0a.1` 尾部
2. `R0 / P0b-lite` 不是统一稳定赢家
3. `P1a` 很强地支持：
   - 当前瓶颈之一是 `global fixed operator scope`

当前最重要文档：

- [工程记录/阶段一结构证据总览.md](../工程记录/阶段一结构证据总览.md)
- [工程记录/PIA-Operator-当前主线总览.md](../工程记录/PIA-Operator-当前主线总览.md)
- [工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md](../工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md)
- [工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](../工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)

## 4. 并行支线

当前仍保留，但不作为默认活跃主线：

- [工程记录/动态主线v1框架收束.md](../工程记录/动态主线v1框架收束.md)
- `SCP branch`
- no-bridge dual-stream

## 5. 当前最该看的结果

### 二阶段

- 当前已完成：
  - 宿主 `Tensor-CSPNet` 复现
  - `E0 / E1 / E2` 最小 smoke
- 当前正在做：
  - `BCIC holdout` 单 seed 全量比较

### 一阶段

- [verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal](../out/_active/verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal)
- [verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2](../out/_active/verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2)
- [verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke](../out/_active/verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke)

## 6. 当前不该再怎么读

请不要再把当前工程地图误读成：

- 仍以旧 `Route B / unified shell` 为默认主体
- 一阶段 `PIA-Operator` 仍是唯一活跃实验载体
- 动态分类线已经接管默认主线

当前更准确的读法是：

**当前活跃层已经切到“外部宿主上的端到端局部闭式残差层验证”；一阶段 `PIA-Operator` 保留为几何与算子证据层。**
