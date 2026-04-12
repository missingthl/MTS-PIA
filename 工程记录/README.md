# 工程记录 README

更新时间：2026-04-12

这份目录现在按**阶段角色**阅读，不再按“时间先后”或“文档发表顺序”阅读。

## 当前默认入口

如果只想知道当前分类工程真实在做什么，请先看这 4 份：

1. [分类二阶段现状.md](/home/THL/project/MTS-PIA/工程记录/分类二阶段现状.md)
2. [Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
3. [分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类调试记录.md)
4. [分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)

当前最重要的读法是：

- **二阶段**
  - 当前活跃实验层
  - 使用外部宿主 baseline 做端到端方法验证
- **一阶段**
  - 当前结构证据层
  - 用于解释二阶段方法为什么这样设计

## 阶段分层

### 阶段二：当前活跃主线

当前二阶段固定为：

- 宿主：
  - `Tensor-CSPNet`
- 方法原型：
  - `Temporal_Block latent + local closed-form residual head`
- 当前协议：
  - `BCIC-IV-2a holdout`

当前最重要文档：

- [分类二阶段现状.md](/home/THL/project/MTS-PIA/工程记录/分类二阶段现状.md)
- [Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)
- [分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类调试记录.md)

### 阶段一：当前保留的结构证据层

阶段一当前不再承担“默认对外方法形态”的职责，而主要负责：

- 提供局部几何与闭式算子的内部证据
- 解释为什么当前二阶段要转向 `local / bipolar / residual / end-to-end`

当前最重要文档：

- [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-当前主线总览.md)
- [PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md)
- [PIA-Operator-R0-多数据集稳定性阶段小结.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md)
- [PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)

### 并行支线

仍保留，但不作为当前默认主叙事：

- [动态主线v1框架收束.md](/home/THL/project/MTS-PIA/工程记录/动态主线v1框架收束.md)
- `SCP branch` 文档族
- no-bridge dual-stream 结果层

## 当前比较口径

当前开始需要把比较口径分开看：

### 一阶段口径

一阶段内部分类主结论默认仍优先对齐：

- `raw + MiniROCKET`

`baseline_0 / same_backbone_no_shaping / F1 / R0 / P0b-lite` 继续只作为内部归因对照。

### 二阶段口径

二阶段当前不是直接和 `raw + MiniROCKET` 比，而是先回答：

1. 原始宿主 `Tensor-CSPNet` 是否可靠复现
2. `E2` 是否优于 `E1`
3. `E2` 是否优于原始宿主 `E0`

也就是说，当前二阶段默认主问题是：

**局部闭式判别残差层能否稳定增强已发表宿主。**

## 当前不再作为默认入口的文档

以下文档仍有参考价值，但不应再作为当前默认总览：

- [分类框架全貌.md](/home/THL/project/MTS-PIA/工程记录/分类框架全貌.md)
  - 当前已降为阶段一历史框架地图
- [PIA-Operator-P0a.1-Stage-C-next-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P0a.1-Stage-C-next-promote.md)
  - 阶段性失败归因
- 旧 `Route B / unified shell` 相关文档

## 非分类文档提醒

以下文档仍有效，但不属于当前分类主地图：

- [时间序列工程现状.md](/home/THL/project/MTS-PIA/工程记录/时间序列工程现状.md)
- [时间序列蓝图.md](/home/THL/project/MTS-PIA/工程记录/时间序列蓝图.md)
- [时间序列调试记录.md](/home/THL/project/MTS-PIA/工程记录/时间序列调试记录.md)

## 当前整理原则

- 不删除阶段一原文档
- 默认入口先落到二阶段
- 阶段一统一降到“结构证据层”
- 当前新增的阶段二结果与实现，优先写进：
  - [分类二阶段现状.md](/home/THL/project/MTS-PIA/工程记录/分类二阶段现状.md)
  - [分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类调试记录.md)
