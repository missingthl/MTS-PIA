# 工程记录 README

更新时间：2026-04-10

这份目录不再默认按“时间顺序”阅读，而是按“当前是否仍是主线”阅读。

## 当前默认入口

如果只想知道当前分类工程做到哪一步、代码现在在跑什么、哪些分支已经降级为并行线，先看：

1. [分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)
2. [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-当前主线总览.md)
3. [动态主线v1框架收束.md](/home/THL/project/MTS-PIA/工程记录/动态主线v1框架收束.md)
4. [PIA-Operator-R0-多数据集稳定性阶段小结.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md)
5. [PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)

## 外部对比契约

从当前阶段开始，所有涉及“主结果 / SOTA 位置 / 对外结论”的比较，默认只认：

- `raw + MiniROCKET`

更准确地说：

- 若该数据集已有官方 fixed-split `raw + MiniROCKET` 结果，则它是唯一默认外部对比对象
- 若暂时没有官方 fixed-split 结果，才允许退回到历史 `raw_minirocket_baseline` family，并且必须显式标注这是次级口径

以下对象从现在开始只保留为**内部消融 / 归因对照**，不再充当主 headline：

- `baseline_0`
- `same_backbone_no_shaping`
- `F1`
- `R0`
- `P0b-lite`

也就是说：

- 可以继续和这些内部臂比较
- 但主结论不能再写成“优于 baseline 就成立”
- 必须优先回答“相对 `raw + MiniROCKET` 在哪里”

## 当前分类主线文档

- [分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)
  - 当前分类工程总地图
  - 用于回答“现役主线、动态线、并行分支、历史背景分别是什么”
- [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-当前主线总览.md)
  - 当前 `PIA-Operator` 默认总览
  - 用于回答“快层 operator 主线怎么实现”和“现在在完整愿景的哪一步”
- [动态主线v1框架收束.md](/home/THL/project/MTS-PIA/工程记录/动态主线v1框架收束.md)
  - 当前动态分类线总览
  - 用于回答“动态表示路径是否已独立成线”
- [PIA-Operator-P0a.1-Computational-Repair-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P0a.1-Computational-Repair-promote.md)
  - `P0a.1` 母文档
  - 解释为什么先修计算链，再谈 refresh
- [PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md)
  - 当前判别闭式解 probe 的正式设计稿
  - 是现有 Stage B 之前最直接的理论入口
- [PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md)
  - 当前默认下一步系统验证
  - 用于回答“当前快层是否已经值得进入一次 delayed geometry rebuild”
- [PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)
  - 下一阶段局部切空间条件算子 probe
  - 用于回答“全局固定算子是否该升级为 query-conditioned local operator”

## 当前保留但不再作为默认入口的分类文档

- [PIA-Operator-P0a.1-Stage-C-next-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-P0a.1-Stage-C-next-promote.md)
  - `same-only + 修 Lambda` 的阶段性失败/收束记录
  - 仍有归因价值，但不是当前默认主线
- [PIA-Operator-Necessity-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-Necessity-Probe-promote.md)
  - 更早的必要性与比较框架文档
  - 仍可回看，但不适合作为当前状态入口
- [分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类调试记录.md)
  - 已完成实验与短结论清单
  - 可查结果，但不宜充当当前状态总览

## 非分类文档提醒

以下文档仍有效，但不属于当前分类记录主地图：

- [时间序列工程现状.md](/home/THL/project/MTS-PIA/工程记录/时间序列工程现状.md)
- [时间序列蓝图.md](/home/THL/project/MTS-PIA/工程记录/时间序列蓝图.md)
- [时间序列调试记录.md](/home/THL/project/MTS-PIA/工程记录/时间序列调试记录.md)

## 文档分层建议

可以把现有文档粗分成三层：

- `当前主线`
  - 当前代码真实在跑的分类链
  - 当前 formal 结果对应的设计与状态总览
- `分流 / 前沿`
  - 比如 `R0 / P0b-lite / P1a`
  - 它们解释“当前主线之后正在往哪里分叉”
- `并行支线`
  - 比如 `动态主线 v1`
  - `SCP branch`
  - `dual-stream`
- `阶段性归因`
  - 比如 `C_next`
  - 它们解释“为什么某条路没有继续推进”
- `历史地图 / 大草案`
  - 比如旧 `Route B / unified shell` 文档
  - 用于回溯，不用于当前默认阅读

## 当前压缩原则

- 不直接删除历史文档
- 默认入口只保留少数几份
- 其余文档通过顶部“阅读提示”降权
- 以后新增分类 promote 时：
  - 先更新 [分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)
  - 再决定是否需要同步补到 [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-当前主线总览.md)
