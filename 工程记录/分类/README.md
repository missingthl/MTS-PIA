# 分类工程记录

更新时间：2026-04-17

这份目录只保留最小入口，并把每份入口文档的职责切开。

## 推荐阅读顺序

1. [00-入口/分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类工程现状.md)
2. [00-入口/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md)
3. [../../docs/CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
4. [../../ASSISTANT_ENTRY.md](/home/THL/project/MTS-PIA/ASSISTANT_ENTRY.md)

## 当前分类工程的官方口径

当前分类工程按双主线并列来读：

- `Tensor-CSPNet + DLCR`
  - EEG / SPD 外部宿主验证线
- `ResNet1D + DLCR`
  - 通用 MTS 主验证线

术语统一：

- `E0 / E1 / E2` 叫**框架**
- 各任务名称叫**数据集**

## 各入口的职责

### `00-入口/分类工程现状.md`

唯一权威状态页，负责：

- 当前双主线定义
- 已核实的结果快照
- 当前最优已核实配置
- 当前未闭环的问题

### `00-入口/分类调试记录.md`

只负责记录：

- 已完成诊断
- 阶段性实验转折点
- 失败分支和排障结论

它不再承担“当前总览”职责。

### `01-阶段二-宿主实验`

只保留宿主专项任务单。  
其中 `Tensor-CSPNet` 任务单用于解释 Tensor 线本身，不再替代整个阶段二入口。

### `02-阶段一-PIA-Operator`

结构证据层。用于解释：

- 为什么 `same/opp` 成立
- 为什么会转向 `local / residual / query-conditioned` 思路

### `03-04`

动态支线和 SCP 支线保留，但不是默认阅读入口。

## 当前默认不写进主线的内容

以下内容当前保留，但不并入主线 ranking：

- `MiniRocket + DLCR`
  - 边界 / 诊断线
- `MBA_ManifoldBridge`
  - standalone 项目
- 未跟踪的 `standalone_projects/RouteB_GeometricAugmentation/`
