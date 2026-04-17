# 工程记录 README

更新时间：2026-04-17

`工程记录/` 现在只负责提供**最小阅读入口**，不再承担完整结果总览职责。

## 最短阅读顺序

1. [分类/00-入口/分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类工程现状.md)
2. [分类/00-入口/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md)
3. [../docs/CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
4. [../ASSISTANT_ENTRY.md](/home/THL/project/MTS-PIA/ASSISTANT_ENTRY.md)

## 当前怎么理解分类工程

当前分类工程按“双主线并列”来读：

- `Tensor-CSPNet + DLCR`
  - EEG / SPD 外部宿主验证线
- `ResNet1D + DLCR`
  - 通用 MTS 主验证线

请不要再把 `Tensor-CSPNet` 任务单当成整个阶段二的唯一入口。

## 目录职责

- `分类/00-入口`
  - 最小入口集
  - 其中：
    - `分类工程现状` 是唯一权威状态页
    - `分类调试记录` 是阶段性诊断时间线
- `分类/01-阶段二-宿主实验`
  - 宿主线专项任务单
- `分类/02-阶段一-PIA-Operator`
  - 一阶段结构证据层
- `分类/03-并行支线-动态`
  - 动态 / 轨迹支线
- `分类/04-并行支线-SCP`
  - SCP 支线
- `分类/90-历史与草案`
  - 不建议优先读

## 当前整理原则

- 只在 `分类工程现状` 里写权威当前结论
- 只写当前仓库里已核实的结果
- 不把未跟踪目录或独立 standalone 项目混进主线入口
