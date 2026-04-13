# ResNet1D tsai 对齐与接入说明

更新时间：2026-04-13

## 1. 上游来源

当前参考实现来自：

- [archive/reference_code/tsai](/home/THL/project/MTS-PIA/archive/reference_code/tsai)

主要核对文件：

- [archive/reference_code/tsai/tsai/models/ResNet.py](/home/THL/project/MTS-PIA/archive/reference_code/tsai/tsai/models/ResNet.py)
- [archive/reference_code/tsai/tsai/models/layers.py](/home/THL/project/MTS-PIA/archive/reference_code/tsai/tsai/models/layers.py)
- [archive/reference_code/tsai/README.md](/home/THL/project/MTS-PIA/archive/reference_code/tsai/README.md)
- [archive/reference_code/tsai/tsai/data/external.py](/home/THL/project/MTS-PIA/archive/reference_code/tsai/tsai/data/external.py)

## 2. tsai 标准 ResNet 结构

`tsai` 中的标准 `ResNet` 结构为：

- 3 个残差块
- 通道数：`64 -> 128 -> 128`
- 每个块 3 层卷积
- 默认卷积核：`7 / 5 / 3`
- 最后接 `AdaptiveAvgPool1d(1)` + `Linear`

需要特别注意：

- 当前 `tsai` 标准实现默认不是 `8 / 5 / 3`，而是 `7 / 5 / 3`
- 当输入通道与输出通道相同时，shortcut 不是 `Identity`，而是 `BN1d`
- 这是我们接入时需要明确对齐的地方

## 3. 本仓接入策略

当前仓库内已经新增 repo-native 宿主实现：

- [models/resnet1d.py](/home/THL/project/MTS-PIA/models/resnet1d.py)
- [models/resnet1d_adapter.py](/home/THL/project/MTS-PIA/models/resnet1d_adapter.py)
- [models/resnet1d_residual_linear.py](/home/THL/project/MTS-PIA/models/resnet1d_residual_linear.py)
- [models/resnet1d_local_closed_form.py](/home/THL/project/MTS-PIA/models/resnet1d_local_closed_form.py)
- [scripts/run_resnet1d_local_closed_form_fixedsplit.py](/home/THL/project/MTS-PIA/scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py)

当前默认口径已经按 `tsai` 调整为：

- kernel sizes: `7 / 5 / 3`
- block channels: `64 / 128 / 128`
- same-channel shortcut: `BN1d`

## 4. tsai 常用实验口径

从 `README` 与 `TSLearner` 文档可读到，`tsai` 在时间序列分类上的常见使用方式是：

- 数据入口：
  - `get_classification_data(...)`
  - 或 `get_UCR_data(...)`
- 数据来源：
  - UCR/UEA classification datasets
- 训练入口：
  - `TSClassifier(...)`
- 常见默认：
  - optimizer: `Adam`
  - lr: `1e-3`
  - metrics: `accuracy`
  - batch size: `[64, 128]`

README 中给出的示例：

- 单变量分类：`ECG200`
- 多变量分类：`LSST`

文档与 notebook 中还直接出现过：

- `NATOPS`
- `FingerMovements`

## 5. 与当前仓库的数据集重合

`tsai` 的 UCR/UEA 分类集合与本仓当前 trial 工厂重合度很高。

已确认重合的数据集包括：

- `NATOPS`
- `FingerMovements`
- `SelfRegulationSCP1`
- `BasicMotions`
- `HandMovementDirection`
- `UWaveGestureLibrary`
- `Epilepsy`
- `AtrialFibrillation`
- `PenDigits`

这意味着：

- 我们不必为了接入 `ResNet1D` 再单独引入一套新数据层
- 可以直接复用当前仓库已有 fixed-split trial loader

## 6. 当前方法学定位

当前 `ResNet1D` 接入的目标，不是复刻 `tsai` 全套 fastai/learner 训练框架，而是：

- 引入一个标准、可微、通用一维宿主
- 让现有 `E0 / E1 / E2` 三臂在通用 TSC backbone 上可直接复用

对应关系为：

- `E0`: `ResNet1D + global linear head`
- `E1`: `ResNet1D + residual linear head`
- `E2`: `ResNet1D + DLCR`

## 7. 下一步建议

第一轮不要急着覆盖全部数据集，建议优先做 fixed-split smoke：

1. `NATOPS`
2. `FingerMovements`
3. `SelfRegulationSCP1`

第一轮 `E2` 默认建议：

- `support_mode = same_only`
- `prototype_aggregation = pooled`
- `local_readout_gate = none`

原因是当前 Tensor-CSPNet 二阶段实验里，`same_only` 比更复杂的 `same/opp` 构造更稳。
