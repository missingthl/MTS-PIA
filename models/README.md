# Models Index

更新时间：2026-04-18

`models/` 现在按“**共享读出层 + 宿主层**”来读，不要再按历史宿主先后顺序阅读。

## 当前最短阅读顺序

如果你只关心当前 `DLCR` 主线，按这个顺序：

1. [local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py)
2. [resnet1d_local_closed_form.py](/home/THL/project/MTS-PIA/models/resnet1d_local_closed_form.py)
3. [resnet1d.py](/home/THL/project/MTS-PIA/models/resnet1d.py)
4. [patchtst_local_closed_form.py](/home/THL/project/MTS-PIA/models/patchtst_local_closed_form.py)
5. [timesnet_local_closed_form.py](/home/THL/project/MTS-PIA/models/timesnet_local_closed_form.py)
6. [tensor_cspnet_adapter.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_adapter.py)

## 目录读法

### 1. 共享读出 / 残差层

- [local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py)

这是当前最关键的文件。`DLCR` 的主要机制都在这里：
- `pinv / dual` 等求解器
- `center_only / center_subproto / flat` 几何分支
- routing 与 dataflow/closed-form probe

### 2. 通用 MTS 主宿主

- [resnet1d.py](/home/THL/project/MTS-PIA/models/resnet1d.py)
- [resnet1d_adapter.py](/home/THL/project/MTS-PIA/models/resnet1d_adapter.py)
- [resnet1d_residual_linear.py](/home/THL/project/MTS-PIA/models/resnet1d_residual_linear.py)
- [resnet1d_local_closed_form.py](/home/THL/project/MTS-PIA/models/resnet1d_local_closed_form.py)

这是当前默认主线：
- `E0`: `ResNet1D + linear`
- `E1`: `ResNet1D + residual linear`
- `E2`: `ResNet1D + DLCR`

### 3. 可更换 Transformer 宿主

- [time_series_library_reference.py](/home/THL/project/MTS-PIA/models/time_series_library_reference.py)
- [patchtst_adapter.py](/home/THL/project/MTS-PIA/models/patchtst_adapter.py)
- [patchtst_residual_linear.py](/home/THL/project/MTS-PIA/models/patchtst_residual_linear.py)
- [patchtst_local_closed_form.py](/home/THL/project/MTS-PIA/models/patchtst_local_closed_form.py)
- [timesnet_adapter.py](/home/THL/project/MTS-PIA/models/timesnet_adapter.py)
- [timesnet_residual_linear.py](/home/THL/project/MTS-PIA/models/timesnet_residual_linear.py)
- [timesnet_local_closed_form.py](/home/THL/project/MTS-PIA/models/timesnet_local_closed_form.py)

这层来自 [Time-Series-Library](/home/THL/project/MTS-PIA/archive/reference_code/Time-Series-Library) 的分类宿主适配：
- `PatchTST`
- `TimesNet`

它们现在和 `ResNet1D` 一样都能接：
- `E0`: base linear
- `E1`: residual linear
- `E2`: DLCR local closed-form readout

当前要注意的一点是：
- `PatchTST` 的原生分类 latent 很大，跑 `pinv` 会比 `ResNet1D` 更重
- 如果只是先验证宿主接入，优先从 `E0 / E1` 或 `dual_pinv` 口径开始
### 4. EEG / SPD 外部宿主

- [tensor_cspnet_adapter.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_adapter.py)
- [tensor_cspnet_residual_linear.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_residual_linear.py)
- `spdnet*`

这条线保留为外部宿主验证，不是当前通用 MTS 主线，但也不是废线。

### 5. 其他与历史宿主

- [raw_cnn1d.py](/home/THL/project/MTS-PIA/models/raw_cnn1d.py)
- [minirocket_dlcr_adapter.py](/home/THL/project/MTS-PIA/models/minirocket_dlcr_adapter.py)
- `manifold* / prototype_mdm / dual_stream*`

这些文件现在更多是：
- 边界线
- 历史宿主
- 一阶段/动态支线证据

默认不建议先从这里进入。
