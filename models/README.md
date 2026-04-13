# Models Index

当前 `models/` 主要分成三类：

- 宿主 backbone
  - [resnet1d.py](/home/THL/project/MTS-PIA/models/resnet1d.py)
  - [raw_cnn1d.py](/home/THL/project/MTS-PIA/models/raw_cnn1d.py)
  - `spdnet*`
- 宿主适配层
  - [tensor_cspnet_adapter.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_adapter.py)
  - [resnet1d_adapter.py](/home/THL/project/MTS-PIA/models/resnet1d_adapter.py)
- 读出 / 残差层
  - [local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py)
  - [tensor_cspnet_residual_linear.py](/home/THL/project/MTS-PIA/models/tensor_cspnet_residual_linear.py)
  - [resnet1d_residual_linear.py](/home/THL/project/MTS-PIA/models/resnet1d_residual_linear.py)
  - [resnet1d_local_closed_form.py](/home/THL/project/MTS-PIA/models/resnet1d_local_closed_form.py)

如果你现在只关心二阶段：

1. 先看 [local_closed_form_residual_head.py](/home/THL/project/MTS-PIA/models/local_closed_form_residual_head.py)
2. 再看宿主 adapter
3. 最后看具体 backbone
