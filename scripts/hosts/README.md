# Hosts

这里放当前二阶段宿主实验入口。

当前两个核心入口：

- [run_tensor_cspnet_local_closed_form_holdout.py](/home/THL/project/MTS-PIA/scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py)
- [run_resnet1d_local_closed_form_fixedsplit.py](/home/THL/project/MTS-PIA/scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py)

默认阅读顺序：

1. Tensor-CSPNet 宿主
2. ResNet1D 宿主

共同特征：

- 都支持 `E0 / E1 / E2`
- 都以“宿主 backbone + 读出层替换/增强”为主
- 都服务于当前二阶段方法验证
