# Support Utilities

这里放被多条实验线复用的共享工具。

当前已经抽出的主要公共模块：

- [fisher_pia_utils.py](/home/THL/project/MTS-PIA/scripts/support/fisher_pia_utils.py)
- [lraes_utils.py](/home/THL/project/MTS-PIA/scripts/support/lraes_utils.py)
- [protocol_split_utils.py](/home/THL/project/MTS-PIA/scripts/support/protocol_split_utils.py)
- [resource_probe_utils.py](/home/THL/project/MTS-PIA/scripts/support/resource_probe_utils.py)
- [local_knn_gate.py](/home/THL/project/MTS-PIA/scripts/support/local_knn_gate.py)

目的：

- 减少“runner 直接互相 import”的情况
- 让共享逻辑有稳定落点
