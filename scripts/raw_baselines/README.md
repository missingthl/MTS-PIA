# Raw Baselines

这里放 raw / bridge / MiniROCKET 宿主与基线脚本。

这一层的角色是：

- 给 `route_b` 系列提供 raw 域评价器
- 给当前工程提供 raw + MiniROCKET 对照
- 保留 bridge / raw baseline 的独立复现实验入口

关键脚本：

- [run_raw_bridge_probe.py](/home/THL/project/MTS-PIA/scripts/raw_baselines/run_raw_bridge_probe.py)
- [run_bridge_curriculum_pilot.py](/home/THL/project/MTS-PIA/scripts/raw_baselines/run_bridge_curriculum_pilot.py)
- [run_raw_minirocket_baseline.py](/home/THL/project/MTS-PIA/scripts/raw_baselines/run_raw_minirocket_baseline.py)
