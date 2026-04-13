# Scripts Index

更新时间：2026-04-13

现在 `scripts/` 根目录只保留统一入口。

如果你只关心当前工程，先只记住四层：

- [hosts](/home/THL/project/MTS-PIA/scripts/hosts)
  二阶段宿主实验入口
- [route_b](/home/THL/project/MTS-PIA/scripts/route_b)
  一阶段与并行支线的结构证据 runner
- [raw_baselines](/home/THL/project/MTS-PIA/scripts/raw_baselines)
  raw / bridge / MiniROCKET 基线与宿主脚本
- [support](/home/THL/project/MTS-PIA/scripts/support)
  仍在被现役代码复用的共享工具
- [legacy_phase](/home/THL/project/MTS-PIA/scripts/legacy_phase)
  `phase9-16` 历史沉积层

## 当前最该看的入口

如果你现在只关心当前分类主线，请按这个顺序：

1. [hosts/run_tensor_cspnet_local_closed_form_holdout.py](/home/THL/project/MTS-PIA/scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py)
2. [hosts/run_resnet1d_local_closed_form_fixedsplit.py](/home/THL/project/MTS-PIA/scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py)
3. [run_in_pia.sh](/home/THL/project/MTS-PIA/scripts/run_in_pia.sh)
4. 其余目录按角色再进入，不要再从根目录平扫脚本文件名

## 目录说明

### `hosts`

当前二阶段宿主层。

- `Tensor-CSPNet + E0/E1/E2`
- `ResNet1D + E0/E1/E2`

### `route_b`

阶段一 `PIA-Operator`、`PIA-Core`，以及动态 / SCP 支线。

这些脚本现在的角色是：

- 结构证据
- 历史主线复盘
- 并行支线继续实验

它们不再和当前二阶段宿主实验混在根目录里。

### `raw_baselines`

原始 raw/bridge/MiniROCKET 宿主与基线层。

这里保留：

- `run_raw_bridge_probe.py`
- `run_bridge_curriculum_pilot.py`
- `run_raw_minirocket_baseline.py`
- 相关 fixed-split / seed-family 版本

### `support`

共享工具层。

已经从根目录抽出去的典型工具有：

- `fisher_pia_utils.py`
- `lraes_utils.py`
- `protocol_split_utils.py`
- `resource_probe_utils.py`
- `local_knn_gate.py`

这层的目的就是避免“拿 runner 当库函数”。

### 其他目录

其余目录都已经下沉成辅助层：

- `analysis`
- `data_prep`
- `probes`
- `forecast`
- `manifold`
- `seed_suites`
- `devtools`
- `legacy_phase`

## 当前阅读顺序

1. [README.md](/home/THL/project/MTS-PIA/README.md)
2. [docs/CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
3. [工程记录/分类/README.md](/home/THL/project/MTS-PIA/工程记录/分类/README.md)
4. [scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py](/home/THL/project/MTS-PIA/scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py)
5. 再按需要进入 `route_b / raw_baselines / support`
6. 最后才进入其余辅助目录

## 当前清理后的边界

这次已经完成了三件最关键的事：

1. `phase*` 历史沉积层已经物理下沉
2. 现役代码对部分旧脚本公共函数的依赖，已经开始抽到稳定模块
3. 根目录里的杂项脚本已经继续按功能分桶，不再平铺

所以现在的 `scripts/` 读法已经和以前不同：

- 根目录：只保留统一入口与少量跨层脚本
- 子目录：按角色、实验层级和用途分层阅读
