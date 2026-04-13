# Scripts Index

更新时间：2026-04-13

这次整理后，`scripts/` 不再把所有年代、所有角色的脚本平铺在同一层。

现在默认把脚本分成十层：

- [hosts](/home/THL/project/MTS-PIA/scripts/hosts)
  二阶段宿主实验入口
- [route_b](/home/THL/project/MTS-PIA/scripts/route_b)
  一阶段与并行支线的结构证据 runner
- [raw_baselines](/home/THL/project/MTS-PIA/scripts/raw_baselines)
  raw / bridge / MiniROCKET 基线与宿主脚本
- [support](/home/THL/project/MTS-PIA/scripts/support)
  仍在被现役代码复用的共享工具
- [analysis](/home/THL/project/MTS-PIA/scripts/analysis)
  结果聚合、报表重建、分析型 probe
- [data_prep](/home/THL/project/MTS-PIA/scripts/data_prep)
  数据准备与资产体检
- [probes](/home/THL/project/MTS-PIA/scripts/probes)
  轻量排障与小探针
- [forecast](/home/THL/project/MTS-PIA/scripts/forecast)
  时间序列预测副线
- [manifold](/home/THL/project/MTS-PIA/scripts/manifold)
  原始流形数据检查
- [seed_suites](/home/THL/project/MTS-PIA/scripts/seed_suites)
  `SEED/SEED-V` 批处理套件
- [devtools](/home/THL/project/MTS-PIA/scripts/devtools)
  工程维护与环境自检
- [legacy_phase](/home/THL/project/MTS-PIA/scripts/legacy_phase)
  `phase9-16` 历史沉积层

## 当前最该看的入口

如果你现在只关心当前分类主线，请按这个顺序：

1. [hosts/README.md](/home/THL/project/MTS-PIA/scripts/hosts/README.md)
2. [hosts/run_tensor_cspnet_local_closed_form_holdout.py](/home/THL/project/MTS-PIA/scripts/hosts/run_tensor_cspnet_local_closed_form_holdout.py)
3. [hosts/run_resnet1d_local_closed_form_fixedsplit.py](/home/THL/project/MTS-PIA/scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py)
4. [run_in_pia.sh](/home/THL/project/MTS-PIA/scripts/run_in_pia.sh)
5. 其余目录按角色再进入，不要再从根目录平扫脚本文件名

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

### `legacy_phase`

`phase9-16` 历史脚本整体下沉后的存放层。

这里仍然保留全部历史脚本，但默认不建议从这里起读。  
需要阶段一历史细节时，再进入这一层。

### `analysis / probes`

这两层解决“实验太多但没有读法”的问题：

- `analysis` 负责聚合与解释
- `probes` 负责快速排障与小验证

它们都不应该和正式 runner 混作一个层级理解。

### `data_prep / seed_suites / devtools`

这三层把之前最影响可读性的杂项脚本整体下沉：

- `data_prep`：数据与资产准备
- `seed_suites`：批量 shell / worker
- `devtools`：环境与工程维护

### `forecast / manifold`

这些是并行副线或数据诊断层：

- `forecast`：时间序列预测
- `manifold`：原始流形检查

## 当前阅读顺序

1. [README.md](/home/THL/project/MTS-PIA/README.md)
2. [docs/CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
3. [工程记录/分类/README.md](/home/THL/project/MTS-PIA/工程记录/分类/README.md)
4. [scripts/hosts/README.md](/home/THL/project/MTS-PIA/scripts/hosts/README.md)
5. 再按需要进入 `route_b / raw_baselines / support`
6. 最后才进入 `analysis / probes / legacy_phase`

## 当前清理后的边界

这次已经完成了三件最关键的事：

1. `phase*` 历史沉积层已经物理下沉
2. 现役代码对部分旧脚本公共函数的依赖，已经开始抽到稳定模块
3. 根目录里的杂项脚本已经继续按功能分桶，不再平铺

所以现在的 `scripts/` 读法已经和以前不同：

- 根目录：只保留统一入口与少量跨层脚本
- 子目录：按角色、实验层级和用途分层阅读
