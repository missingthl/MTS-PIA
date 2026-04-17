# Assistant Entry

更新时间：2026-04-17

这份文件给外部语言助手一个**最短、最不容易读偏**的仓库入口。

## 先记住这 4 条

1. 当前分类线是**双主线并列**：
   - `Tensor-CSPNet + DLCR`：EEG/SPD 外部宿主验证线
   - `ResNet1D + DLCR`：当前通用 MTS 主验证线
2. 当前默认阅读优先级是：**先看 `ResNet1D`，但不要把 `Tensor-CSPNet` 当作废线。**
3. 公开入口里只写**已核实主结果**：
   - 必须能直接回溯到 `summary.json / history.csv / log / report`
4. `MiniRocket + DLCR` 是**边界/诊断线**，`MBA_ManifoldBridge` 是**standalone 项目**，都不应被直接写成当前主仓库分类主线排名的一部分。

## 最短阅读路径

1. [README.md](/home/THL/project/MTS-PIA/README.md)
2. [CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
3. [分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类工程现状.md)
4. [分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md)
5. [Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)

## 当前双主线怎么读

### 1. `Tensor-CSPNet + DLCR`

定位：

- EEG/SPD 外部宿主验证线
- 作用是验证 `DLCR` 不是只能活在自家一阶段代码里

当前已核实结果：

- `BCIC-IV-2a holdout` 宿主复现约 `0.7238`
- `fp64 mainline`
  - `E0 = 0.7103`
  - `E1 = 0.7083`
  - `E2 = 0.7018`

当前读法：

- 宿主可靠
- `E2` 在这个宿主上尚未建立优势

权威入口：

- [stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md)

### 2. `ResNet1D + DLCR`

定位：

- 当前通用 MTS 主验证线
- 当前公开结果与后续框架升级，默认先读这条线

当前已核实结果：

- `E2 tau=0.2 corrected` 21 数据集 sweep 已完成
- 相对当前 `E0`：`15 胜 / 1 平 / 5 负`

代表性提点：

- `NATOPS`: `0.9056 -> 0.9389`
- `FingerMovements`: `0.5200 -> 0.5800`
- `UWaveGestureLibrary`: `0.7594 -> 0.8313`
- `Heartbeat`: `0.3659 -> 0.5951`
- `Epilepsy`: `0.9275 -> 0.9855`

当前读法：

- `DLCR` 已经不是想法验证，而是系统验证主线
- 但 `tau=0.2` 不能被写成统一最优默认值

权威入口：

- [run_e2_tau02_fullscale_20260414_v2Corrected.log](/home/THL/project/MTS-PIA/out/_active/run_e2_tau02_fullscale_20260414_v2Corrected.log)
- [verify_e2_tau02_fullscale_20260414](/home/THL/project/MTS-PIA/out/_active/verify_e2_tau02_fullscale_20260414)
- [run_3baseline_batch_sys_20260414_v4.log](/home/THL/project/MTS-PIA/out/_active/run_3baseline_batch_sys_20260414_v4.log)

## 温度与机制读法

不要再默认写：

- “`tau=0.2` 是当前最优默认配置”

当前仓库已核实的结论是：

- `NATOPS`: `tau=0.5` 最优
- `FingerMovements`: `tau=0.2` 最优
- `SelfRegulationSCP1`: `tau=0.5` 最优
- `UWaveGestureLibrary`: `tau=1.0` 最优

所以更准确的说法是：

- 温度是框架内生变量
- 但当前表现出明显数据集依赖

证据入口：

- [run_subproto_temp_sweep_20260414.log](/home/THL/project/MTS-PIA/out/_active/run_subproto_temp_sweep_20260414.log)

## 哪些东西不要读偏

### `MiniRocket + DLCR`

定位：

- 边界/诊断线

可以说：

- 它用于检查 `DLCR` 是否强依赖宿主协同演化

不要说：

- “MiniRocket 全量官方 baseline / SOTA 对比已经完整完成”

原因：

- 当前公开聚合工件还不完整，不能支撑这个说法

### `MBA_ManifoldBridge`

定位：

- 独立 standalone 项目

可以说：

- 它有自己的报告与发布轨道

不要说：

- 它属于当前主仓库分类主线的排名对照

## 当前最该避免的 4 个误读

1. 不要把 `Tensor-CSPNet` 写成当前单一活跃主线。
2. 不要把 `ResNet1D` 写成“还在预备接入”的宿主。
3. 不要把未核实的 `MiniRocket` 胜场写成既成事实。
4. 不要把“三基线已全部闭环”当成当前已完成结论。

## 一句话版

如果你只能记一句：

**当前分类仓库应按“双主线并列、`ResNet1D` 优先阅读、只写已核实主结果、`MiniRocket` 与 `MBA` 不并入主线排名”来理解。**
