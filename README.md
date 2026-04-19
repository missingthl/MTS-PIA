# MTS-PIA Workspace

更新时间：2026-04-17

这个仓库现在应该按“**双主线并列 + 证据分层**”来读，而不是按单一历史主线来读。

## 当前最短入口

如果你想最快接住当前工程，只看下面 5 份：

1. [docs/CURRENT_ENGINEERING_MAP.md](/home/THL/project/MTS-PIA/docs/CURRENT_ENGINEERING_MAP.md)
2. [工程记录/分类/00-入口/分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类工程现状.md)
3. [工程记录/分类/00-入口/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md)
4. [ASSISTANT_ENTRY.md](/home/THL/project/MTS-PIA/ASSISTANT_ENTRY.md)
5. [工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](/home/THL/project/MTS-PIA/工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)

运行资源与服务器隔离说明看：

- [SERVER_RESOURCE_GUIDE.md](/home/THL/project/MTS-PIA/SERVER_RESOURCE_GUIDE.md)

## 当前官方口径

当前分类工程有两条同级主线：

- `Tensor-CSPNet + DLCR`
  - EEG / SPD 外部宿主验证线
  - 当前协议是 `BCIC-IV-2a holdout`
- `ResNet1D + DLCR`
  - 通用 MTS 主验证线
  - 当前协议是 fixed-split 多数据集框架对比

这里的术语统一为：

- `E0 / E1 / E2` 叫**框架**
- `NATOPS / FingerMovements / ...` 叫**数据集**

## 当前已核实的主结果

### `Tensor-CSPNet`

- 外部宿主复现已完成，`BCIC-IV-2a holdout` 平均准确率约 `0.7238`
- 当前单 seed 对比读法：
  - `E0 = 0.7103`
  - `E1 = 0.7083`
  - `E2 = 0.7018`
- 当前结论：
  - 宿主可靠
  - `E2` 还没有在该宿主上建立优势

### `ResNet1D`

- `E2 tau=0.2 corrected` 已完成 21 个数据集的全量 sweep
- 结合当前仓库里可直接核对的 `E0` 日志：
  - `E2` 相对 `E0`：**15 个数据集提点 / 1 个打平 / 5 个回落**
- 代表性结果：
  - `NATOPS`: `0.9056 -> 0.9389`
  - `FingerMovements`: `0.5200 -> 0.5800`
  - `UWaveGestureLibrary`: `0.7594 -> 0.8313`
  - `Heartbeat`: `0.3659 -> 0.5951`
- 当前温度结论已经更新：
  - `tau=0.2` 不是统一最优默认值
  - 四数据集温度扫描显示：
    - `NATOPS`: `tau=0.5` 最优
    - `FingerMovements`: `tau=0.2` 最优
    - `SelfRegulationSCP1`: `tau=0.5` 最优
    - `UWaveGestureLibrary`: `tau=1.0` 最优

## 当前边界与独立项目

- `MiniRocket + DLCR`
  - 当前定位为**边界 / 诊断线**
  - 已完成 4 个数据集的 frozen-feature 诊断
  - 但当前仓库里的官方 MiniRocket 全量汇总工件还不完整，所以不把它写成已闭环主结论

- `MBA_ManifoldBridge`
  - 当前定位为**独立 standalone 项目**
  - 不并入当前主仓库分类主线 ranking

## 当前怎么读目录

当前优先看的目录：

- `models/`
- `scripts/`
- `docs/`
- `工程记录/`
- `out/`

默认不建议先从这些地方入手：

- `archive/`
- `standalone_projects/`
- `route_b_unified/`（除非你是在追一阶段证据）

## 当前最需要注意的事

这轮整理后的公开口径只写**已核实主结果**。  
任何无法由当前仓库中 `summary.json / history.csv / log / report` 直接支撑的说法，都不再作为权威入口结论。
