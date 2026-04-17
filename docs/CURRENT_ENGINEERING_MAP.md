# Current Engineering Map

更新时间：2026-04-17

这份文件只回答两件事：

1. 当前仓库的官方主线是什么
2. 目录和入口应该先看哪里

## 当前工程的四层结构

1. **分类二阶段外部宿主验证层**
2. **分类一阶段结构证据层**
3. **并行分类支线**
4. **standalone 与历史层**

其中当前活跃层是：

- **分类二阶段外部宿主验证层**

## 当前双主线

当前分类二阶段不是单一宿主，而是双主线并列：

- `Tensor-CSPNet + DLCR`
  - EEG / SPD 外部宿主验证线
  - 当前协议：`BCIC-IV-2a holdout`
- `ResNet1D + DLCR`
  - 通用 MTS 主验证线
  - 当前协议：fixed-split 多数据集框架对比

术语统一：

- `E0 / E1 / E2` = **框架**
- `NATOPS / FingerMovements / ...` = **数据集**

## 当前最短阅读路径

1. [工程记录/分类/00-入口/分类工程现状.md](../工程记录/分类/00-入口/分类工程现状.md)
2. [工程记录/分类/00-入口/分类调试记录.md](../工程记录/分类/00-入口/分类调试记录.md)
3. [ASSISTANT_ENTRY.md](../ASSISTANT_ENTRY.md)
4. [工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md](../工程记录/分类/01-阶段二-宿主实验/Tensor-CSPNet-端到端局部闭式残差层实现任务单.md)

## 当前已核实结果口径

### `Tensor-CSPNet`

- `BCIC-IV-2a holdout` 外部宿主复现已完成，平均准确率约 `0.7238`
- 当前单 seed 比较：
  - `E0 = 0.7103`
  - `E1 = 0.7083`
  - `E2 = 0.7018`
- 当前读法：
  - 宿主可靠
  - `E2` 还没有在该宿主上稳定优于 `E0 / E1`

### `ResNet1D`

- `E2 tau=0.2 corrected` 已完成 21 个数据集的全量 sweep
- 结合当前仓库里可直接核对的 `E0` 日志：
  - `E2` 相对 `E0`：**15 胜 / 1 平 / 5 负**
- 当前最值得记住的代表性提点：
  - `NATOPS`: `0.9056 -> 0.9389`
  - `FingerMovements`: `0.5200 -> 0.5800`
  - `UWaveGestureLibrary`: `0.7594 -> 0.8313`
  - `Heartbeat`: `0.3659 -> 0.5951`
- 当前温度结论：
  - 固定 `tau=0.2` 不能再写成统一最优默认值
  - 已核实的四数据集扫描显示：
    - `NATOPS`: `tau=0.5` 最优
    - `FingerMovements`: `tau=0.2` 最优
    - `SelfRegulationSCP1`: `tau=0.5` 最优
    - `UWaveGestureLibrary`: `tau=1.0` 最优

## 边界线与 standalone

- `MiniRocket + DLCR`
  - 当前定位：**边界 / 诊断线**
  - 已完成 4 个数据集的 frozen-feature 诊断
  - 但当前主仓库中的官方 MiniRocket 全量汇总工件并不完整，因此不把它写成闭环主结论

- `MBA_ManifoldBridge`
  - 当前定位：**独立 standalone 项目**
  - 保留其内部实验报告，但不并入当前主仓库分类主线 ranking

## 目录怎么读

优先进入：

- `models/`
- `scripts/`
- `docs/`
- `工程记录/`
- `out/`

按需进入：

- `route_b_unified/`
  - 当你需要追一阶段结构证据时再看
- `archive/`
  - 当你需要看上游参考实现时再看
- `standalone_projects/`
  - 当你专门切到独立项目时再看

## 当前不再建议的读法

不要再把当前仓库理解成：

- 把 `Tensor-CSPNet` 当成单一主线
- `ResNet1D` 只是预备代码
- 把 `MiniRocket` 的全量 SOTA 对照视为已在当前主仓库里完整沉淀
- `MBA` 已经并入主仓库分类主线

当前更准确的总读法是：

**双主线并列，`ResNet1D` 是当前通用 MTS 主结果入口，`Tensor-CSPNet` 是 EEG/SPD 外部宿主验证入口；边界线和 standalone 保留，但不混入主线 ranking。**
