# ResNet1D + DLCR `center_prob_tangent` 全量阶段报告

更新时间：2026-04-19

## 1. 报告范围

本报告覆盖 `ResNet1D + DLCR` 新线 `center_prob_tangent` 在 **21 个 fixed-split MTS 数据集**上的首轮全量正式判定。

这轮不再回答“新线是否已经全局登顶”，而是回答更关键的分级问题：

1. `center_prob_tangent_v3` 是否已经从四数据集局部成功，推进到 **全局可接受**  
2. 它相对 `center_only / best center_subproto` 的强弱，是否已经足够支撑它继续作为当前主推进线  
3. 新线内部的三层数学部件里，当前真正站住的是哪一层，哪一层还没有收圆  

## 2. 实验口径

固定项：

- 宿主：`ResNet1D`
- 框架：`E2`
- `solve_mode = pinv`
- `support_mode = same_only`
- `prototype_aggregation = pooled`

当前正式比较条件：

- `center_only`
- `best center_subproto`
- `center_prob_tangent_v3`

其中新线 `center_prob_tangent_v3` 的内部结构为：

- `Ledoit-Wolf` 收缩
- `MDL` 自动定秩
- `PPCA` 后验软投影
- `pinv`

注意：

- `E0` 本轮不重跑，只作为上一轮正式矩阵的参考列进入对照
- `best center_subproto` 按既有 21 数据集最优 `tau` 在新代码口径下重跑
- 本轮全量之前，`Phase 0.5` 已通过 gate：
  - `posterior_direction_ok = true`
  - `rescue_ok = true`
  - `phase1_gate_passed = true`

## 3. 完成情况

- `Phase 0`：完成
- `Phase 0.5`：完成
- `Phase 1`：`63 / 63` 全部完成

直接可读表：

- [center_prob_tangent_phase05_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_prob_tangent_20260419/center_prob_tangent_phase05_table.csv)
- [center_prob_tangent_fullscale_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_prob_tangent_20260419/center_prob_tangent_fullscale_table.csv)
- [center_prob_tangent_phase05_summary.json](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_prob_tangent_20260419/center_prob_tangent_phase05_summary.json)
- [center_prob_tangent_fullscale_summary.json](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_prob_tangent_20260419/center_prob_tangent_fullscale_summary.json)

## 4. 全量结果总览

平均 `test_acc`：

- `center_only = 0.7483`
- `center_prob_tangent_v3 = 0.7459`
- `best center_subproto = 0.7450`

单数据集冠军数：

- `center_only = 9`
- `best center_subproto = 8`
- `center_prob_tangent_v3 = 4`

相对 `E0`：

- `center_prob_tangent_v3 vs E0`：`10 胜 / 1 平 / 10 负`

相对 `center_only`：

- `center_prob_tangent_v3 vs center_only`：`10 胜 / 3 平 / 8 负`

相对 `best center_subproto`：

- `center_prob_tangent_v3 vs best center_subproto`：`7 胜 / 4 平 / 10 负`

这组统计说明：

- 新线不是全局最优
- 但它也不是局部偶然成功后在全盘崩掉的失败分支
- 更准确的定位是：**已进入全局可接受区，但还没有收敛成统一默认主线**

## 5. 代表性结果

| 数据集 | `center_only` | `best center_subproto` | `center_prob_tangent_v3` | 读法 |
| :--- | ---: | ---: | ---: | :--- |
| `NATOPS` | `0.9778` | `0.9556` | `0.9389` | 新线当前没保住平滑强集的最高攻击性 |
| `Epilepsy` | `0.9638` | `0.9855` | `0.9783` | 新线仍有竞争力，但没有追平最强散点支撑 |
| `SelfRegulationSCP1` | `0.7679` | `0.8362` | `0.7952` | 相对 `center_tangent V1` 已有恢复，但仍未超过最强旧线 |
| `FingerMovements` | `0.5800` | `0.5700` | `0.4600` | 当前仍是新线最明显短板之一 |
| `JapaneseVowels` | `0.9270` | `0.9811` | `0.9757` | 新线能逼近强散点支撑，但还差最后一步 |
| `Libras` | `0.9389` | `0.8556` | `0.9556` | 新线在局部结构强的数据集上可以明显胜出 |
| `UWaveGestureLibrary` | `0.7938` | `0.7969` | `0.8031` | 新线有小幅稳定增益 |
| `MotorImagery` | `0.5300` | `0.5300` | `0.5700` | 新线在部分难集上确实带来额外结构收益 |

## 6. 全局结构读法

### 6.1 `selected_rank` 没有塌成无意义常数

核心统计：

- 平均 `mean_selected_rank ≈ 2.2603`
- 最低数据集均值：`0.0`（`SelfRegulationSCP1`）
- 最高数据集均值：`4.0`（如 `FingerMovements / Heartbeat / SelfRegulationSCP2`）

读法：

- `MDL` 没有把所有数据集都压成同一个固定秩
- 这说明新线里的自动定秩层**确实在工作**
- 当前问题不在“rank 完全没起作用”，而在“它起作用后是否过于激进”

### 6.2 `k0_fallback_rate` 很高，而且还没有形成干净的噪声选择性

核心统计：

- 全局平均 `k0_fallback_rate ≈ 0.4349`

高触发数据集：

- `SelfRegulationSCP1 = 1.0000`
- `Pendigits ≈ 0.8854`
- `HAR ≈ 0.8185`
- `Cricket ≈ 0.6667`
- `MotorImagery ≈ 0.6600`

低触发数据集：

- `FingerMovements = 0.0000`
- `Heartbeat = 0.0000`
- `SelfRegulationSCP2 = 0.0000`

读法：

- `k=0` 不是偶然现象，而是当前 `MDL` 层的强行为
- 它并没有只在“最脏、最不可信”的局部结构上触发
- 所以当前还不能把“`k=0` 回退到类中心”写成已经完成校准的优雅机制

更准确地说：

**当前 `MDL` 已经成功引入了选择性，但这份选择性还没有被校准成干净的结构判断。**

### 6.3 `PPCA posterior` 当前基本处于塌缩态

核心统计：

- 全局平均 `posterior_confidence_mean ≈ 2.24e-12`
- 全局平均 `posterior_confidence_far_decay_mean ≈ 1.0`
- 当前没有任何数据集的 `posterior_confidence_mean > 1e-6`

读法：

- `PPCA` 这一层在代码和数值上已经接入
- 但它目前**没有形成有用的软边界几何**
- 换句话说，`v3` 的改进当前主要不是来自“漂亮的概率软投影”，而更像是：
  - `LW` 稳住了谱
  - `MDL` 在硬选秩
  - 但 `PPCA posterior` 还没有真正长成行为有效变量

因此当前最重要的结构判断是：

**新线现在真正站住的是 `LW + MDL`，而不是 `PPCA posterior`。**

## 7. 当前结论

### 7.1 可以确认的结论

1. `center_prob_tangent_v3` 已经完成从局部成功到全量正式判定的推进  
   它不是短程 smoke 成功后在 21 数据集上整体崩掉的分支。

2. 新线已经达到“全局可接受”  
   它的平均 `test_acc` 高于 `best center_subproto`，相对 `center_only / best center_subproto` 也都拿到了非少数派胜场。

3. 新线当前还不是统一默认主线  
   因为它的全局统计还没有超过 `center_only`，单数据集冠军数也只有 `4`。

4. 当前新线最可信的收获是：  
   - `Ledoit-Wolf` 收缩是有效的  
   - `MDL` 自动定秩是有效但尚未校准完成的  
   - `PPCA posterior` 尚未形成真正行为有效的软边界层  

### 7.2 当前不能写的结论

1. 不能写“`center_prob_tangent_v3` 已经全局登顶”  
2. 不能写“`PPCA` 软投影已经成为这轮收益的主要来源”  
3. 不能写“`k=0` 回退已经被校准成可靠的噪声识别机制”  

## 8. 当前定位与下一步

当前更稳的定位是：

- `center_only`：长期强基线
- `best center_subproto`：当前最稳的旧主增强线
- `center_prob_tangent_v3`：**已经具备继续作为当前理论主推进线的资格，但还不是成熟完全体**

下一步不应回头退回 `center_tangent V1`，也不应急着引入 `same_opp`。  
当前最应该继续收的是两件事：

1. 修 `PPCA posterior`，让 `posterior_confidence` 真正长成行为变量  
2. 校准 `MDL / k0_fallback`，让 `k=0` 更像“有选择地保命”，而不是“高频误杀结构”  

一句话收口：

**这轮全量结果说明，新线已经证明自己值得继续推进，但它现在更像“已站住的半成品完全体”，而不是已经收圆的统一默认框架。**
