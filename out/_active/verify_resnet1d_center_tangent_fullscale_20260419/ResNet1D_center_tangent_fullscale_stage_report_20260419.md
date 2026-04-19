# ResNet1D + DLCR `center_tangent(k=4)` 全量补充报告

更新时间：2026-04-19

## 1. 报告范围

本报告覆盖 `ResNet1D + DLCR` 在 **21 个 fixed-split MTS 数据集**上的 `center_tangent(k=4)` 全量补充验证。

这轮不是重新定义主矩阵，而是回答一个更窄的问题：

1. `center_tangent(k=4)` 在全盘上是否已经足够强，可以升格为和 `center_only / center_subproto` 同级的默认主分支  
2. 它相对已有两条主分支的优势是普遍优势，还是选择性优势  

## 2. 实验口径

固定项：

- 宿主：`ResNet1D`
- 框架：`E2`
- `solve_mode = pinv`
- `support_mode = same_only`
- `prototype_aggregation = pooled`

本轮变量：

- `prototype_geometry_mode = center_tangent`
- `tangent_rank = 4`
- `tangent_source = subproto_offsets`

对照来自上一轮已完成的 21 数据集正式矩阵：

- `E0`
- `E2 + center_only`
- `E2 + best center_subproto`

## 3. 完成情况

- 数据集数：`21`
- `center_tangent(k=4)` 条件：`21 / 21` 全部完成
- 对照表：
  - [center_tangent_comparison_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_tangent_fullscale_20260419/center_tangent_comparison_table.csv)

## 4. 胜负统计

相对 `E0`：

- `11 胜 / 1 平 / 9 负`

相对 `center_only`：

- `8 胜 / 5 平 / 8 负`

相对 `best center_subproto`：

- `5 胜 / 3 平 / 13 负`

## 5. 代表性结果

| 数据集 | `E0` | `center_only` | `best center_subproto` | `center_tangent(k=4)` | 读法 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `NATOPS` | `0.8389` | `0.9611` | `0.9611` | `0.9889` | `center_tangent` 明显最强 |
| `Libras` | `0.9000` | `0.8556` | `0.9444` | `0.9611` | `center_tangent` 有额外收益 |
| `RacketSports` | `0.8289` | `0.8421` | `0.8487` | `0.8684` | `center_tangent` 小幅胜出 |
| `JapaneseVowels` | `0.9730` | `0.9784` | `0.9892` | `0.9946` | `center_tangent` 进一步提点 |
| `SelfRegulationSCP1` | `0.7884` | `0.8601` | `0.8703` | `0.6758` | `center_tangent` 明显失利 |
| `Heartbeat` | `0.6732` | `0.6683` | `0.6439` | `0.4732` | `center_tangent` 当前有害 |
| `Handwriting` | `0.5412` | `0.5518` | `0.5776` | `0.5094` | `center_tangent` 不稳 |

## 6. 当前结论

### 6.1 可以确认的结论

1. `center_tangent(k=4)` 不是无效分支  
   它在部分数据集上已经能明显超过 `center_only` 和 `best center_subproto`。

2. `center_tangent(k=4)` 目前更像 **选择性高上限分支**  
   它的优势主要出现在 `NATOPS / Libras / RacketSports / JapaneseVowels / ArticularyWordRecognition` 这类数据集上。

3. `center_tangent(k=4)` 还不能替代 `best center_subproto` 作为统一默认结构  
   因为它相对 `best center_subproto` 的统计是 `5 胜 / 3 平 / 13 负`。

### 6.2 当前不能写的结论

1. 不能写“`center_tangent(k=4)` 已经是新的全局最优主线”  
2. 不能写“`center_tangent` 可以直接替换 `center_subproto`”  
3. 不能写“它在所有数据集上都比 `center_only` 更稳”  

## 7. 当前定位

当前更稳的定位是：

- `center_only / center_subproto`：已完成 21 数据集正式验证的核心几何分支
- `center_tangent(k=4)`：已完成全量验证、但呈现明显数据集依赖的候选几何分支

也就是说：

**`center_tangent` 已经证明自己值得继续保留，但现阶段更适合定位为“强候选分支”，而不是新的统一默认框架。**
