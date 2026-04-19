# ResNet1D + DLCR 本阶段实验报告

更新时间：2026-04-18

## 1. 报告范围

本报告覆盖当前 `ResNet1D + DLCR` 主线在 **21 个 fixed-split MTS 数据集**上的这一轮正式机制矩阵实验。

本轮实验目标不是继续加新机制，而是收紧三个问题：

1. `center_subproto` 相比 `center_only` 是否真的带来额外结构收益  
2. `subproto_temperature` 是否真在打路由分辨率  
3. 子原型是否被 query 真正使用，而不只是几何上被拉开  

## 2. 实验口径

### 2.1 固定主线

- 宿主：`ResNet1D`
- 框架：`E2`
- `solve_mode = pinv`
- `support_mode = same_only`
- `prototype_aggregation = pooled`

### 2.2 本轮主变量

- `prototype_geometry_mode ∈ {center_only, center_subproto}`
- `subproto_temperature ∈ {1.0, 0.5, 0.2, 0.1}`

冻结项：

- `routing_temperature = 1.0`
- `class_prior_temperature = 1.0`

注意：

- `class_prior_temperature` 本轮只是冻结诊断变量，不进入主矩阵
- `center_only` 本轮定义为 **positive-support class-center baseline**
- `committee_mean / flat / same_opp / 新 gate / 新 fusion` 不进入本轮

### 2.3 硬复用约束

旧结果只有在以下字段**完全一致**时才允许复用：

- `dataset`
- `arm`
- `split_protocol`
- `runner_protocol`
- `seed`
- `host_backbone`
- `epochs`
- `train_batch_size`
- `test_batch_size`
- `closed_form_solve_mode`
- `prototype_geometry_mode`
- `local_support_mode`
- `prototype_aggregation`
- `local_readout_gate`
- `dataflow_probe`
- `routing_temperature`
- `class_prior_temperature`
- `subproto_temperature`

本轮实际结果：

- 没有发生大规模旧结果复用
- 当前结论可按新口径直接读取

## 3. 数据集与矩阵

### 3.1 数据集

本轮共 21 个数据集：

- `har`
- `natops`
- `fingermovements`
- `selfregulationscp1`
- `basicmotions`
- `handmovementdirection`
- `uwavegesturelibrary`
- `epilepsy`
- `atrialfibrillation`
- `pendigits`
- `racketsports`
- `articularywordrecognition`
- `heartbeat`
- `selfregulationscp2`
- `libras`
- `japanesevowels`
- `cricket`
- `handwriting`
- `ering`
- `motorimagery`
- `ethanolconcentration`

### 3.2 条件矩阵

每个数据集共 6 条条件：

1. `E0`
2. `E2 + center_only`
3. `E2 + center_subproto + tau=1.0`
4. `E2 + center_subproto + tau=0.5`
5. `E2 + center_subproto + tau=0.2`
6. `E2 + center_subproto + tau=0.1`

总计：

- `21 × 6 = 126` 条条件
- `126 / 126` 已全部完成

## 4. 结果总览

结果表：

- [behavioral_results_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_results_table.csv)

机制表：

- [behavioral_mechanism_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_mechanism_table.csv)

### 4.1 胜负统计

相对 `E0`：

- `center_only vs E0`：`8 胜 / 2 平 / 11 负`
- `best center_subproto vs E0`：`14 胜 / 1 平 / 6 负`

相对 `center_only`：

- `best center_subproto vs center_only`：`14 胜 / 4 平 / 3 负`

### 4.2 最优温度分布

`best_subproto_tau` 在 21 个数据集上的分布：

- `tau = 1.0`：`9` 个数据集
- `tau = 0.5`：`3` 个数据集
- `tau = 0.2`：`5` 个数据集
- `tau = 0.1`：`4` 个数据集

结论：

**不存在统一默认温度。**

### 4.3 代表性结果

| 数据集 | `E0` | `center_only` | `best center_subproto` | 最优 `tau` | 读法 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `HAR` | `0.9430` | `0.9410` | `0.9518` | `1.0` | 子原型层带来小幅稳定收益 |
| `NATOPS` | `0.8389` | `0.9611` | `0.9611` | `0.5` | 类中心层已极强，子原型层只在合适温度下追平 |
| `FingerMovements` | `0.6100` | `0.5900` | `0.5800` | `1.0` | 子原型层当前没有带来额外收益 |
| `SelfRegulationSCP1` | `0.7884` | `0.8601` | `0.8703` | `0.1` | 低温对子原型路由有明显帮助 |
| `UWaveGestureLibrary` | `0.8156` | `0.8281` | `0.8406` | `0.1` | 子原型层带来额外提升 |
| `Heartbeat` | `0.6732` | `0.6683` | `0.6439` | `1.0` | 当前子原型层整体有害 |
| `Epilepsy` | `0.9348` | `0.8623` | `0.9928` | `0.5` | 子原型层带来本轮最大结构收益之一 |
| `Libras` | `0.9000` | `0.8556` | `0.9444` | `0.5` | 类中心不够，子原型层明显有效 |

## 5. 三个机制问题的回答

### 5.1 `center_subproto` 是否真的带来额外结构收益？

结论：**是，但不是统一成立。**

证据：

- 相对 `center_only`，`best center_subproto` 在 `21` 个数据集里：
  - `14` 胜
  - `4` 平
  - `3` 负

说明：

- `center_only` 不是弱基线，它本身已经非常强
- 但 `center_subproto` 在多数数据集上确实能再往前走一步

本轮子原型层收益最明显的数据集包括：

- `Epilepsy`：`+0.1304`
- `Libras`：`+0.0889`
- `JapaneseVowels`：`+0.0541`
- `HandMovementDirection`：`+0.0405`
- `UWaveGestureLibrary`：`+0.0125`
- `SelfRegulationSCP1`：`+0.0102`

当前不支持“子原型层总是更好”的数据集包括：

- `Heartbeat`
- `EthanolConcentration`
- `FingerMovements`

### 5.2 `subproto_temperature` 是否真在打路由分辨率？

结论：**是行为有效变量，但明显数据集相关。**

这不是表面结论，因为当前矩阵只保留了真正行为有效的变量：

- `center_subproto + same_only` 下，真正进入预测行为的是 `subproto_temperature`
- `class_prior_temperature` 本轮只作冻结诊断变量，不参与主结论

从整体平均看，温度降低确实会让 routing 变尖：

| 条件 | 平均 `entropy` | 平均 `max weight` |
| :--- | ---: | ---: |
| `tau=1.0` | `0.9992` | `0.2633` |
| `tau=0.5` | `0.9969` | `0.2766` |
| `tau=0.2` | `0.9818` | `0.3171` |
| `tau=0.1` | `0.9350` | `0.3859` |

但“变尖”并不自动等于“更好”：

- `NATOPS`：最佳是 `tau=0.5`
- `SCP1`：最佳是 `tau=0.1`
- `UWave`：最佳是 `tau=0.1`
- `FingerMovements`：降温后并没有带来收益

因此：

**温度确实在打路由分辨率，但它不是统一默认值，而是数据集相关变量。**

### 5.3 子原型是否被 query 真正使用？

结论：**被使用了，但更多是“广泛轮用”，不是稳定尖锐专用。**

本轮新增指标：

- `subproto_top1_occupancy_entropy`
- `subproto_usage_effective_count`

当前读法：

- 平均 `effective_count` 大多在 `3.0+`
- 说明 4 个子原型并不是“只有 1 个在用”
- 但结合高 `entropy` 可以看出：
  - 它们更多是在被 query **广泛轮用**
  - 还没有普遍长成“每个 query 明确偏向某个固定子原型”的强分工状态

也就是说：

**子原型已经不是没用，但当前多数数据集上的使用方式仍偏软。**

## 6. 当前阶段结论

### 6.1 可以确认的结论

1. `center_only` 是一个必须保留的强基线  
   它不是陪跑基线，而是当前 `DLCR` 主线里非常强的正类类中心支撑版本。

2. `center_subproto` 已经具备额外结构收益  
   在 `21` 个数据集里，多数情况下它能超过 `center_only`，说明子原型层不是无效装饰。

3. `subproto_temperature` 是真实行为变量  
   但当前只能写成**数据集相关变量**，不能再写成统一默认值。

4. 当前主线应继续保留为：
   - `ResNet1D`
   - `E2`
   - `pinv`
   - `same_only`
   - `pooled`
   - `center_only / center_subproto` 二者并列作为当前核心几何分支

### 6.2 当前不能写的结论

1. 不能写“`tau=0.2` 是统一最优默认配置”  
2. 不能写“子原型已经在所有数据集上学成尖锐分工”  
3. 不能写“`center_only` 已经被完全取代”  

## 7. 下一步建议

基于本轮结果，下一步不建议大面积扩机制，而建议继续沿“行为有效变量优先”推进：

1. 保留 `center_only` 作为长期强基线  
2. 对 `center_subproto` 继续只围绕：
   - `subproto_temperature`
   - 数据流解释
   做收敛
3. 只有在重新引入负类支撑进入 solve 时，才把 `class_prior_temperature` 恢复成主变量  
4. 当前不建议把：
   - `committee_mean`
   - `same_opp`
   - 新 gate / 新 fusion  
   拉回主线，避免再次混变量

## 8. 直接可读入口

- [behavioral_matrix_manifest.json](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_matrix_manifest.json)
- [behavioral_results_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_results_table.csv)
- [behavioral_mechanism_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_mechanism_table.csv)
- [behavioral_summary.json](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_summary.json)
