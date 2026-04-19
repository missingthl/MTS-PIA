# ResNet1D + DLCR `tangent probe` 阶段报告

更新时间：2026-04-18

## 1. 报告范围

本报告覆盖 `ResNet1D + DLCR` 在 **21 个 fixed-split MTS 数据集**上的 `tangent probe` 诊断结果。

本轮目标不是比较精度，而是回答：

1. 当前 `center_subproto` 的子原型偏移是否呈现明显低秩切空间  
2. `center_tangent` 的候选秩是否可以直接压到 `k < 4`  

## 2. Probe 口径

- 宿主：`ResNet1D`
- 框架：`E2`
- `prototype_geometry_mode = center_subproto`
- `subproto_temperature = 1.0`
- `tangent_rank = 3`
- `tangent_source = subproto_offsets`

可读表：

- [tangent_probe_dataset_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_tangent_probe_20260418/tangent_probe_dataset_table.csv)
- [tangent_probe_class_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_tangent_probe_20260418/tangent_probe_class_table.csv)

## 3. 核心结果

- 数据集数：`21`
- `recommended_candidate_ranks` 的并集：`[4]`
- 平均 `rank95_mean = 4.0`
- 平均 `effective_rank_mean ≈ 3.9433`
- 平均 `top1_energy_ratio_mean ≈ 0.2987`
- 平均 `top1_top2_spectral_gap_mean ≈ 0.0343`

## 4. 代表性读法

相对更“集中”的数据集：

- `HAR`
  - `top1_top2_spectral_gap ≈ 0.0904`
  - `effective_rank ≈ 3.7694`
- `Pendigits`
  - `gap ≈ 0.0722`
  - `effective_rank ≈ 3.8825`

相对更“不支持低秩化”的数据集：

- `BasicMotions`
  - `gap ≈ 0.0062`
  - `effective_rank ≈ 3.9968`
- `RacketSports`
  - `gap ≈ 0.0119`
  - `effective_rank ≈ 3.9850`
- `AtrialFibrillation`
  - `gap ≈ 0.0130`
  - `effective_rank ≈ 3.9913`

## 5. 当前结论

1. 当前 `center_subproto` 的偏移谱**没有显示出明显低秩切空间**  
   按 `95%` 能量标准看，21 个数据集都指向完整 `4` 维支持。

2. 当前不支持把 `center_tangent` 的第一版默认写成 `k in {1,2,3}` 的低秩切空间版本  

3. 如果继续推进 `center_tangent`，最稳的首轮候选应该保留：
   - `k = 4`

一句话说：

**这轮 probe 打通了技术路径，但它支持的是“保守的 `k=4` 切空间支撑”，而不是“已经足够低秩化”的强降维结论。**
