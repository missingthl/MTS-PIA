# Active Output Layer

更新时间：2026-04-18

这里不再承担“完整历史说明书”的角色，只做一件事：

- 告诉我们 **当前应该先看哪些结果**
- 区分 **主结果 / 辅助诊断 / 历史支线 / scratch**

如果只是想快速理解当前分类主线，优先按下面顺序阅读。

## 1. 当前主结果

### `ResNet1D + DLCR` 行为有效变量正式矩阵

- [verify_resnet1d_dlcr_behavioral_matrix_20260418](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418)

这是当前最权威的一轮：
- `21` 个 fixed-split MTS 数据集
- `126` 条条件
- 主比较：
  - `E0`
  - `E2 + center_only`
  - `E2 + center_subproto + subproto_temperature ∈ {1.0, 0.5, 0.2, 0.1}`

先看：
- [ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md)
- [behavioral_results_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_results_table.csv)
- [behavioral_mechanism_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_mechanism_table.csv)

### `center_tangent(k=4)` 21 数据集全量补充

- [verify_resnet1d_center_tangent_fullscale_20260419](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_tangent_fullscale_20260419)

这轮回答的是：
- `center_tangent(k=4)` 是否已经足够强，可以升格成统一默认主线

先看：

- [ResNet1D_center_tangent_fullscale_stage_report_20260419.md](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_tangent_fullscale_20260419/ResNet1D_center_tangent_fullscale_stage_report_20260419.md)
- [center_tangent_comparison_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_tangent_fullscale_20260419/center_tangent_comparison_table.csv)

### `ResNet1D + DLCR` 21 数据集 `tau=0.2` 全量验证

- [verify_e2_tau02_fullscale_20260414](/home/THL/project/MTS-PIA/out/_active/verify_e2_tau02_fullscale_20260414)

这是当前主线收敛前的一轮“大规模正式验证”，适合回答：
- `E2 + center_subproto + tau=0.2` 作为单一配置时，在 21 个数据集上整体表现如何

### `Tensor-CSPNet + DLCR` 外部宿主验证

- [verify_tensor_cspnet_local_closed_form_holdout_20260412](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412)

这条线保留为 EEG/SPD 外部宿主验证，不是当前通用 MTS 主线，但不是废线。

## 2. 当前辅助诊断

这些目录不是当前“最终战报”，但对机制理解仍然重要。

### 子原型温度扫描

- [verify_resnet1d_subproto_temp_sweep_20260414](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_subproto_temp_sweep_20260414)

作用：
- 证明 `subproto_temperature` 是行为有效变量
- 同时证明它是**数据集相关变量**，不是统一默认值

### `tangent probe`

- [verify_resnet1d_tangent_probe_20260418](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_tangent_probe_20260418)

作用：
- 判断当前 `center_subproto` 是否支持低秩切空间读法
- 给 `center_tangent` 的候选 `rank` 提供谱证据

先看：

- [ResNet1D_tangent_probe_stage_report_20260418.md](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_tangent_probe_20260418/ResNet1D_tangent_probe_stage_report_20260418.md)
- [tangent_probe_dataset_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_tangent_probe_20260418/tangent_probe_dataset_table.csv)

### 数据流探针

- [verify_resnet1d_dataflow_probe_20260414](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dataflow_probe_20260414)
- [verify_resnet1d_local_closed_form_fixedsplit_20260414](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_local_closed_form_fixedsplit_20260414)

作用：
- 看 `center_only / center_subproto` 的 routing 与 local/final 关系
- 看子原型是否真的被 query 使用

## 3. 历史结果处理方式

当前 GitHub 口径只保留：

- 当前主结果入口
- 当前辅助诊断入口
- 可直接阅读的阶段报告与汇总表

大量中间运行产物、深层 `per_seed / per_subject / run_meta` 结果默认不再作为仓库主展示层保存。

## 4. 顶层日志与日志归档

当前顶层只保留了少量仍在直接阅读中的关键日志：

- [run_3baseline_batch_sys_20260414_v4.log](/home/THL/project/MTS-PIA/out/_active/run_3baseline_batch_sys_20260414_v4.log)
- [run_e2_tau02_fullscale_20260414.log](/home/THL/project/MTS-PIA/out/_active/run_e2_tau02_fullscale_20260414.log)
- [run_e2_tau02_fullscale_20260414_v2Corrected.log](/home/THL/project/MTS-PIA/out/_active/run_e2_tau02_fullscale_20260414_v2Corrected.log)

其余旧日志已经归档到：

- [out/_active/_logs_archive](/home/THL/project/MTS-PIA/out/_active/_logs_archive)

零散历史表格归档到：

- [out/_active/_tables_archive](/home/THL/project/MTS-PIA/out/_active/_tables_archive)

## 5. Scratch

- [out/_active/_scratch](/home/THL/project/MTS-PIA/out/_active/_scratch)

这里放的是：
- debug
- dryrun
- smoke
- 临时矩阵产物

这些内容默认不进入正式结论。

## 6. 最短阅读顺序

如果现在只想快速接管当前结果，按这个顺序看：

1. [ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md)
2. [behavioral_results_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_results_table.csv)
3. [behavioral_mechanism_table.csv](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_mechanism_table.csv)
4. [ResNet1D_center_tangent_fullscale_stage_report_20260419.md](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_center_tangent_fullscale_20260419/ResNet1D_center_tangent_fullscale_stage_report_20260419.md)
5. [ResNet1D_tangent_probe_stage_report_20260418.md](/home/THL/project/MTS-PIA/out/_active/verify_resnet1d_tangent_probe_20260418/ResNet1D_tangent_probe_stage_report_20260418.md)
