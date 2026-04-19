# Active Output Layer

更新时间：2026-04-19

这里现在按“四层”来读：

1. 当前主结果层
2. 当前辅助诊断层
3. 历史结果层
4. 本地临时层

如果你是 agent，默认只需要接住第 1 层和第 2 层。

## 1. 当前主结果层

### 1.1 `ResNet1D + DLCR` 行为变量正式矩阵

- [verify_resnet1d_dlcr_behavioral_matrix_20260418](verify_resnet1d_dlcr_behavioral_matrix_20260418)

当前权威入口：

- [verify_resnet1d_dlcr_behavioral_matrix_20260418/ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md](verify_resnet1d_dlcr_behavioral_matrix_20260418/ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md)
- [verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_results_table.csv](verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_results_table.csv)
- [verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_mechanism_table.csv](verify_resnet1d_dlcr_behavioral_matrix_20260418/behavioral_mechanism_table.csv)

这是当前 `ResNet1D + DLCR` 的默认主结果入口。

### 1.2 `tangent probe`

- [verify_resnet1d_tangent_probe_20260418](verify_resnet1d_tangent_probe_20260418)

当前权威入口：

- [verify_resnet1d_tangent_probe_20260418/ResNet1D_tangent_probe_stage_report_20260418.md](verify_resnet1d_tangent_probe_20260418/ResNet1D_tangent_probe_stage_report_20260418.md)
- [verify_resnet1d_tangent_probe_20260418/tangent_probe_dataset_table.csv](verify_resnet1d_tangent_probe_20260418/tangent_probe_dataset_table.csv)

这是当前的谱证据层，已经是 `21 / 21` 全数据集结果。

### 1.3 `center_tangent(k=4)` 全量正式对照

- [verify_resnet1d_center_tangent_fullscale_20260419](verify_resnet1d_center_tangent_fullscale_20260419)

当前权威入口：

- [verify_resnet1d_center_tangent_fullscale_20260419/ResNet1D_center_tangent_fullscale_stage_report_20260419.md](verify_resnet1d_center_tangent_fullscale_20260419/ResNet1D_center_tangent_fullscale_stage_report_20260419.md)
- [verify_resnet1d_center_tangent_fullscale_20260419/center_tangent_comparison_table.csv](verify_resnet1d_center_tangent_fullscale_20260419/center_tangent_comparison_table.csv)

这是当前的正式性能层，已经是 `21 / 21` 全数据集结果。

### 1.4 `Tensor-CSPNet + DLCR`

- [verify_tensor_cspnet_local_closed_form_holdout_20260412](verify_tensor_cspnet_local_closed_form_holdout_20260412)

当前角色：

- 外部宿主验证线
- 不是当前默认主结果入口

## 2. 当前辅助诊断层

这些结果重要，但不是默认第一阅读层：

- [verify_resnet1d_subproto_temp_sweep_20260414](verify_resnet1d_subproto_temp_sweep_20260414)
- [verify_resnet1d_dataflow_probe_20260414](verify_resnet1d_dataflow_probe_20260414)
- [verify_e2_tau02_fullscale_20260414](verify_e2_tau02_fullscale_20260414)
- [verify_3baseline_minirocket_20260414](verify_3baseline_minirocket_20260414)

## 3. 历史结果层

下面这些目录仍保留在仓库里，但不再作为默认结果入口：

- `verify_route_b_*`
- `verify_axis_pullback_*`
- `verify_pia_core_*`
- `route_b_dynamic_*`
- `verify_logeuclidean_*`

如果不是专门回溯历史结构证据，默认跳过它们。

## 4. 本地临时层

默认跳过：

- `_tmp*`
- `_scratch`

另外要注意：

- 当前一些深层原始运行树，例如 `e0/`、`e2/` 下的逐 run 目录，主要是本地 raw artifact
- GitHub 和 agent 默认应该先读上层的 `md / csv / json` 汇总文件，而不是直接扎进深层逐 run 目录

## 5. 最短阅读顺序

1. [verify_resnet1d_dlcr_behavioral_matrix_20260418/ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md](verify_resnet1d_dlcr_behavioral_matrix_20260418/ResNet1D_DLCR_behavioral_matrix_stage_report_20260418.md)
2. [verify_resnet1d_tangent_probe_20260418/ResNet1D_tangent_probe_stage_report_20260418.md](verify_resnet1d_tangent_probe_20260418/ResNet1D_tangent_probe_stage_report_20260418.md)
3. [verify_resnet1d_center_tangent_fullscale_20260419/ResNet1D_center_tangent_fullscale_stage_report_20260419.md](verify_resnet1d_center_tangent_fullscale_20260419/ResNet1D_center_tangent_fullscale_stage_report_20260419.md)
4. [verify_tensor_cspnet_local_closed_form_holdout_20260412](verify_tensor_cspnet_local_closed_form_holdout_20260412)
