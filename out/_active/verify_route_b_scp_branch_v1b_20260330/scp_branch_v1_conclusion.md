# SCP-Branch v1 Conclusion

更新时间：2026-03-30

same-backbone 对照口径：baseline 与 v1 复用完全相同的 dense backbone、normalization 与 dynamic_minirocket 训练协议；唯一差别是是否执行 train-only local shaping 写回。
当前不做 replay / curriculum / neighborhood propagation / test-time routing。

## selfregulationscp1

- `same_backbone_no_shaping`: 0.6348 +/- 0.0000
- `local_shaping`: 0.6418 +/- 0.0000
- `delta_test_macro_f1`: 0.0070 +/- 0.0000
- `delta_nearest_margin`: 0.0000 +/- 0.0000
- `delta_between_separation`: 0.0001 +/- 0.0000
- `delta_within_compactness`: -0.0000 +/- 0.0000
- `delta_temporal_stability`: 0.0000 +/- 0.0000
- `local_step_distortion_ratio_mean`: 1.5714 +/- 0.0000
- `admitted_margin_mean_before`: -0.1123 +/- 0.0000
- `admitted_same_dist_mean_before`: 0.6796 +/- 0.0000
- `margin_to_score_conversion`: 847.4229 +/- 0.0000
