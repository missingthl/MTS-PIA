# P0a.1 Stage-A Computational Repair Conclusion

更新时间：2026-04-01

当前只实现阶段 A：`B0 current_sigmoid_minimal`、`A1/A2`、`A1s/A2s`，以及新增的稳健全局冻结尺度分支 `A1r/A2r`。
slow-layer refresh / class-conditioned routing / delayed refresh 均未进入本轮。
A1/A2/A1s/A2s 使用 train-only 拟合后的 frozen operator，并沿用 train-derived budget scale 作用于 val/test。

## natops

- `same_backbone_no_shaping`: 0.8287 +/- 0.0000
- `mean_centered`: 0.7837 +/- 0.0000
- `current_sigmoid_minimal`: 0.8177 +/- 0.0000
- `sigmoid_clip_tanh`: 0.8177 +/- 0.0000
- `sigmoid_clip_tanh_local_median`: 0.7837 +/- 0.0000
- `sigmoid_clip_tanh_scaled`: 0.8177 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled`: 0.7837 +/- 0.0000
- `sigmoid_clip_tanh_scaled_iqr`: 0.8177 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_iqr`: 0.8231 +/- 0.0000
- `sigmoid_clip_tanh_response_vs_margin_correlation`: -0.2277 +/- 0.0000
- `sigmoid_clip_tanh_local_median_response_vs_margin_correlation`: 0.0458 +/- 0.0000
- `sigmoid_clip_tanh_scaled_response_vs_margin_correlation`: -0.1313 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_response_vs_margin_correlation`: 0.1714 +/- 0.0000
- `sigmoid_clip_tanh_scaled_iqr_response_vs_margin_correlation`: -0.1124 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_iqr_response_vs_margin_correlation`: 0.1770 +/- 0.0000
- `sigmoid_clip_tanh_activation_coverage_ratio`: 1.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_activation_coverage_ratio`: 0.9361 +/- 0.0000
- `sigmoid_clip_tanh_scaled_activation_coverage_ratio`: 0.4503 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_activation_coverage_ratio`: 0.0021 +/- 0.0000
- `sigmoid_clip_tanh_scaled_iqr_activation_coverage_ratio`: 0.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_iqr_activation_coverage_ratio`: 0.0002 +/- 0.0000
- `sigmoid_clip_tanh_gate_saturation_ratio`: 0.9999 +/- 0.0000
- `sigmoid_clip_tanh_local_median_gate_saturation_ratio`: 0.9163 +/- 0.0000
- `sigmoid_clip_tanh_scaled_gate_saturation_ratio`: 0.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_gate_saturation_ratio`: 0.0001 +/- 0.0000
- `sigmoid_clip_tanh_scaled_iqr_gate_saturation_ratio`: 0.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_iqr_gate_saturation_ratio`: 0.0000 +/- 0.0000
