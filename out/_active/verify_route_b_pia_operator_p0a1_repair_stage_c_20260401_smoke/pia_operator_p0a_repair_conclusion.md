# P0a.1 Stage-A Computational Repair Conclusion

更新时间：2026-04-01

当前实现覆盖阶段 A，以及基于 `A2r` 前端的阶段 C 首个变体：`soft_weighted_fit_a2r`。
slow-layer refresh / class-conditioned routing / delayed refresh / continuous geometric force field 均未进入本轮。
阶段 C 保留 TELM2 的闭式/广义逆式求解主线，只在样本度量上引入 prototype-local 的软权重。

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
- `soft_weighted_fit_a2r`: 0.7891 +/- 0.0000
- `sigmoid_clip_tanh_response_vs_margin_correlation`: -0.2277 +/- 0.0000
- `sigmoid_clip_tanh_local_median_response_vs_margin_correlation`: 0.0458 +/- 0.0000
- `sigmoid_clip_tanh_scaled_response_vs_margin_correlation`: -0.1313 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_response_vs_margin_correlation`: 0.1714 +/- 0.0000
- `sigmoid_clip_tanh_scaled_iqr_response_vs_margin_correlation`: -0.1124 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_iqr_response_vs_margin_correlation`: 0.1770 +/- 0.0000
- `soft_weighted_fit_a2r_response_vs_margin_correlation`: 0.1688 +/- 0.0000
- `sigmoid_clip_tanh_activation_coverage_ratio`: 1.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_activation_coverage_ratio`: 0.9361 +/- 0.0000
- `sigmoid_clip_tanh_scaled_activation_coverage_ratio`: 0.4503 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_activation_coverage_ratio`: 0.0021 +/- 0.0000
- `sigmoid_clip_tanh_scaled_iqr_activation_coverage_ratio`: 0.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_iqr_activation_coverage_ratio`: 0.0002 +/- 0.0000
- `soft_weighted_fit_a2r_activation_coverage_ratio`: 0.0003 +/- 0.0000
- `sigmoid_clip_tanh_gate_saturation_ratio`: 0.9999 +/- 0.0000
- `sigmoid_clip_tanh_local_median_gate_saturation_ratio`: 0.9163 +/- 0.0000
- `sigmoid_clip_tanh_scaled_gate_saturation_ratio`: 0.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_gate_saturation_ratio`: 0.0001 +/- 0.0000
- `sigmoid_clip_tanh_scaled_iqr_gate_saturation_ratio`: 0.0000 +/- 0.0000
- `sigmoid_clip_tanh_local_median_scaled_iqr_gate_saturation_ratio`: 0.0000 +/- 0.0000
- `soft_weighted_fit_a2r_template_mean_direction_cosine`: 0.0232 +/- 0.0000
- `soft_weighted_fit_a2r_effective_sample_size`: 191.8466 +/- 0.0000
