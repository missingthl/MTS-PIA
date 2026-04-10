# PIA Operator Value Probe Conclusion

更新时间：2026-04-01

主实验只比较 Arm A（mean/prototype-centered update）与 Arm B（single-template PIA update）。
参考几何固定；不做 slow-layer refresh、不做 replay、不做 test-time adaptation。
operator 统一在 train-only 上拟合，拟合后冻结，并以同一参数作用于 train/val/test。

## natops

- `same_backbone_no_shaping`: 0.8287 +/- 0.0000
- `mean_centered`: 0.7837 +/- 0.0000
- `single_template_pia`: 0.8177 +/- 0.0000
- `delta_single_template_pia_vs_mean_centered`: 0.0341 +/- 0.0000
- `delta_mean_centered_margin`: 0.0387 +/- 0.0000
- `delta_single_template_pia_margin`: 0.0002 +/- 0.0000
- `mean_centered_margin_gain_per_unit_distortion`: 0.8519 +/- 0.0000
- `single_template_pia_margin_gain_per_unit_distortion`: 0.0031 +/- 0.0000
- `template_mean_direction_cosine`: 0.0580 +/- 0.0000
- `single_template_pia_response_vs_margin_correlation`: -0.2277 +/- 0.0000
- `single_template_pia_activation_coverage_ratio`: 0.9958 +/- 0.0000
