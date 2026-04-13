# P0a.1 C3 Conclusion

更新时间：2026-04-01

当前实现固定 `A2r` 前端，只比较三条线：`C0 unweighted same-only`、`C2 weighted same-only`、`C3 bipolar discriminative weighted`。
本轮仍保留 TELM2 的广义逆/闭式解主线；`C3` 的变化发生在拟合目标 `Y`、bipolar 样本池和权重矩阵 `Lambda`，不是对纯 Stage-C same-only 修复的替代。

## natops

- `same_backbone_no_shaping`: 0.5147 +/- 0.0000
- `mean_centered`: 0.5097 +/- 0.0000
- `c0_unweighted_a2r`: 0.5116 +/- 0.0000
- `c2_median_min_weighted_a2r`: 0.5030 +/- 0.0000
- `c3_bipolar_discriminative_weighted_a2r`: 0.5180 +/- 0.0000
- `c0_template_mean_direction_cosine`: 0.0111 +/- 0.0000
- `c2_template_mean_direction_cosine`: 0.0232 +/- 0.0000
- `c3_template_mean_direction_cosine`: 0.1592 +/- 0.0000
- `c0_effective_sample_ratio`: 1.0000 +/- 0.0000
- `c2_effective_sample_ratio`: 0.6780 +/- 0.0000
- `c3_effective_sample_ratio`: 0.6865 +/- 0.0000
- `c2_min_proto_effective_sample_size`: 4.5072 +/- 0.0000
- `c3_same_proto_effective_sample_size`: 5.5080 +/- 0.0000
- `c3_opp_proto_effective_sample_size`: 5.5965 +/- 0.0000
- `c3_same_opp_count_ratio`: 1.0000 +/- 0.0000
- `c3_same_opp_weight_mass_ratio`: 1.0000 +/- 0.0000
- `same/opp pool audit`: 未见明显系统性失衡，主归因可以更集中在判别目标重写。
