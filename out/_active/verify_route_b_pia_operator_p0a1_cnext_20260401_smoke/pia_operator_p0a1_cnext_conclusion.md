# P0a.1 C_next Conclusion

更新时间：2026-04-01

当前实现固定 `A2r` 前端，只比较阶段 C 的三条线：`C0 unweighted`、`C1 mean_dist_weighted`、`C2 median_min_weighted`。
本轮仍保留 TELM2 的广义逆/闭式解主线；变化仅发生在样本度量矩阵 `Lambda`。

## natops

- `same_backbone_no_shaping`: 0.8287 +/- 0.0000
- `mean_centered`: 0.7837 +/- 0.0000
- `c0_unweighted_a2r`: 0.8231 +/- 0.0000
- `c1_mean_dist_weighted_a2r`: 0.7891 +/- 0.0000
- `c2_median_min_weighted_a2r`: 0.7891 +/- 0.0000
- `c0_template_mean_direction_cosine`: 0.0111 +/- 0.0000
- `c1_template_mean_direction_cosine`: 0.0232 +/- 0.0000
- `c2_template_mean_direction_cosine`: 0.0232 +/- 0.0000
- `c0_effective_sample_ratio`: 1.0000 +/- 0.0000
- `c1_effective_sample_ratio`: 0.9992 +/- 0.0000
- `c2_effective_sample_ratio`: 0.6780 +/- 0.0000
- `c0_min_proto_effective_sample_size`: 8.0000 +/- 0.0000
- `c1_min_proto_effective_sample_size`: 7.9625 +/- 0.0000
- `c2_min_proto_effective_sample_size`: 4.5072 +/- 0.0000
