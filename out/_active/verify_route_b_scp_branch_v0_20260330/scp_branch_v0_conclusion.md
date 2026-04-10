# SCP-Branch v0 Conclusion

更新时间：2026-03-30

本轮只验证 prototype-memory 对象本身是否比随机窗口集合更有结构。
硬约束：冻结 dense backbone，不做 replay / curriculum / PIA-guided local geometry。

## selfregulationscp1

- `static_linear`: 0.5244 +/- 0.0000
- `dense_dynamic_gru`: `skipped`
- `dense_dynamic_minirocket`: 0.6348 +/- 0.0000
- `raw_minirocket`: 0.6792 +/- 0.0000

- `prototype within_compactness`: `0.8315`
- `random within_compactness`: `1.1116`
- `prototype between_separation`: `1.0786`
- `random between_separation`: `1.7320`
- `prototype nearest_margin`: `0.0064`
- `random nearest_margin`: `0.0022`
- `prototype temporal_stability`: `0.9887`
- `random temporal_stability`: `0.9909`

- `low_coverage_class_count`: `0` / `2`
- `safe_coverage_mean`: `1.0000`
