# Dense Trajectory Probe Conclusion

更新时间：2026-03-30

本轮主比较对象：`static_linear / dense_dynamic_gru / dense_dynamic_minirocket`。
参考绝对强基线：`raw + MiniROCKET`。
硬约束：冻结表示与增强主链，只将 `hop_len` 强制压到 `1`，padding 固定为 `max(train_max_len, 9)` 的 edge pad。

## natops

- `static_linear`: 0.4797 +/- 0.0000
- `dense_dynamic_gru`: 0.6759 +/- 0.0000
- `dense_dynamic_minirocket`: 0.6940 +/- 0.0000
- `raw_minirocket`: 0.7171 +/- 0.0000

- `sparse_len_mean -> dense_len_mean`: `5.0 -> 24.0`
- `dense_gru_minus_sparse_gru`: `-0.0097`
- `dense_minirocket_minus_sparse_minirocket`: `+0.0523`
- `dense_minirocket_minus_raw_minirocket`: `-0.0231`

## selfregulationscp1

- `static_linear`: 0.4648 +/- 0.0000
- `dense_dynamic_gru`: 0.4802 +/- 0.0000
- `dense_dynamic_minirocket`: 0.6348 +/- 0.0000
- `raw_minirocket`: 0.6792 +/- 0.0000

- `sparse_len_mean -> dense_len_mean`: `8.0 -> 718.0`
- `dense_gru_minus_sparse_gru`: `-0.0342`
- `dense_minirocket_minus_sparse_minirocket`: `+0.0592`
- `dense_minirocket_minus_raw_minirocket`: `-0.0444`
