# No-Bridge Dual-Stream Conclusion

更新时间：2026-03-29

本轮比较对象：`spatial_only / manifold_only / dual_stream`。

## natops

- `dual_stream`: val_macro_f1=0.5593 +/- 0.0000, test_macro_f1=0.5686 +/- 0.0000
- `manifold_only`: val_macro_f1=0.6371 +/- 0.0000, test_macro_f1=0.6586 +/- 0.0000
- `spatial_only`: val_macro_f1=0.4541 +/- 0.0000, test_macro_f1=0.5441 +/- 0.0000

- 当前 test_macro_f1 最强的是 `manifold_only`。

## selfregulationscp1

- `dual_stream`: val_macro_f1=0.8504 +/- 0.0000, test_macro_f1=0.7873 +/- 0.0000
- `manifold_only`: val_macro_f1=0.6866 +/- 0.0000, test_macro_f1=0.5060 +/- 0.0000
- `spatial_only`: val_macro_f1=0.8504 +/- 0.0000, test_macro_f1=0.8495 +/- 0.0000

- 当前 test_macro_f1 最强的是 `spatial_only`。
