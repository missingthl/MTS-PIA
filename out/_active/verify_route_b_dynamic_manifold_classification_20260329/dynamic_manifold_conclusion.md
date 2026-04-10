# Dynamic Manifold Classification Conclusion

更新时间：2026-03-29

本轮主比较对象：`static_linear / dynamic_meanpool / dynamic_gru`。
参考外部强基线：`raw + MiniROCKET`。

## natops

- `static_linear`: 0.5478 +/- 0.0000
- `dynamic_meanpool`: 0.6127 +/- 0.0000
- `dynamic_gru`: 0.7572 +/- 0.0000
- `raw + MiniROCKET` (reference): 0.7171 +/- 0.0000

- 当前最佳动态/静态模型：`dynamic_gru`。
- `dynamic > static`：`yes`
- `GRU > mean-pool`：`yes`
- `NATOPS no obvious degradation vs static`：`yes`
- `best dynamic vs raw+MiniROCKET`：`+0.0401`

## selfregulationscp1

- `static_linear`: 0.4947 +/- 0.0000
- `dynamic_meanpool`: 0.4553 +/- 0.0000
- `dynamic_gru`: 0.5658 +/- 0.0000
- `raw + MiniROCKET` (reference): 0.6792 +/- 0.0000

- 当前最佳动态/静态模型：`dynamic_gru`。
- `dynamic > static`：`yes`
- `GRU > mean-pool`：`yes`
- `SCP1 trajectory benefit`：`yes`
- `best dynamic vs raw+MiniROCKET`：`-0.1134`
