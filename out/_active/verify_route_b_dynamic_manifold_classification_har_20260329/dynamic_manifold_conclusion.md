# Dynamic Manifold Classification Conclusion

更新时间：2026-03-29

本轮主比较对象：`static_linear / dynamic_meanpool / dynamic_gru`。
参考外部强基线：`raw + MiniROCKET`。

## har

- `static_linear`: 0.6811 +/- 0.0000
- `dynamic_meanpool`: 0.6461 +/- 0.0000
- `dynamic_gru`: 0.7332 +/- 0.0000
- `raw + MiniROCKET` (reference): 0.9595 +/- 0.0000

- 当前最佳动态/静态模型：`dynamic_gru`。
- `dynamic > static`：`yes`
- `GRU > mean-pool`：`yes`
- `best dynamic vs raw+MiniROCKET`：`-0.2262`
