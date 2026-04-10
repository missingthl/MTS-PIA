# Dynamic Terminal Probe Conclusion

更新时间：2026-03-30

本轮主比较对象：`static_linear / dynamic_gru / dynamic_minirocket`。
参考外部强基线：`raw + MiniROCKET`。
硬约束：`dynamic_minirocket` 只吃 `z_seq`，`static` 不做伪序列 MiniROCKET，padding 固定为 `max(train-max-len, 9)` 的 edge pad。

## natops

- `static_linear`: 0.4797 +/- 0.0000
- `dynamic_gru`: 0.6857 +/- 0.0000
- `dynamic_minirocket`: 0.6269 +/- 0.0000
- `raw + MiniROCKET` (reference): 0.7171 +/- 0.0000

- 当前最佳内部终端：`dynamic_gru`
- `dynamic_minirocket > dynamic_gru`：`not_yet`
- `dynamic_minirocket > static_linear`：`yes`
- `dynamic_minirocket vs raw + MiniROCKET`：`-0.0902`
- `dynamic_gru vs raw + MiniROCKET`：`-0.0314`

## selfregulationscp1

- `static_linear`: 0.4648 +/- 0.0000
- `dynamic_gru`: 0.5144 +/- 0.0000
- `dynamic_minirocket`: 0.5662 +/- 0.0000
- `raw + MiniROCKET` (reference): 0.6792 +/- 0.0000

- 当前最佳内部终端：`dynamic_minirocket`
- `dynamic_minirocket > dynamic_gru`：`yes`
- `dynamic_minirocket > static_linear`：`yes`
- `dynamic_minirocket vs raw + MiniROCKET`：`-0.1130`
- `dynamic_gru vs raw + MiniROCKET`：`-0.1648`
