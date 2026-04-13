# 密集轨迹终端探针 T8-pre Promote

更新时间：2026-03-30

## 一、唯一核心问题

当显著减小滑动步长，使隐空间轨迹 `z_seq` 从稀疏散点回归为密集流线后，直接交给 `MiniROCKET` 捕捉其高维演化特征，是否能显著优于当前 `dynamic_gru`，并缩小与 `raw_minirocket` 的差距。

## 二、当前全部冻结

- 冻结 `trajectory_representation` 的其余部分
- 冻结 `window_len`
- 冻结 `z_dim`
- 禁止任何 `augmentation / rebasis / feedback`
- 不做双流
- 不做 bridge
- 不做 raw 回写

## 三、唯一改动

- 仅把动态表示的 `hop_len` 强制设为 `1`

## 四、主比较对象

1. `static_linear`
2. `dense_dynamic_gru`
3. `dense_dynamic_minirocket`
4. `raw_minirocket`

## 五、硬约束

- `dynamic_minirocket` 只吃 `z_seq`
- `static` 不做伪序列 `MiniROCKET`
- padding 固定为 `target_len = max(train_max_dense_z_seq_len, 9)` 的 edge pad
- normalization 在 dense `z_seq` 生成后重新计算
- 所有 padding / normalization 只发生在 `z_seq` 表示层

## 六、数据集

- `NATOPS`
- `SelfRegulationSCP1`

## 七、必须输出

1. `dense_trajectory_probe_summary.csv`
2. `dense_trajectory_stride_impact_diagnostics.csv`
3. `dense_trajectory_conclusion.md`

## 八、一句话执行目标

冻结当前动态主线其余一切，只把 `hop_len` 压到 `1`，把更密的 `z_seq` 直接交给 `MiniROCKET`，判断当前动态线的主要损失究竟来自采样稀疏与终端不匹配，还是来自表示本身的信息损耗。
