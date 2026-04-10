# SCP-Branch v1b Anchor Tightening Promote

## 核心问题
在保持 dense backbone、prototype-memory、local shaping 方向与强度不变的前提下，只收紧 shaping 作用窗口的选择规则，`SCP1` 是否会比 `v1` 更容易把局部 margin 改善转成终端收益。

## 当前冻结
- `raw -> dense z_seq -> dynamic_minirocket`
- `prototype-memory`
- `beta = 0.5`
- `epsilon_scale = 0.10`
- train-only shaping
- 不做 replay / curriculum / neighborhood propagation / test-time routing

## 唯一改动
- `v1`: 每个 prototype 选最近的 `M=16` 个真实成员窗口
- `v1b`: 每个 prototype 先保留 `same_dist <= classwise prototype-member median` 的成员，再按 `margin_before = opp_dist - same_dist` 升序取前 `M_tight=8`

## 解释口径
- `v1b` 只改“谁被整形”
- 不改“怎么整形”
- 因此若 `v1b > v1`，更可能说明 admitted anchors 需要更边界化，而不是 shaping 强度不足

## 重点输出
- `delta_test_macro_f1`
- `delta_nearest_margin`
- `delta_between_separation`
- `delta_within_compactness`
- `delta_temporal_stability`
- `local_step_distortion_ratio_mean`
- `admitted_margin_mean_before`
- `admitted_same_dist_mean_before`

## 成功标准
- `v1b > v1` 于 `delta_test_macro_f1`
- 或在不明显恶化 `step_distortion` 的前提下，`delta_nearest_margin` 与 `margin_to_score_conversion` 明显优于 `v1`
