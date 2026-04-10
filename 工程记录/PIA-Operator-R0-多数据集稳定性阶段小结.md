# PIA Operator R0 多数据集稳定性阶段小结

## 定位
这份记录只回答一个问题：

> `R0 = post-fast frozen-identity refit` 是否在多个 MiniROCKET 常用数据集上表现稳定。

当前口径：

- terminal: `dynamic_minirocket`
- seed: `1`
- 与四臂对照：
  - `baseline_0`
  - `F1 = fast mainline`
  - `R0 = post-fast refit no rebuild`
  - `P0b-lite = delayed refresh`

说明：

- 这里的 `R0` 不是“完全不动 geometry 的纯 refit”
- 它更准确地说是：
  - `frozen membership / frozen identity`
  - 在 `post-fast` 状态上重算局部坐标中心
  - 再做 second-pass refit

## 已完成数据集
- `natops`
- `selfregulationscp1`
- `fingermovements`
- `basicmotions`
- `handmovementdirection`
- `uwavegesturelibrary`
- `epilepsy`

补充：

- `HAR` 单独运行过，但在当前资源与时间预算下过重，本轮未纳入结论。
- `BasicMotions` 为满分饱和任务，只能作为“不区分”的参考，不能作为 `R0` 稳定性证据。

## 结果表

| dataset | baseline | F1 | R0 | P0b-lite | R0 - F1 | R0 - baseline |
|---|---:|---:|---:|---:|---:|---:|
| `basicmotions` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| `epilepsy` | 0.9858 | 0.9787 | 0.9929 | 0.9787 | +0.0142 | +0.0071 |
| `fingermovements` | 0.5088 | 0.4982 | 0.4699 | 0.4992 | -0.0282 | -0.0388 |
| `handmovementdirection` | 0.2727 | 0.2727 | 0.2727 | 0.2727 | +0.0000 | +0.0000 |
| `natops` | 0.6749 | 0.6813 | 0.6862 | 0.6581 | +0.0050 | +0.0113 |
| `selfregulationscp1` | 0.6348 | 0.6380 | 0.6324 | 0.6425 | -0.0056 | -0.0025 |
| `uwavegesturelibrary` | 0.7193 | 0.7196 | 0.6950 | 0.7100 | -0.0246 | -0.0243 |

原始汇总表见：

- [verify_route_b_r0_multidataset_summary_20260403.tsv](/home/THL/project/MTS-PIA/out/_active/verify_route_b_r0_multidataset_summary_20260403.tsv)

## 当前结论

### 1. `R0` 不是跨数据集稳定赢家
从已完成的 7 组里看：

- 明显正向：
  - `natops`
  - `epilepsy`
- 明显负向：
  - `fingermovements`
  - `uwavegesturelibrary`
- 近乎无差别：
  - `basicmotions`
  - `handmovementdirection`
- 轻微负向：
  - `selfregulationscp1`

因此当前不能把 `R0` 收成：

- “通用有效的 second-pass 策略”

更稳的说法是：

- `R0` 在部分数据集上有效
- 但它具有明显的数据集依赖性

### 2. `P0b-lite` 仍然不稳定
在这批数据里，`P0b-lite` 也没有表现出统一增益。

因此当前更不该把主线推进成：

- delayed refresh formal

### 3. `R0` 更像“结构性机会分支”，不是默认主线
从现有结果看，`R0` 值得保留，但身份应调整为：

- `R0 branch / post-fast frozen-identity refit probe`

而不是：

- 当前统一默认策略

## 口径风险

### 1. `response_vs_margin_correlation` 不能直接当跨 arm 统一代理
各 arm 使用的是各自 operator 下的局部 geometry 来算 margin，因此：

- `F1`
- `R0`
- `P0b-lite`

之间的 correlation 不完全同口径。

### 2. `template_mean_direction_cosine` 存在符号歧义
当前方向指标是 raw cosine，未做 sign correction。

因此：

- 某些 `R0` 解可能只是整体符号翻转
- 不能仅凭该指标绝对值直接判断“方向一定坏了”

### 3. `R0` 不是纯 refit
这一点必须反复强调：

- `R0` 会在 `post-fast` 状态上重算 prototype-local 坐标中心
- 因此它已经包含了“冻结身份下的坐标重对齐”

## 下一步建议

当前最合理的推进方向不是：

- 扩 `R0` formal
- 扩 `P0b-lite`

而是：

1. 单独把 `R0` 从 `P0b-lite` 分支中拆成独立 probe
2. 继续分解 `R0` 收益到底来自：
   - frozen-identity recentering
   - 还是 second-pass operator refit
3. 若后续要扩数据集，优先继续选：
   - MiniROCKET 常用集
   - 且避免已经饱和或极重的任务作为第一优先级

## 一句话收口
**当前 `R0` 已被证明是一个真实但不稳定的正信号：它在 `natops/epilepsy` 上有效，在 `fingermovements/uwavegesturelibrary` 上明显失效，因此还不能上升为统一主线，只适合作为独立机会分支继续拆解。**
