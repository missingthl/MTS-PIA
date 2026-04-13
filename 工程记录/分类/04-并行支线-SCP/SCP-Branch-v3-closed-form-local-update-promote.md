# SCP-Branch v3 Closed-form Local Update Promote

## 共享前提
- `dense z_seq`
- `dynamic_minirocket`
- `prototype-memory`
- `v1b` 的 tight anchors / local shaping 口径
- 单轮、离线、train-only

## 唯一核心问题
prototype-memory 上的局部几何，能不能用低成本闭式更新变得更稳、更可分。

## 最小实现
- 不更新 terminal
- 不改 backbone
- 只更新：
  - prototype
  - local representative states
  - local discriminative direction
- 第一版采用：
  - prototype moving average
  - medoid refresh
  - local margin direction recompute

## 主比较
- `v0 prototype-memory`
- `v1b shaped memory`
- `v3 closed-form updated memory`

## 成功标准
- 结构指标改善
- 更新代价显著低于重新训练
- 更新后对象更稳或更可分
