# SCP-Branch v2 Single Replay Round Promote

## 共享前提
- `dense z_seq`
- `dynamic_minirocket`
- `prototype-memory`
- `v1b` 的 tight anchors / local shaping 口径
- 单轮、离线、train-only

## 唯一核心问题
single replay 能不能把 `v1b` 的局部 shaping 信号，转成更稳定的终端收益。

## 最小实现
- 只回放一次
- 只在 `z_seq` 层
- 只对已 shaped 的窗口位置回写
- 未被 shaping 的位置保持原值
- 不做 replay curriculum / online / test-time update

## 主比较
- `same_backbone_no_shaping`
- `v1b_local_shaping`
- `v2_single_replay`

## 成功标准
- `v2 > v1b`
- `v2 > no_shaping`
- stitching / continuity 代价没有明显爆炸
