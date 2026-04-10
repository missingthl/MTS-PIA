# PIA Operator P0b-lite Delayed Refresh Probe Promote

## 任务名称
`SCP-Branch P0b-lite: One-Step Delayed Geometry Refresh Probe`

## 阅读提示

这份文档不是完整 slow-layer closed-loop 设计，而是当前 `P0a.1` 之后的最小系统验证稿。

如果只想知道当前主线已经做到哪一步，请先看：

- [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/PIA-Operator-当前主线总览.md)

## 零、战略定位

当前 `P0a.1` 已经把快层主干推进到了：

- `A2r` 响应器
- `C3LR` 判别闭式解
- `r > 1` committee readout
- `optional B3` 连续几何耦合

这说明当前最关键的问题已经不是：

- 快层算子能不能跑
- 快层算子能不能学到东西

而是：

> 当前快层主干是否已经足够好到，值得让 slow geometry 跟进一次 delayed rebuild。

因此，这一步不是 full slow refresh，不是完整 fast-slow closed loop，而是：

> 只做一次最小、受控、单轮、可归因的 `P0b-lite delayed refresh probe`。

一句话说：

> `P0b-lite` 只回答“当前快层是否已经值得进入一次 delayed geometry rebuild”，不回答“完整慢层系统是否已经成立”。

---

## 一、唯一核心问题

**在固定 backbone、固定当前快层主干、固定 train-only/freeze 口径的前提下，若让当前快层 operator 先对 train manifold 作用一次，再基于作用后的 train states 重建一次 local geometry，这次 delayed geometry rebuild 是否会比“只做一次 fast operator”带来更好的结构与终端收益。**

这一步只问：

- 当前快层是否已经足够好到能为 slow geometry 创造更好的后继状态
- delayed refresh 的收益是否真实来自 geometry rebuild
- 还是仅仅来自“在 post-fast manifold 上又 refit 一次”

当前不问：

- 完整 multi-round slow loop 是否成立
- refresh scheduling / rollback 是否成立
- class-conditioned routing 是否已经需要并入

---

## 二、与当前工程的关系

这一轮严格建立在当前主线之上。

第一版默认固定：

- `A2r`
- `C3LR`
- `r = 4`
- `global readout`

因此，`P0b-lite` 不是重新发明一个新 operator，而是测试：

> 当前主线快层作用一次之后，slow geometry 是否已经值得重建一次。

若 `P0b-lite` 成功，只能说明：

- 当前快层主干已具备 delayed effect 价值

不能说明：

- full slow-layer refresh 已成立
- refresh scheduling 已成立
- rollback policy 已成立

---

## 三、必须守住的边界

### 1. 不进入完整 slow-layer closed loop

这一轮不做：

- multi-round refresh
- acceptance / rollback loop
- online adaptation

只允许：

- 一次性的 delayed rebuild

### 2. 快层主干在第一版中必须冻结

第一版写死：

- `A2r + C3LR + r4 + global readout`

这样才能回答：

- 当前主线是否已经值得进入 delayed refresh

### 3. `B3` 不作为第一版默认主线

原因：

- `NATOPS` 上当前最稳仍是 `global readout`
- `B3` 更适合作为后续并列 policy shadow

### 4. rebuild 只允许 train-only，再冻结到 val/test

继续保持：

- rebuild 只发生在 train
- rebuild 完后 geometry 冻结
- 同一口径作用于 `train/val/test`

---

## 四、当前我们在整体框架里的位置

按完整愿景来放，当前的位置是：

1. `P0` 已完成
   - 快层不是完全无效
2. `P0a.1` 已完成
   - 快层主干已经从“数值修补”推进到“判别闭式解 + higher-r + optional geometry coupling”
3. 现在进入 `P0b-lite`
   - 不是完整闭环
   - 只是第一次测试：
     - 这台赛车跑过一圈之后
     - 地图是否已经值得重画一次

---

## 五、实验主场与推进顺序

### 第一阶段：主场只做 `NATOPS seed=1`

固定：

- dataset: `natops`
- terminal: `dense_dynamic_gru`
- seed: `1`
- fast default: `A2r + C3LR + r4 + global readout`

### 第二阶段：若主场成立，再扩到 `NATOPS multi-seed`

默认：

- seeds: `1,2,3`

### 第三阶段：若 `NATOPS` 站住，再带到 `SCP1`

固定：

- dataset: `selfregulationscp1`
- terminal: `dynamic_minirocket`

第一版仍先沿用同一主线：

- `A2r + C3LR + r4 + global readout`

若需要，再把：

- `B3`

作为 shadow policy 并列比较。

---

## 六、四臂对照的最小实验设计

### `B0`: same_backbone_no_shaping

作用：

- 给出不进入 fast operator、不进入 delayed rebuild 的固定 backbone 参照

### `F1`: current_fast_mainline

默认定义：

- `A2r + C3LR + r4 + global readout`

作用：

- 给出当前快层主干本身的单轮收益

### `R0`: post-fast refit without delayed rebuild

定义：

1. 先在原始 frozen geometry 上拟合并应用 `F1`
2. 得到 post-fast train states `Z_train^1`
3. 不允许重新做 slow-side object discovery
4. 只允许在**原有 object identity 冻结**的前提下，对 post-fast states 做一次 operator refit

作用：

- 控制“只是因为 post-fast 状态变了，又 refit 一次 operator”这件事本身带来的收益

### `P0b-lite`: one-step delayed refresh

定义：

1. 先在原始 frozen geometry 上拟合并应用 `F1`
2. 得到 post-fast train states `Z_train^1`
3. 基于 `Z_train^1` 重新运行一次 train-only local geometry build
4. 得到 delayed refreshed geometry `G^1`
5. 再基于 `G^1` 拟合 second-pass operator
6. freeze 后统一作用于 `train/val/test`

作用：

- 测 delayed rebuild 是否真的带来了新的 geometry value

---

## 七、为什么需要 `R0` 这个控制组

如果没有 `R0`，那么当 `P0b-lite` 成功时，很难知道收益到底来自：

- operator 在 post-fast manifold 上又 refit 了一次

还是来自：

- slow geometry 真的重建得更好了

因此：

> `R0` 的作用是把“二次 refit 收益”从“真实 delayed refresh 收益”里剥出来。

---

## 八、实验执行流程

### Step 0：冻结当前默认快层主干

第一版写死：

- `A2r + C3LR + r4 + global readout`

### Step 1：运行 `B0`

### Step 2：运行 `F1`

### Step 3：运行 `R0`

目的：

- 测 post-fast manifold 上的“二次 refit 收益”
- 不允许重建 slow geometry

### Step 4：运行 `P0b-lite`

目的：

- 测 delayed rebuild 是否带来额外价值

### Step 5：只在 `P0b-lite > max(F1, R0)` 时，才扩到 formal

否则：

- 本轮停在 `seed=1`
- 不进入 multi-seed
- 不讨论 full slow loop

---

## 九、必须新增的诊断指标

### 继续保留

- `test_macro_f1`
- `response_vs_margin_correlation`
- `activation_coverage_ratio`
- `margin_gain_per_unit_distortion`
- `template_mean_direction_cosine`

### `P0b-lite` 必须新增

- `post_fast_geometry_within_compactness`
- `post_fast_geometry_between_separation`
- `post_fast_geometry_margin`
- `post_fast_geometry_temporal_stability`

- `delayed_refresh_within_compactness`
- `delayed_refresh_between_separation`
- `delayed_refresh_margin`
- `delayed_refresh_temporal_stability`

- `refit_only_delta_vs_f1`
- `delayed_refresh_delta_vs_f1`
- `delayed_refresh_delta_vs_r0`

- `geometry_rebuild_window_count`
- `geometry_rebuild_prototype_count`
- `second_pass_template_mean_direction_cosine`

---

## 十、必须输出的文件

- `pia_operator_p0b_lite_config_table.csv`
- `pia_operator_p0b_lite_per_seed.csv`
- `pia_operator_p0b_lite_geometry_diagnostics.csv`
- `pia_operator_p0b_lite_operator_diagnostics.csv`
- `pia_operator_p0b_lite_score_diagnostics.csv`
- `pia_operator_p0b_lite_response_diagnostics.csv`
- `pia_operator_p0b_lite_conclusion.md`

---

## 十一、必须回答的问题

1. 当前快层主干是否已经足够好到：
   - 能让 post-fast train manifold 变成一个值得 rebuild 的对象

2. `R0` 是否优于 `F1`

3. `P0b-lite` 是否优于 `R0`

4. delayed rebuild 后，geometry 结构是否改善：
   - within compactness
   - between separation
   - nearest margin
   - temporal stability

5. 若 `P0b-lite` 成立，是否值得扩到：
   - multi-seed formal
   - 再之后才是更完整的 slow-layer refresh policy

---

## 十二、成功标准

### 弱成立

- `R0 >= F1`
- 且 `P0b-lite >= F1`

### 中等成立

- `P0b-lite > R0`
- 且 delayed rebuild 后 geometry 至少一项核心结构指标改善
- 且 `test_macro_f1` 不明显退化

### 强成立

同时满足：

- `P0b-lite` 稳定优于 `F1` 与 `R0`
- `response_vs_margin_correlation` 不明显变坏
- `activation_coverage_ratio` 保持健康
- geometry rebuild 指标出现一致改善
- 值得扩到 multi-seed formal

---

## 十三、若结果不成立，如何解释

### 情形 A
`R0` 与 `P0b-lite` 都不优于 `F1`

说明：

- 当前快层虽然能直接提分
- 但还不足以为 slow-side rebuild 提供有价值的后继状态

### 情形 B
`R0 > F1`，但 `P0b-lite <= R0`

说明：

- post-fast manifold 上的二次 refit有价值
- 但 delayed geometry rebuild 本身尚未带来额外价值

### 情形 C
`P0b-lite > R0`，但 geometry 指标没有同步改善

说明：

- delayed rebuild 可能有效
- 但当前 slow-side 诊断口径仍不足

### 情形 D
geometry 看起来变好，但终端分数不动

说明：

- 当前 delayed effect 可能主要停留在结构层
- 还没被 terminal 稳定读出

---

## 十四、一句话执行目标

**在固定当前快层默认主线的前提下，通过 `B0 / F1 / R0 / P0b-lite` 四臂对照，验证当前快层主干是否已经足够好到能为 slow geometry 创造一次真正有价值的 delayed rebuild，并把“二次 refit 收益”和“真实 delayed refresh 收益”清楚分开。**
