# Current Engineering Map

更新时间：2026-04-10

这份文件只回答五个问题：

1. 当前分类工程的现役主线是什么
2. 当前已经完成到哪一步
3. 哪些是分流探针，哪些是并行支线
4. 当前最该看的代码和结果是什么
5. 哪些内容已经降级为历史背景

## 1. 当前分类工程的现役主线

当前默认主线已经不再是早期：

- `phase15 / freeze`
- `Route B curriculum`
- `unified shell`

当前现役主线应固定理解为：

`raw -> dense z_seq -> frozen local geometry -> closed-form operator -> A2r response -> optional B3 coupling -> terminal`

最短总览：

- [工程记录/PIA-Operator-当前主线总览.md](../工程记录/PIA-Operator-当前主线总览.md)

核心代码入口：

- [route_b_unified/pia_operator_value_probe.py](../route_b_unified/pia_operator_value_probe.py)
- [scripts/run_route_b_pia_operator_p0a1_c3.py](../scripts/run_route_b_pia_operator_p0a1_c3.py)
- [scripts/run_route_b_pia_operator_p0a1_b3.py](../scripts/run_route_b_pia_operator_p0a1_b3.py)

当前最稳的实现读法是：

- slow layer 仍冻结
- fast layer 已扩成 operator family
- 当前最成熟的组合是：
  - `A2r`
  - `C3LR`
  - `r > 1`
  - `optional B3`

## 2. 当前已经完成到哪一步

### 2.1 当前主线完成位置

当前最准确的位置是：

**仍在 `P0a.1` 尾部。**

已经完成：

1. `A2r`
   - 响应器修到数值健康
2. `C_next`
   - 证明 same-only 修 `Lambda` 不足
3. `C3LR`
   - 证明判别目标重写是关键变量
4. `B3`
   - 证明连续几何耦合是有效部件，但不是统一默认赢家
5. `r > 1`
   - 证明当前快层不应继续被单轴容量限制

### 2.2 当前 formal 结果锚点

结果入口：

- [natops formal conclusion](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal/pia_operator_p0a1_b3_conclusion.md)
- [scp1 formal conclusion](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2/pia_operator_p0a1_b3_conclusion.md)

当前最稳口径：

- `NATOPS`
  - `same_backbone_no_shaping = 0.8131 +/- 0.0222`
  - `c3lr_r4_global_a2r = 0.8131 +/- 0.0139`
- `SelfRegulationSCP1`
  - `same_backbone_no_shaping = 0.6537 +/- 0.0144`
  - `c3lr_r1_global_a2r = 0.6560 +/- 0.0104`
  - `b3_r4_continuous_geom = 0.6557 +/- 0.0118`

当前主问题不是“算子完全学不出来”，而是：

- 学到的轴如何稳定转成对终端有利的局部推进场

结构证据见：

- [geometry_visual_summary.md](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/geometry_visual_summary.md)

## 3. 当前分流探针

### 3.1 `R0 / P0b-lite`

定位：

- `R0` 是 post-fast frozen-identity refit probe
- `P0b-lite` 是 one-step delayed refresh probe

主文档：

- [工程记录/PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md](../工程记录/PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md)
- [工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md](../工程记录/PIA-Operator-R0-多数据集稳定性阶段小结.md)

当前结论：

- `R0` 在 `natops / epilepsy` 上有正信号
- 在 `fingermovements / uwavegesturelibrary` 上明显失效
- `P0b-lite` 也没有成为统一稳定赢家

所以这条线当前是：

- 机会分支
- 结构拆解分支

而不是统一默认主线。

### 3.2 `P1a`

定位：

- 从 `global fixed operator`
- 转向 `query-conditioned local operator`

主文档：

- [工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](../工程记录/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)

结果入口：

- [p1a stage1 conclusion](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke/pia_operator_p1a_stage1_conclusion.md)

当前最关键结果：

- `FingerMovements`
  - `baseline_0 = 0.5088`
  - `f1_global_mainline = 0.4982`
  - `p1a_s1_offline_local_wls = 0.5288`

因此当前若问“最新活跃突破点在哪”，答案是：

- 主线仍停在 `P0a.1` 尾部
- 最新前沿在 `P1a`

## 4. 并行分类支线

### 4.1 动态分类线

这条线已经单独收束，不再和 `PIA-Operator` 混成同一主叙事。

总览：

- [工程记录/动态主线v1框架收束.md](../工程记录/动态主线v1框架收束.md)

当前读法：

- `dense z_seq` 已经站进主视野
- 动态线当前应拆成：
  - `动态主线 v1`
  - `研究外挂层`

### 4.2 no-bridge dual-stream

入口：

- [scripts/run_route_b_dual_stream_no_bridge.py](../scripts/run_route_b_dual_stream_no_bridge.py)
- [verify_route_b_dual_stream_no_bridge_20260329](../out/_active/verify_route_b_dual_stream_no_bridge_20260329)

当前状态：

- 已工程闭环
- 仍是并行验证分支

### 4.3 SCP branch

当前文档族：

- [工程记录/SCP-Branch-v0-prototype-memory-promote.md](../工程记录/SCP-Branch-v0-prototype-memory-promote.md)
- [工程记录/SCP-Branch-v1-local-separation-shaping-promote.md](../工程记录/SCP-Branch-v1-local-separation-shaping-promote.md)
- [工程记录/SCP-Branch-v2-geometry-refresh-promote.md](../工程记录/SCP-Branch-v2-geometry-refresh-promote.md)
- [工程记录/SCP-Branch-v3-closed-form-local-update-promote.md](../工程记录/SCP-Branch-v3-closed-form-local-update-promote.md)

当前状态：

- 是 prototype-memory / local geometry 的独立探索线
- 不应与当前默认 `PIA-Operator` 主线混写

## 5. 当前最该看的代码和结果

### 代码

- [route_b_unified/pia_operator_value_probe.py](../route_b_unified/pia_operator_value_probe.py)
- [route_b_unified/trajectory_representation.py](../route_b_unified/trajectory_representation.py)
- [route_b_unified/trajectory_minirocket_evaluator.py](../route_b_unified/trajectory_minirocket_evaluator.py)
- [scripts/run_route_b_pia_operator_p0a1_c3.py](../scripts/run_route_b_pia_operator_p0a1_c3.py)
- [scripts/run_route_b_pia_operator_p0a1_b3.py](../scripts/run_route_b_pia_operator_p0a1_b3.py)
- [scripts/run_route_b_pia_operator_p1a_stage1_local_wls.py](../scripts/run_route_b_pia_operator_p1a_stage1_local_wls.py)

### 结果

- [verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal](../out/_active/verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal)
- [verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2](../out/_active/verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2)
- [verify_route_b_pia_operator_p0a1_c3_20260401_smoke](../out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke)
- [verify_route_b_r0_multidataset_minirocket_20260403_smoke](../out/_active/verify_route_b_r0_multidataset_minirocket_20260403_smoke)
- [verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke](../out/_active/verify_route_b_pia_operator_p1a_stage1_local_wls_20260403_smoke)

## 6. 当前哪些只是历史背景

以下文档仍有参考价值，但不应再充当当前默认地图：

- [docs/ROUTE_B_MAIN_BODY.md](./ROUTE_B_MAIN_BODY.md)
- [docs/ROUTE_B_UNIFIED_POLICY.md](./ROUTE_B_UNIFIED_POLICY.md)
- [docs/ROUTE_B_UNIFIED_INTERFACES.md](./ROUTE_B_UNIFIED_INTERFACES.md)
- [docs/ROUTE_B_UNIFIED_MAPPING.md](./ROUTE_B_UNIFIED_MAPPING.md)
- [docs/phase15_mainline_freeze.md](./phase15_mainline_freeze.md)

它们主要用于回答：

- 当前 operator 主线是从哪些旧链路里收回来的
- 为什么旧 Route B / unified shell 不再是默认主体

## 7. 一句话状态

**当前分类工程最准确的读法是：`PIA-Operator` 已在 `P0a.1` 尾部站住为现役主线，`R0/P0b-lite` 是机会分支，`P1a` 是最新前沿，动态分类线已独立成平行系统路径。**
