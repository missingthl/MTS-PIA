# PIA Operator 当前主线总览

更新时间：2026-04-01

## 一句话状态

当前已经完成的不是完整的“快慢自适应流形导航引擎”，而是它的**冻结慢层下的快层 operator 主干版**。

更准确地说：

- slow layer 仍冻结
- reference geometry 仍冻结
- backbone 仍冻结
- 但快层已经从“单模板数值修补”推进到了：
  - `A2r` 响应器
  - `C3LR` 判别闭式解
  - `B3` 连续几何耦合
- `r > 1` 的 committee readout

## 当前比较口径

从这份总览开始，主结果的比较口径固定为两层：

### 1. 外部主对比

默认只认：

- `raw + MiniROCKET`

若该数据集已有官方 fixed-split `raw + MiniROCKET` 结果，则它是唯一默认外部 headline。

### 2. 内部归因对比

以下对象仍然保留，但只用于内部消融与归因：

- `same_backbone_no_shaping`
- `baseline_0`
- `F1`
- `R0`
- `P0b-lite`

因此以后文档里的正确读法是：

- “是否比 `raw + MiniROCKET` 更近 / 更远 / 更优”
- 再补充“相对内部 baseline 的增益来自哪里”

而不是反过来。

## 当前框架怎么实现

当前实现链条是：

`raw -> dense z_seq -> frozen local geometry -> closed-form operator -> A2r response -> optional B3 coupling -> terminal`

对应代码入口主要在：

- [pia_operator_value_probe.py](/home/THL/project/MTS-PIA/route_b_unified/pia_operator_value_probe.py)
- [run_route_b_pia_operator_p0a1_b3.py](/home/THL/project/MTS-PIA/scripts/route_b/run_route_b_pia_operator_p0a1_b3.py)

### 1. 冻结 backbone

当前主场：

- `NATOPS + dynamic_gru`

当前交叉验证：

- `SelfRegulationSCP1 + dynamic_minirocket`

### 2. 冻结 local geometry

当前仍基于：

- prototype-memory
- local representative states
- tight anchors
- train-only frozen geometry

当前还没有：

- slow refresh
- candidate generation
- rollback
- class-conditioned routing

### 3. 阶段 A 已落成 `A2r`

当前默认响应器不是早期 `sigmoid overflow` 那条线，而是：

- local median centering
- frozen IQR scale
- clip + tanh

它解决的是：

- 响应数值域错配
- 全域饱和
- gate 失控

### 4. same-only Stage C 已证明“只修 Lambda 不够”

`C_next` 已经给出阶段性结论：

- `Lambda` 可以从近似常数修成有分层的权重矩阵
- 但 `same-only + weighted closed-form` 仍不能把模板方向稳定推成局部判别轴

所以当前主线没有继续停在 `same-only`。

### 5. 当前真正推进的是 `C3LR`

当前主线不是 auto-associative target，而是：

- hetero-associative discriminative closed-form
- bipolar same/opp pooling
- weighted generalized inverse
- `r > 1` committee readout

这里最关键的变化不是换求解器，而是：

- 把拟合目标从“重构输入形状”
- 改成“沿一维轴拉开 same / opp”

### 6. 当前 `B3` 是可选几何读出层

`u_template` 不再总是被当成全局硬推方向。

当前可选地把它和局部几何轴缝合：

\[
g_{geom}(z)=\langle u_{template},u_{geom}(z)\rangle
\]

\[
Force(z)=local\_step(z)\cdot a_{resp}(z)\cdot g_{geom}(z)\cdot u_{geom}(z)
\]

这一步的意义是：

- `u_template` 只做大方向引导
- 真正推力回到局部几何方向
- 避免全局硬推撕裂轨迹

## 当前在完整构思中的哪一步

如果按完整愿景“快慢自适应流形导航引擎”来放位置，当前处在：

1. `P0` 已完成
   - 证明单模板 PIA 不是完全无效
   - 但还未独立立住
2. `P0a.1` 阶段 A 已完成
   - 响应器修到可工作
3. `P0a.1` 阶段 C / `C_next` 已完成一轮失败归因
   - 证明 same-only 的 `Lambda` 修复不够
4. `P0a.1-C3` 已完成
   - 证明判别目标重写是决定性变量之一
5. `P0a.1-B3` 已完成 smoke 与 formal
   - 证明连续几何耦合是有效部件
   - 但不是所有数据集上的统一默认赢家

所以当前最准确的位置是：

**仍然在 `P0a.1` 的尾部。**

还没有进入：

- `P0b delayed refresh`
- slow-layer refresh
- closed-loop trigger
- 完整快慢闭环

## 下一阶段接口

如果继续往前，当前最值得开的不是继续给全局静态算子补小超参，而是：

- [PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md)

它的定位不是替换现有 `C3/B3`，而是测试：

- 当前全局固定 `W`
- 是否该升级成 query-conditioned local `W_local(z_t)`

第一版只在 `FingerMovements` 这样的负向站上做最小 offline local-WLS probe。

## 当前最稳的实现读法

不要再把当前框架理解成“单模板 PIA 单一路线”。

当前更准确的读法是：

**冻结慢层下的快层 operator family**

其中当前已经真实跑通的核心组合是：

- `A2r`
- `C3LR`
- `optional B3`
- `r > 1`

## 当前 multi-seed 结果怎么读

结果文件：

- [natops formal conclusion](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_b3_natops_multiseed_20260401_formal/pia_operator_p0a1_b3_conclusion.md)
- [scp1 formal conclusion](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_b3_scp1_multiseed_20260401_formal_v2/pia_operator_p0a1_b3_conclusion.md)

### NATOPS

当前最稳主线是：

- `c3lr_r4_global_a2r = 0.8131 +/- 0.0139`

同站参考：

- `same_backbone_no_shaping = 0.8131 +/- 0.0222`

这说明：

- `r > 1 + discriminative closed-form` 已经能追平当前 backbone baseline
- `B3` 在 NATOPS 上不是稳定默认赢家

### SCP1

当前 cross-check 最好的两条线很接近：

- `c3lr_r1_global_a2r = 0.6560 +/- 0.0104`
- `b3_r4_continuous_geom = 0.6557 +/- 0.0118`

同站参考：

- `same_backbone_no_shaping = 0.6537 +/- 0.0144`

这说明：

- 框架在 SCP1 上是活的
- `B3` 在 SCP1 上比在 NATOPS 上更自然

## 相比 raw + MiniROCKET 的位置

外部参考见：

- [dynamic_manifold_conclusion.md](/home/THL/project/MTS-PIA/out/route_b_dynamic_manifold_classification_20260329_formal/dynamic_manifold_conclusion.md)

当前读法：

- 历史文档里曾有“`NATOPS` 高于 `raw + MiniROCKET`”的表述，但那对应的是旧内部 `raw_minirocket_baseline` 口径
- 当前若使用更严格的官方 fixed-split `raw + MiniROCKET`，则应以该口径为准
- 因此以后不再用内部 `raw_minirocket_baseline` 充当主 headline，除非该数据集暂时没有官方 fixed-split 结果

所以当前框架的战略状态是：

- 主场 `NATOPS` 已经站住
- `SCP1` 还没完全封顶

## 当前不该混淆的几点

- 当前不是 slow refresh 成功
- 当前不是 complete fast-slow closed loop
- 当前不是 `r=1` 已证明最终足够
- 当前也不是“单靠修 Lambda 成功”

当前真正成立的是：

- 响应器修复成立
- 判别闭式解成立
- `r > 1` 解放后框架明显更有生命力
- `B3` 已经是可工作的几何读出层，但不是统一默认策略

## 当前推荐阅读顺序

1. [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md)
2. [PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P0a.1-C3-Discriminative-Closed-Form-Probe-promote.md)
3. [PIA-Operator-P0a.1-Computational-Repair-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P0a.1-Computational-Repair-promote.md)
4. [PIA-Operator-P0a.1-Stage-C-next-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P0a.1-Stage-C-next-promote.md)
5. [PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md)

## 当前阶段性结论

当前工程已经从“局部修补很多小算子”压缩成了一条更清楚的主线：

**冻结几何与 backbone，先把快层 operator 学对、读对、接到终端上。**

而当前最接近默认主线的实现，不再是早期 `single-template r=1 same-only`，而是：

**`A2r + C3LR + r>1`，并视数据集决定是否接 `B3`。**

## 当前框架与下一步的兼容性

当前框架的一个重要优点是：它已经不是一串难以拆开的研究性 patch，而是一个可以继续向下接系统层动作的显式快层接口。

### 已经兼容的后续方向

#### 1. `P0b` 风格的 one-step delayed refresh

当前链条已经具备进入一次受控 delayed refresh 的最小条件：

- operator 是 train-only 拟合且可冻结的
- 作用后的 `z_seq` 是显式可读、可重放的
- local geometry 仍然是独立对象，可在 operator 作用后重新统计
- `r > 1` 与 `B3` 都不会破坏“先作用、后刷新”的基本时序

因此，从接口上看，当前框架已经**兼容**：

- 先用冻结快层 operator 改变 train states
- 再在改变后的 states 上重建一次 local geometry
- 再看这次 delayed refresh 是否真的带来额外收益

#### 2. 更正式的 multi-template / higher-r operator family

这件事已经不是未来概念，而是现有代码能力的一部分：

- `r > 1` 已经能通过 `mean_committee` 接入同一闭式解链
- 当前 multi-seed 结果也说明：`r > 1` 在 NATOPS 上是关键增益来源之一

所以更高模板数不是与当前框架冲突的扩展，而是当前框架已经打开的维度。

#### 3. dataset-conditioned readout policy

当前结果本身已经说明：

- `NATOPS` 更稳的是 `c3lr_r4_global_a2r`
- `SCP1` 上 `B3` 更自然

这意味着当前框架已经兼容“同一 operator family，不同数据集选择不同读出策略”的 policy 层，而不需要重写内核。

### 只部分兼容、但还不能直接混进去的方向

#### 1. full slow-layer refresh / candidate generation / rollback

当前还没有：

- refresh candidate proposal
- acceptance rule
- rollback rule
- refresh 后的闭环调度

所以从系统边界看，当前只兼容：

- 一次性的 delayed refresh probe

还不兼容：

- 完整 slow-layer closed loop

#### 2. class-conditioned routing

当前 geometry 已经是 prototype-local 的，但 operator 的运行时选择策略还没有被正式抽象成 routing layer。

所以它与当前内核并不冲突，但也还没有真正进入实现队列。

### 当前最不该混淆的地方

#### 1. 不要把 `B3` 当成统一默认赢家

`B3` 已经成立为可工作部件，但当前 multi-seed 显示：

- 在 `SCP1` 上它更自然
- 在 `NATOPS` 上它不如 `c3lr_r4_global_a2r` 稳

因此它当前更像：

- 可选读出策略

而不是：

- 统一默认算子

#### 2. 不要把旧的 margin 诊断直接当成全部真相

随着 `C3LR`、`B3`、`r > 1` 进入同一条链，旧的：

- `response_vs_margin_correlation`
- `margin_gain_per_unit_distortion`

仍然有价值，但已经不再能完全覆盖“新法向如何被终端读出”这个问题。

也就是说，下一步往系统层推进时，诊断口径本身也需要一起升级。

## 下一步如何推进

当前最稳的推进顺序不是继续做更多局部调参，而是把已经活起来的快层主干接入下一层系统动作。

### 第一步：把当前快层主干正式冻结成 `P0a.1` 的阶段性默认实现

推荐默认读法：

- `NATOPS` 主线默认：
  - `A2r + C3LR + r4 + global readout`
- `SCP1` 交叉验证保留：
  - `A2r + C3LR/B3 + r4`

这一步的意义不是再比谁小数点高一点，而是：

- 明确当前“拿去接下一层实验”的默认快层是谁

### 第二步：进入 `P0b-lite`，只做一次受控 delayed refresh

这是当前最兼容、也最应该推进的下一步。

推荐口径：

- 仍冻结 backbone
- 仍不做 full slow loop
- 只在 train 上做一次：
  - fast operator apply
  - delayed geometry rebuild
  - second-pass evaluation

这样能直接回答：

- 现在这个快层主干，是否已经足够好到能真正喂给 slow layer 一次有意义的 delayed effect

对应正式流程稿见：

- [PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P0b-lite-Delayed-Refresh-Probe-promote.md)

### 第三步：把 `B3` 从“研究分支”转成“可比较读出策略”

当前不建议再把 `B3` 当成必须统一进主线的单一路径。

更稳的做法是：

- 在 `P0b-lite` 里把它作为与 global readout 并列的读出策略
- 看 delayed refresh 条件下，哪种读出更适合哪类数据

这样既保留 `B3` 的价值，也避免它过早变成全局默认。

## 当前推荐的系统级判断

如果只压成一句话：

**当前框架已经足够兼容下一步的受控 delayed refresh，但还不适合直接跳进完整 slow-layer closed loop；最稳的推进方式是先把现在这条快层主干当成冻结前端，进入一次 `P0b-lite`。**
