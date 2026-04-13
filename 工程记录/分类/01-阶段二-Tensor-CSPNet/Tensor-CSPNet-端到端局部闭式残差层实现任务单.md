# Tensor-CSPNet 端到端局部闭式残差层实现任务单

更新时间：2026-04-13

阅读提示：

- 如果你想先知道这个任务单在整个分类工程里的位置，请先看：
  - [分类二阶段现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类二阶段现状.md)
- 这份文件不负责回顾全部一阶段历史，而只负责当前二阶段实现与实验推进。

## 0. 阶段定位与阶段目标

### 0.1 本次任务在总蓝图中的阶段定位

本次任务不再属于旧的“冻结慢层下的快层 operator family”继续加分支，也不属于 `P1a offline local-WLS` 的延长线。

本次任务在总蓝图中的准确位置是：

**从“外部宿主 baseline 已复现”进入“第一版端到端局部闭式残差层 smoke 实现阶段”。**

它承接的是两类既有证据：

- 内部结构证据：
  - 当前主线仍停在 [PIA-Operator-当前主线总览.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-当前主线总览.md) 所描述的 `P0a.1` 尾部
  - [分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类工程现状.md) 已明确当前主问题正在从 `axis -> force` 转向 `operator scope`
  - [PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md](/home/THL/project/MTS-PIA/工程记录/分类/02-阶段一-PIA-Operator/PIA-Operator-P1a-Zero-Prior-Structured-State-Mapper-Probe-promote.md) 与 [分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类/00-入口/分类调试记录.md) 已给出 “全局固定算子作用域不足” 的强信号
- 外部宿主证据：
  - `Tensor-CSPNet` 已在本机完整复现到 `BCIC-IV-2a holdout = 0.7238`
  - 结果文件位于：
    - [tensor_cspnet_holdout_group_gpu0_20260411_032842.csv](/home/THL/project/MTS-PIA/archive/reference_code/Tensor-CSPNet-and-Graph-CSPNet/results/tensor_cspnet_holdout_group_gpu0_20260411_032842.csv)
    - [tensor_cspnet_holdout_group_gpu1_20260411_032805.csv](/home/THL/project/MTS-PIA/archive/reference_code/Tensor-CSPNet-and-Graph-CSPNet/results/tensor_cspnet_holdout_group_gpu1_20260411_032805.csv)
    - [tensor_cspnet_holdout_group_gpu2_20260411_033455.csv](/home/THL/project/MTS-PIA/archive/reference_code/Tensor-CSPNet-and-Graph-CSPNet/results/tensor_cspnet_holdout_group_gpu2_20260411_033455.csv)

### 0.2 本阶段目标

本阶段目标不是直接完成全文方法，也不是扩展到 `Graph-CSPNet / CV / KU / 多数据集 formal`。

本阶段只做四件事：

1. 以 `Tensor-CSPNet` 为宿主，做第一版**端到端局部闭式残差层**实现
2. 默认把局部闭式层作用在 `Temporal_Block` 后的低维 latent 上
3. 用残差方式融合原 classifier，而不是直接替换原头
4. 在 `BCIC-IV-2a holdout` 上验证：
   - 训练稳定
   - `local closed-form residual head` 至少不弱于原始宿主
   - 并且优于“只是多加一个 residual 头”的普通对照

### 0.3 本阶段不做

- 不做 `Graph-CSPNet`
- 不做 `CV` 协议
- 不做 `KU` 数据集
- 不做第二插入点并行搜索
- 不做 `B3 / delayed refresh / curriculum` 的重新引入
- 不做 oracle same/opp 路由
- 不做大量超参矩阵

### 0.4 当前加速阶段进展

`2026-04-12` 起，二阶段在不改变方法定义的前提下，新增一层“实现级加速”任务。

这层加速的目标是：

- 不改 `E0 / E1 / E2` 的方法学定义
- 不先动 `epochs / seeds / protocol`
- 先把 Tensor-CSPNet 宿主实现从“可跑”推进到“高效可迭代”

当前已落地的第一波改动包括：

1. `modeig_forward -> batched eigh`
2. `BatchDiag -> diag_embed`
3. `BiMap / Graph_BiMap` 通道循环向量化
4. 结构化混合精度：
   - SPD 路径保持 `float64`
   - `Temporal_Block / Classifier / residual heads / beta / prototypes` 改为 `float32`
5. `BCIC holdout Tensor-CSPNet` subject 级缓存

当前第一条已完成的等价性验证：

- [acceleration_pass1_summary_20260412.md](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/acceleration_pass1_summary_20260412.md)
- `subject 5`
  - 旧实现：`acc=0.6076`, `loss=1.090845`, `wallclock=6188.9s`
  - 新实现：`acc=0.6007`, `loss=1.097744`, `wallclock=486.3s`

当前读法：

- 第一波实现级加速已经基本成立
- 下一步是补 `subject 1` 等价性点
- 然后进入 `batch 29 / 58 / 87` 的吞吐-精度联合验证

### 0.5 当前最新实验结果

当前已经有两组值得当阶段锚点的结果：

1. `fp64 mainline`
2. `GPU2 fp32`

当前 `fp64 mainline` 单 seed 全量结果：

- `E0 = 0.7103`
- `E1 = 0.7083`
- `E2 = 0.7018`

当前 `GPU2 fp32` 聚合结果：

- `E0 = 0.7118`
- `E1 = 0.7114`
- `E2 = 0.6983`

权威汇总入口：

- [stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/notes/stage2_mainline_fp64_and_gpu2_fp32_summary_20260413.md)

当前读法：

- `SPD fp32` 已经被证明是一个可工作的候选数值策略
- 但 `E2` 目前还没有稳定优于 `E0 / E1`
- 本任务单后续的重点应进一步转向：
  - `E2` 的方法改造
  - 而不是继续无上限扩训练链路本身

## 1. 本阶段架构冻结

### 1.1 宿主 backbone

宿主固定为上游参考实现：

- [archive/reference_code/Tensor-CSPNet-and-Graph-CSPNet/utils/model.py](/home/THL/project/MTS-PIA/archive/reference_code/Tensor-CSPNet-and-Graph-CSPNet/utils/model.py)
- 目标对象：`Tensor_CSPNet_Basic`

本阶段不修改上游参考仓库源码；新实现通过本库 wrapper / adapter 接入。

### 1.2 默认插入位置

默认插入位置固定为：

**`Temporal_Block` 之后的低维 latent。**

原因：

- 这是当前宿主已经天然形成的低维判别子空间
- 比在 `LogEig` 后额外补 `bottleneck` 更自然
- 超参更少
- 数值更稳

备选位置“`LogEig` 后 + bottleneck”只作为后备方案，不进入本阶段实验矩阵。

### 1.3 融合方式

第一版统一采用残差融合：

`final_logit = base_logit + beta * local_closed_form_logit`

其中：

- `base_logit` 来自原始 Tensor-CSPNet classifier
- `local_closed_form_logit` 来自新加的局部闭式残差头
- `beta` 为可学习标量或极低维门控，默认从小值开始

本阶段不允许把原 classifier 直接删掉。

### 1.4 局部判别层的结构约束

第一版局部闭式层只保留四个核心部件：

1. learnable class prototype bank
2. soft routing / attention
3. latent-space local closed-form solve
4. residual logit fusion

闭式解一律在低维 latent 空间里做，不直接吃高维 `LogEig` 展平特征。

### 1.5 允许的核心自由度

本阶段只允许以下四个核心自由度进入配置：

- 每类 prototype 数 `M`
- soft routing 温度 `tau`
- 闭式解 ridge `c`
- 残差门控 `beta`

除此之外，不再继续扩新的“力学/刷新/课程/多读出”超参。

## 2. 第一版实验矩阵

本阶段实验矩阵固定为三个 arm：

### E0. 原始宿主

- `Tensor-CSPNet`

用途：

- 外部宿主 baseline
- 必须对齐已经复现的 `BCIC-IV-2a holdout = 0.7238`

### E1. 普通残差头对照

- `Tensor-CSPNet + residual linear adapter`

用途：

- 排除“只要多加一个头就会提点”的解释
- 作为最小结构增强对照

### E2. 方法主臂

- `Tensor-CSPNet + local closed-form residual head`

用途：

- 验证增益是否来自“局部闭式几何判别”，而不是普通 residual 扩容

本阶段不增加第四个 arm。

## 3. 代码实现任务

### T1. 宿主包装层

目标：

- 在不改上游源码的前提下，拿到：
  - `latent h`
  - `base_logit`
  - `final_logit`

建议文件：

- `models/tensor_cspnet_adapter.py`

最低要求：

- 能复用上游 `Tensor_CSPNet_Basic` 的 `BiMap_Block / LogEig / Temporal_Block / Classifier`
- 能在 forward 中显式返回 `h` 与 `base_logit`

完成标准：

- shape 对齐
- 参数初始化与原宿主兼容
- `E0` 包装后精度不应明显偏离 `0.7238`

### T2. 普通残差头对照实现

目标：

- 先实现 `E1`

建议文件：

- `models/tensor_cspnet_residual_linear.py`

最低要求：

- 输入：`Temporal_Block` 后 latent `h`
- 输出：`residual_logit`
- 融合：`base_logit + beta * residual_logit`

完成标准：

- 训练稳定
- 可以作为 `E2` 的强对照

### T3. 局部闭式残差头实现

目标：

- 实现 `E2`

建议文件：

- `models/local_closed_form_residual_head.py`

最低要求：

- learnable prototype bank
- soft routing
- class-wise local support construction
- `torch.linalg.solve` 求局部闭式解
- 输出 `local_closed_form_logit`

完成标准：

- 前向无 NaN / Inf
- backward 可跑通
- 不依赖 test 真标签

### T4. 统一训练入口

目标：

- 统一 `E0 / E1 / E2` 的训练与评测入口

建议文件：

- `scripts/run_tensor_cspnet_local_closed_form_holdout.py`

最低要求：

- 同一数据协议
- 同一随机种子接口
- 同一日志输出格式
- arm 通过参数切换

完成标准：

- 能单独跑 `E0`
- 能单独跑 `E1`
- 能单独跑 `E2`

### T5. 结果记录与汇总

目标：

- 固定输出格式，便于后续写文档和比较

建议输出：

- `per-subject csv`
- `aggregate csv`
- `run_meta json`

最低要求：

- 保存每个 subject 的 `acc / loss / wallclock`
- 保存本次 arm、`M / tau / c / beta` 配置

## 4. 实验执行顺序

### S1. 代码级 smoke

只验证：

- forward / backward
- shape
- 无数值爆炸

不要求先跑完整 9-subject。

### S2. `subject 1` 单臂 smoke

顺序：

1. `E0`
2. `E1`
3. `E2`

目的：

- 先确认 `E2` 没把宿主训练链路拖崩

### S3. `BCIC holdout` 单 seed 全量

顺序：

1. `E0`
2. `E1`
3. `E2`

目的：

- 做第一次有意义的宿主级比较

### S4. `5 seeds` formal

前提：

- `E2` 在单 seed 上显示出正信号

目的：

- 验证增益是否稳定

## 5. 结果判定标准

### 5.1 本阶段成功标准

满足以下三条即可视为本阶段成功：

1. `E2` 训练稳定，不出现系统性数值崩溃
2. `E2` 不弱于 `E0`
3. `E2` 明显优于 `E1`

### 5.2 本阶段最理想结果

若满足以下条件，则进入论文候选状态：

- 单 seed 已明显优于 `E0`
- 多数 subject 为正增益
- `5 seeds` 后平均增益仍为正
- `E2 > E1` 关系稳定成立

## 6. 失败时的回退规则

若 `E2` 失败，不允许立刻扩超参矩阵。

先按以下顺序排查：

1. `beta` 是否过大
2. ridge `c` 是否过小
3. routing 是否过硬
4. prototype 数 `M` 是否过大
5. 求解是否发生病态矩阵问题

只有在这些基本数值问题排除后，才允许讨论第二插入点或额外课程学习。

## 7. 当前最重要的约束

本任务单只服务于一件事：

**把“局部闭式判别层”第一次做成一个少超参、端到端、可直接挂在已发表宿主上的方法原型。**

因此后续实现中必须持续遵守：

- 不回到旧的冻结几何主线
- 不把 offline local-WLS 直接搬成正式系统
- 不把上游宿主改成无法归因的新网络
- 不在第一版同时扩多个插入位置
- 不在第一版同时扩多个复杂正则与课程
