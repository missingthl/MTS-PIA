# PIA Operator Value Probe Promote

## 任务名称
`SCP-Branch P0: PIA Operator Value Probe`

## 零、战略定位
当前框架已经逐步收束出一个更清晰的系统分工：

- 表示层负责把 raw EEG 映射到局部状态轨迹 `z_seq`
- 慢层负责刷新局部流形参考
- 快层负责在当前参考几何下对局部状态施加判别作用

与此同时，一个更根本的问题已经被逼出来：

> **PIA 本体是否真的比“均值 / prototype + 同样调制机制”多带来独立价值。**

如果这件事不先回答，后续无论把 PIA 写成：

- `PIA-Geom`
- `PIA-Disc`
- class-conditioned template family
- prototype-conditioned local operator

都会越来越难说明：
- 优势到底来自 `PIA` 本体
- 还是来自后续外加的门控、prototype、调制和局部控制机制

因此，这一轮不是继续扩系统，而是做一个最小、干净、可归因的价值对照实验。

一句话说：

> 在进入“新型 PIA”之前，先验证 `PIA operator` 相比 `mean/prototype-centered operator` 是否表现出独立价值。

---

## 一、唯一核心问题
**在完全相同的 backbone、对象定义、局部门控和调制预算下，`single-template PIA operator` 是否比 `mean/prototype-centered update` 带来更好的局部几何改善与终端可读性。**

这一步只问：
- `PIA operator` 本体是否有独立价值

当前不问：
- 多轮闭环是否成立
- continual / online calibration 是否成立
- 慢层 refresh 是否已经最终定型
- `SEED family` 是否同步成立
- class-conditioned PIA 是否已经是最终答案

---

## 二、为什么现在必须先做这个

### 1. 当前 PIA 的理论身份和工程身份已经开始分叉
理论上，PIA 应该是：
- 模板响应函数
- 几何作用算子
- 不只是点估计器

但当前工程里，PIA 仍有相当一部分行为接近：
- `mean + direction`
- `prototype-aware local push`

所以现在必须先验证：
- 当前工程里的 `PIA` 是否真的释放了比均值法更多的东西

### 2. 如果这一步不做，后面“新 PIA”会越来越难解释
继续直接做：
- `PIA-Geom`
- `PIA-Disc`
- class-conditioned templates
- 新快慢闭环

会导致一个风险：

> 系统越来越复杂，但无法回答“到底为什么非 PIA 不可”。

### 3. 这是当前所有后续路线的前置判断
如果 `PIA` 明显优于 mean/prototype-centered operator：
- 才值得继续把它上升为框架核心算子

如果两者差不多：
- 就应当更谨慎地把 PIA 退回为一种几何先验/可选模块

---

## 三、当前全部冻结
这一轮必须极度克制，避免再把变量混在一起。

### Backbone 固定
- `raw -> dense z_seq -> terminal`

### 当前对象定义固定
- `prototype-memory`
- `local representative states`
- `v1b tight anchors`
- 当前 `train-only` 局部对象定义

### 终端固定
- `dynamic_minirocket`

原因：
- 当前 `SCP` 分支默认终端就是这条
- 本轮不再比较 head

### 训练/评估协议固定
- train-only
- offline
- single-round
- 不做 replay
- 不做 curriculum
- 不做 online / test-time update
- 不做 NATOPS 兼容
- 不做双流
- 不引入新 gate zoo

补充硬约束：
- `Baseline 0` 继续在原始 dense trajectory 上训练/评估
- `Arm A / Arm B / Arm C` 都必须只在 `train-only` 上拟合 operator 参数
- operator 一旦拟合完成，必须冻结，并以**完全相同的参数**作用于 `train/val/test`
- 这不是 test-time adaptation；它只是 frozen operator 下的同参考系变换

---

## 四、主比较对象
本轮只做 `SCP1`。

### Baseline 0
`same_backbone_no_shaping`

作用：
- 给出“什么都不做”的终端参考

### Arm A
`mean/prototype-centered modulated update`

定义：
- 使用相同的 prototype-memory 与 tight anchors
- 更新方向只允许基于：
  - same prototype / representative
  - nearest opposite prototype / representative
- 不允许引入 PIA 模板学习

这条线代表：

> 如果没有 PIA，本框架还能做到的最强 mean/prototype-centered operator。

### Arm B
`single-template PIA update`

定义：
- 使用 `TELM2/PIA` 的单模板算子
- `r_dimension = 1`
- train-only pooled windows 学一个模板方向
- 其它调制预算与 Arm A 保持一致

这条线代表：

> 当前最小 PIA 本体相对于均值法是否多带来东西。

### Arm C（附加实验，非主实验）
`class-conditioned PIA update`

定义：
- 使用 train-only one-vs-rest 或 class-conditioned template family
- 每类一个判别模板方向
- 其它对象定义与 Arm A/B 保持一致

这条线代表：

> 如果把 PIA 从“自表征模板”提升为“判别模板”，它是否比均值法和单模板 PIA 更合适。

当前定位：
- 不是本轮主问题
- 只作为 second-stage probe
- 仅在 `Arm A vs Arm B` 站稳后才值得正式升格
- 若 `Arm A vs Arm B` 未出现明确差异，`Arm C` 不进入本轮实现

---

## 五、第一版方法学约束

### 1. 三条线必须同预算
以下全部必须一致：
- same backbone
- same prototype-memory
- same admitted anchors
- same local step约束
- same operator拟合口径
- same operator作用范围
- same train split
- same terminal

也就是说：

> 只有 operator 本体不同，不允许偷偷给 PIA 致优条件。

### 2. 主实验先只做 `Arm A vs Arm B`
当前主实验只比较：
- `Arm A: mean/prototype-centered update`
- `Arm B: single-template PIA update`

`Arm C` 不进入主结论。

### 3. 主实验只做快层局部作用
当前不把慢层 geometry refresh 混进来。

原因：
- 不然 `PIA operator` 和 `reference refresh` 会再次缠在一起

这一轮测的是：

> 当前参考几何固定时，局部作用规则是否必须依赖 PIA。

进一步写死：
- 慢层参考几何固定
- 不做 refresh
- 不做 candidate 生成
- 不做 rollback 审计链

### 4. 不做 test-time routing
- 不做 class-conditioned test-time container selection
- 不做任何 test-time 自适应更新

### 5. 不允许在 test-time 额外改写 `z_seq`
- 禁止在 test-time 使用与训练期不同的 operator 口径
- 若某个 operator arm 在训练期对 `z_seq` 做了变换，则同一 frozen operator 必须同样作用于 `val/test`
- 不允许 train/test 的流转分布因为 operator 口径不同而发生不一致

---

## 六、三条线的最小数学口径

### Arm A: mean/prototype-centered modulated update
对一个被选中的局部状态 `z`：

- `p_same`: same prototype / representative
- `p_opp`: nearest opposite prototype / representative

方向可写成：

\[
d_A(z)=\lambda_{intra}(z)(p_{same}-z)+\lambda_{inter}(z)(z-p_{opp})
\]

其中：
- 调制项允许存在
- 但只允许标量门控
- 禁止引入额外全局 reference
- 禁止引入 PIA 模板响应

### Arm B: single-template PIA update
对同一对象：

- 先用 train windows 学单模板 `h(x)=\phi(w^T x+b)`
- 再把模板响应转成局部作用方向

要求：
- `\phi` 必须显式非线性
- 第一版锁定为当前 `TELM2` 可实现的非线性激活，不允许 `linear / identity`
- 默认优先：`activation = sigmoid`
- 不额外引入 class-conditioned routing
- 不额外引入多模板 family

### Arm C: class-conditioned PIA update
对每个类 `c`：

- 用 train-only `OvR` 学一套类模板
- 对属于类 `c` 的训练局部状态，用对应模板生成局部作用方向

第一版写死：
- 单模板/单轴 per class
- 不引入 cluster-wise template zoo
- 不写成 template family

---

## 七、必须输出的文件
- `pia_operator_value_config_table.csv`
- `pia_operator_value_per_seed.csv`
- `pia_operator_value_dataset_summary.csv`
- `pia_operator_value_structure_diagnostics.csv`
- `pia_operator_value_score_diagnostics.csv`
- `pia_operator_value_acceptance_summary.csv`
- `pia_operator_value_conclusion.md`

---

## 八、必须看的指标

### 结构指标
- `delta_between_prototype_separation`
- `delta_nearest_prototype_margin`
- `delta_within_prototype_compactness`
- `delta_temporal_assignment_stability`
- `local_step_distortion_ratio_mean`
- `local_step_distortion_ratio_p95`

### 终端指标
- `test_macro_f1`
- `delta_vs_no_shaping`
- `delta_vs_mean_centered`
- `margin_to_score_conversion`
- `margin_gain_per_unit_distortion`

### 审计指标
- `operator_norm_mean`
- `operator_direction_stability`

可选补充指标：
- `accepted_window_ratio`
- `shaped_window_ratio`

---

## 九、必须回答的问题
1. `single-template PIA` 是否优于 `mean/prototype-centered update`
2. 如果 `PIA` 没有明显优于 mean-centered，问题更像：
   - PIA 本体未释放优势
   - 当前对象定义太弱
   - 还是当前 readout 不匹配
3. 若 `margin ↑` 但 `score` 不升，是否说明：
   - 终端 readout 还读不到这类几何增益
4. `PIA` 相对 mean-centered update 的增益，到底更像来自：
   - 模板响应的区域选择性
   - 还是仅仅来自更强的方向偏置
5. 附加实验里，`class-conditioned PIA` 是否优于：
   - `mean/prototype-centered update`
   - `single-template PIA`
6. 在相同或更小的几何扰动下，`PIA` 是否能带来更高的 margin 增益效率

---

## 十、成功标准

### 弱成立
- `single-template PIA` 在结构指标上优于 mean-centered arm
- 且没有更明显的 step distortion

### 中等成立
- `single-template PIA > mean/prototype-centered update`
- 并开始出现非零 `margin_to_score_conversion`

### 强成立
同时满足：
- `single-template PIA` 在结构指标和终端分数上都优于 mean-centered arm
- 可以明确说明：
  - PIA 的优势来自模板算子本体
  - 不是只来自后续调制机制

---

## 十一、若结果不成立，如何解释

### 情形 A
`mean-centered ≈ single-template PIA`

说明：
- 当前单模板 PIA 本体尚未证明独立必要性

### 情形 B
`class-conditioned PIA > single-template PIA`，但 `≈ mean-centered`

说明：
- “分类化方向”可能有用
- 但 PIA 仍未证明相对 mean/prototype-centered operator 的根本优势

### 情形 C
`PIA arms` 结构改善但终端不读

说明：
- 可能是 current readout mismatch
- 不应直接判为 PIA 无效

---

## 十二、与后续路线的关系
只有这一轮站住，后续才值得判断：
- PIA 更适合进入慢层
- 还是更适合进入快层
- 还是只适合作为几何先验模块

如果这一轮站不住，则应改口：
- PIA 不是框架内核
- 至多是几何先验模块

---

## 十三、一句话执行目标
**在完全相同的 backbone、对象定义、调制预算和门控条件下，先用最干净的 `Arm A vs Arm B` 主实验，回答 `single-template PIA operator` 是否真的比 `mean/prototype-centered operator` 多带来独立价值；`Arm C` 只作为附加实验，不预设为本轮主结论来源。**
