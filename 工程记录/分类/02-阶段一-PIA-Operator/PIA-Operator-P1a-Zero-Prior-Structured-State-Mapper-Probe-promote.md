# PIA Operator P1a Zero-Prior Structured State Mapper Probe

## 任务名称
`SCP-Branch P1a: Zero-Prior Structured State Mapper Probe`

## 零、战略定位
`P0/P0a.1` 当前已经把冻结慢层下的快层主干基本跑通：

- `A2r` 响应器已经数值健康
- `C3LR` 判别闭式解已经能把模板方向显著拉向局部判别法向
- `B3` 连续几何耦合已经证明“全局方向 + 局部执行”这条路是成立的
- `r > 1` 已经证明当前框架不应继续被 `r=1` 单轴容量绑住

但 `R0 / P0b-lite` 又进一步暴露出新的瓶颈：

- 在 `NATOPS` 这类相对状态型数据集上，`R0` 能收正收益
- 在 `FingerMovements / UWaveGestureLibrary` 这类轨迹型、多模态数据集上，`R0` 会明显掉分

这说明当前最深的问题已经不再首先是：

- 数值域没修好
- 判别轴学不出来
- 或局部耦合完全不存在

而更像是：

> 当前框架仍然默认“单一全局几何对象 + 单一全局闭式解 + 统一写回”足以覆盖多模态时序流形。

`P1a` 的目标不是继续在这个全局对象观上打补丁，而是单独开一个新 probe：

> 把 PIA 从“冻结全局算子”推进成“由冻结规则生成的条件局部算子”，检查局部切空间闭式解是否能在轨迹型数据集上消除全局算子失稳。

一句话说：

> `P1a` 不是 slow refresh，也不是再修一个全局模板，而是第一次正式测试“局部切空间条件算子”是否比“静态全局算子”更贴合当前数据流形。

---

## 一、唯一核心问题
**在保持 backbone、terminal、广义逆闭式解核心和 train-only 边界不变的前提下，若把全局单次闭式解改成“按 query window 条件生成的局部 bipolar 闭式解”，是否能显著缓解轨迹型多模态数据集上的全局算子失稳。**

这一步只问：

- 全局静态 `W` 是否是当前多模态轨迹任务的主瓶颈
- 局部切空间 `W_local(z_t)` 是否能比全局 `W` 更稳
- 这种局部化是否足以把 `FingerMovements` 拉回到 baseline 之上

当前不问：

- Memory Bank 最终形态是否已经确定
- Dual-Role Readout 是否已经必须并入
- slow refresh 是否应该开启
- 完整 global-local 双层框架是否已经可以一次性落地

---

## 二、与当前工程的关系
`P1a` 必须和现有分支严格区分：

- `C3LR` 修的是判别目标
- `B3` 修的是全局方向与局部几何的执行耦合
- `R0/P0b-lite` 修的是 post-fast 后的对象重对齐或一次 delayed rebuild
- `P1a` 修的是：
  - **算子求解的作用域**
  - 从全局固定 pool 改成 query-conditioned local pool

因此：

> 若 `P1a` 成功，它不能被表述成“又一个权重核调好了”。

它更准确地意味着：

- 当前多模态任务需要局部切空间求解
- 单个冻结全局 `W` 不是充分表达
- PIA 有潜力从“静态全局算子”升级成“条件局部算子生成器”

---

## 三、必须守住的边界

### 1. 不改闭式解核心
这一轮仍必须保留：

- 广义逆 / 正规方程 / 闭式解主线
- train-only 拟合
- freeze 后作用于 `train/val/test`

不允许改成：

- 反向传播
- 多 epoch 端到端训练
- 在线增量更新参数

### 2. 不使用 test 真标签构造局部池
这是本轮硬约束。

`P1a` 阶段 1 允许：

- 用 frozen geometry 做 coarse routing
- 用 query 与 train local objects 的距离做邻域检索

不允许：

- 用 test/val 真标签决定 same/opp pool
- 用 test-time 真类别直接挑局部邻居

### 3. 阶段 1 只做最小局部 probe
这一轮不一次性并入：

- Constructive Memory Bank
- Dual-Role Readout
- slow refresh
- full global-local 双层系统

阶段 1 只回答：

> 局部切空间闭式解这件事本身值不值得继续往下做。

### 4. 外部主对比对象固定为 `raw + MiniROCKET`
从 `P1a` 开始，外部主结果默认只认：

- `raw + MiniROCKET`

规则：

- 若该数据集已有官方 fixed-split `raw + MiniROCKET` 结果，则它是唯一默认外部 headline
- `B0 / F1` 仍然保留，但只作为内部归因与消融对照
- 主结论不能再写成“优于同 backbone baseline 即成立”
- 必须优先回答：
  - `P1a` 相对 `raw + MiniROCKET` 还差多少
  - 以及相对 `B0 / F1` 具体改善了什么

---

## 四、P1a 总体蓝图
如果按完整愿景看，`P1a` 是从：

- `global fixed operator`

过渡到：

- `conditional local operator generator`

更准确地说，未来完整 `P1a` 目标形态是：

1. 全局层给 coarse discriminative prior
2. 局部层在当前样本附近的切空间上生成 `W_local(z_t)`
3. 终端读出的不再只是“被推过的原始坐标”，而是更结构化的建构/判别双角色坐标

但这一轮先只落最小阶段。

---

## 五、阶段 1：Offline Local WLS Probe

### 目标
在不引入 Memory Bank、不引入双轨输出、不改 terminal 接口的前提下，先做：

> 在 frozen geometry 上，为每个 query window 独立构建局部 bipolar pool，并求一个 `W_local(z_t)`。

### 主测试站

- dataset: `fingermovements`
- seed: `1`
- terminal: `dynamic_minirocket`

原因：

- 这是当前 `R0` 的明确负向站
- 最适合检验“局部切空间理论”是否真的能缓解全局失稳

### 对照臂

#### `B0`
`same_backbone_no_shaping`

作用：

- 给出当前 backbone 原始参考
- 仅作为内部归因对照，不作为外部主 headline

#### `F1`
`current_global_mainline`

当前默认定义：

- `A2r + C3LR + r4 + global readout`

作用：

- 给出当前最强全局闭式解主线
- 仅作为内部归因对照，不作为外部主 headline

#### `P1a-S1`
`offline_local_bipolar_wls`

作用：

- 给出局部切空间闭式解的第一版最小实现

#### `RAW`
`raw + MiniROCKET`

作用：

- 当前默认外部主对比对象
- 用于回答 `P1a` 离外部强参考还有多远

---

## 六、阶段 1 的唯一实现对象
这一步只改：

- 算子拟合的作用域
- 从全局单次 pool 改成 query-conditioned local pool

不改：

- backbone
- terminal
- `A2r` 响应形态
- smoothing / budget matching 机制
- `C3LR` 的闭式解数学核心

---

## 七、阶段 1 的数值定义

### 1. Coarse routing
对每个 query window `z_t`：

- 在 frozen geometry 的全部 prototypes 中找最近 prototype
- 记其为当前 `same prototype`
- 再按 frozen geometry 规则，为该 prototype 取固定最近 opposite prototype

说明：

- 这里的“same/opp”是 **router 语义**
- 不是 query 的真标签

### 2. Local bipolar pool
在 router 决定的：

- same prototype admitted windows
- paired opposite prototype admitted windows

中分别按 query 距离取局部最近邻。

第一版默认：

- `K_same = K_opp = anchors_per_prototype / 2`
- 即沿用现有 `anchors_per_prototype = 8` 时，默认 `K = 4`

这不是新的自由超参，而是：

- 对当前 frozen admitted pool 的局部化再利用

### 3. Local pair axis
对 same / opp 局部邻域分别算加权中心：

\[
\mu_{same}(z_t)=\frac{\sum_i w_i^{same} z_i}{\sum_i w_i^{same}}
\]

\[
\mu_{opp}(z_t)=\frac{\sum_j w_j^{opp} z_j}{\sum_j w_j^{opp}}
\]

定义当前 query 的局部判别轴：

\[
u_{pair}(z_t)=normalize(\mu_{same}(z_t)-\mu_{opp}(z_t))
\]

### 4. Local weighting
same 与 opp 两侧仍沿用 `C3` 中已经验证过的 `median-min` 衰减：

\[
w=\exp\left(-\frac{d-d_{min}}{\max(d_{med}-d_{min},1e-8)}\right)
\]

然后对 same / opp 两侧分别做总质量归一化。

### 5. Local closed-form fit
以当前 local bipolar pool 为拟合集，直接求：

\[
W_{local}(z_t)=\arg\min_M \|\Lambda^{1/2}(PM-Y_{disc})\|^2+\lambda\|M\|^2
\]

其中：

- same 目标为 `+1 * u_pair(z_t)`
- opp 目标为 `-1 * u_pair(z_t)`

这一步仍是广义逆闭式解，不改求解器。

### 6. Local execution
第一版不并入 B3。

对每个 query window：

- 用它自己的 `W_local(z_t)` 得到 `r_t`
- 在序列内做 local median centering
- 用各自 local fit pool 的 IQR scale 做归一
- 再用 `tanh` 形成 gate
- 最后沿 `u_local(z_t)` 执行扰动

也就是：

\[
\Delta z_t = \epsilon \cdot local\_step(z_t)\cdot a_{resp}^{local}(z_t)\cdot u_{local}(z_t)
\]

其中：

- `u_local(z_t)` 就是 local operator 的 readout direction

---

## 八、必须输出的诊断

### 保留
- `test_macro_f1`
- `operator_to_step_ratio_mean`
- `response_vs_margin_correlation`
- `template_mean_direction_cosine`

### 新增
- `raw_minirocket_reference_available`
- `raw_minirocket_reference_source`
- `raw_minirocket_reference_test_macro_f1`
- `baseline_delta_vs_raw_minirocket`
- `f1_delta_vs_raw_minirocket`
- `p1a_delta_vs_raw_minirocket`

- `router_mode`
  - `nearest_prototype_frozen_geometry`

- `local_pool_same_count_mean`
- `local_pool_opp_count_mean`
- `local_same_weight_mass_mean`
- `local_opp_weight_mass_mean`
- `local_pair_axis_router_cosine_mean`
  - 当前 local pair axis 与 router prototype pair axis 的平均 cosine

- `local_template_direction_cosine_mean`
  - 当前 local operator direction 与 local pair axis 的平均 cosine

- `local_operator_fit_count`
  - 当前总共求解了多少次 local closed-form

- `local_operator_runtime_seconds`
  - 局部闭式解耗时

---

## 九、放行标准与 Stop Rule

### 弱成立
- `P1a-S1` 相比 `F1`
  - `FingerMovements` 上 `test_macro_f1` 明显回升

### 中等成立
- `P1a-S1` 相比 `B0`
  - 至少回到 baseline 之上
  - 且 `operator_to_step_ratio_mean` 未失控
- 同时：
  - 相对 `raw + MiniROCKET` 的差距明显小于 `F1`

### 强成立
- `P1a-S1` 相比 `B0 / F1`
  - 明显优于二者
  - 且显著缩小与 `raw + MiniROCKET` 的差距
  - 且 `response_vs_margin_correlation` 不进一步恶化
  - 准许进入 Memory Bank 接管阶段

### 熔断
若出现以下任一情况，则本轮停止：

- local operator 总耗时 > terminal 训练/评估耗时的 `5x`
- `FingerMovements` 相比 `F1` 再次显著恶化
- same/opp 局部池经常退化到极小无效规模

---

## 十、失败时的解释

### 情形 A
局部闭式解仍显著掉分

说明：

- 当前问题不只是“全局 `W` 太粗”
- 或 frozen geometry router 本身已经不够好

### 情形 B
方向指标升了，但 `test_macro_f1` 仍差

说明：

- 局部轴理论有信号
- 但执行读出或 router 仍不够稳

### 情形 C
局部 probe 成功，但耗时失控

说明：

- 理论成立
- 但必须走 Memory Bank / cache / family router 的工程化路线

---

## 十一、一句话执行目标
**在保持 backbone、terminal、A2r 和广义逆闭式解核心不变的前提下，先在 `FingerMovements` 上用 frozen geometry 做一次最小的 query-conditioned local bipolar WLS probe，验证“局部切空间条件算子”是否足以缓解全局静态算子在多模态轨迹流形上的失稳。**
