# Phase15 冻结主线节点级系统诊断

更新时间：2026-03-19

## 1. 诊断范围

本报告只诊断当前冻结主线，不引入新分支，不修改协议语义，不接入：

- `Gate3`
- `Fisher / C0`
- `Controller Lite`
- `raw-bridge`

冻结主线固定为：

`raw trial -> 多频带带通 -> 窗口切分 -> 协方差 / SPD -> log-center -> z-space -> Step1B 多方向 PIA -> Gate1 / Gate2 -> LinearSVC -> trial aggregation`

当前证据面只覆盖已闭环的 5 组：

- `seed1`
- `har`
- `natops`
- `fingermovements`
- `selfregulationscp1`

`seediv / seedv` 当前证据不足，不写诊断结论。

本轮补充生成的配套诊断表在：

- `out/phase15_mainline_diagnostics_20260319/direction_health_table.csv`
- `out/phase15_mainline_diagnostics_20260319/gate_mechanism_delta_table.csv`
- `out/phase15_mainline_diagnostics_20260319/highdim_risk_summary.csv`

## 2. 总结论

### 2.1 当前框架有没有整体崩

没有整体崩。

证据：

- `seed1` baseline `macro_f1 = 0.7549`，与官方协议 MiniROCKET `0.7723` 只差约 `0.0174`
- `har / natops / fingermovements / selfregulationscp1` 的 baseline 都能稳定闭环
- `Step1B / Gate` 的机制指标不是随机噪声，能稳定改变：
  - `knn_intrusion_rate`
  - `margin_drop_median`
  - `flip_rate`

所以当前不是“表示层全坏”或“训练器全坏”，更像是：

- 中段增强链路可以改变几何结构
- 但这种改变经常没有对准真实测试分布

### 2.2 当前最像哪一类问题

当前最像：

1. `Step1B` 方向库错配导致的训练分布偏移
2. `Gate` 能做部分过滤，但主要改善几何指标，不能稳定把改善转成精度收益
3. `seed1` 的高维切空间不是唯一根因，但明显在放大 Step1B / Gate 的已有缺陷

### 2.3 当前最不像哪一类问题

当前最不像：

- 单纯的 `LinearSVC` 训练器问题
- 单纯的 `trial aggregation` 问题
- 单纯的 `SPD / log-center` 表示层已经失效

## 3. 节点级风险排序

当前 Top 6 风险排序如下：

1. `Node E: Step1B 多方向 PIA`
   - 最可能的主问题源
   - 主要风险：`distribution shift`
   - 次级风险：伪多样性

2. `Node F: Gate1 / Gate2`
   - 当前更像“弱纠偏器”，不是主源头
   - 风险：筛选不够语义化，改善机制值但不稳定改善精度

3. `Node D: z-space 高维切空间`
   - 当前更像问题放大器，不是唯一源头
   - 在 `seed1` 上最明显

4. `Node B + C: 协方差 / SPD + log-center`
   - 当前证据更像“可用但不完美”
   - 可能放大 long EEG 的类间平均化风险，但不是首要嫌疑

5. `Node A: 窗口切分`
   - 对 `seed1` 很重要
   - 更像把窗口语义变弱、让 aggregation 接手救场

6. `Node H: trial aggregation`
   - 不是源头
   - 更像“掩盖器 / 放大器”
   - 在 `seed1` 上会把小的窗口偏移放大成 trial-level 波动

`Node G: LinearSVC` 当前排位靠后，不是第一嫌疑点。

## 4. 节点级故障定位

### Node A：窗口切分

功能：

- 把 trial 切成多个窗口，供 SPD / z-space 计算

最可能引入的失真：

- 把长 trial 的连续语义切碎
- 让训练集中大量相似窗口进入同一 trial
- 在 `seed1` 上制造“窗口判别弱、trial 聚合强”的结构

更像：

- `distribution shift`
- 不是典型 mode collapse

异常信号：

- `seed1` baseline：`trial_f1 = 0.7549`，`window_f1 = 0.6493`，gap 约 `0.1057`
- fixed split 四组：`trial_f1 == window_f1`

判断：

- `seed1` 的窗口级语义明显弱于 trial 级语义
- 说明窗口不是稳定判别单元，trial aggregation 在兜底

证据强度：

- 强

### Node B：协方差 / SPD 构造

功能：

- 把多通道波形压成空间统计结构

最可能引入的失真：

- 对长 EEG 的时序细节做强压缩
- 把类间真正有用的时序差异折叠进二阶统计

更像：

- 可能导致表示偏差
- 当前不像 mode collapse 主源

异常信号：

- `seed1` baseline 仍然稳，说明 SPD 本身不是直接失效
- fixed split 四组同一表示链也能稳定闭环

判断：

- 当前证据不足以把 SPD 认定为主问题源
- 更像“可用表示”，但它把后续增强完全限制在二阶统计空间

证据强度：

- 中

### Node C：log-center

功能：

- 用训练集均值对 SPD 做对数中心化

最可能引入的失真：

- long EEG 上可能把类间有用的绝对偏移压平
- 把增强和真实测试分布都投到同一个全局中心附近

更像：

- 轻度 `distribution shift` 放大器

异常信号：

- 目前没有直接证据表明 log-center 单独导致崩溃
- `seed1` baseline 仍可用，说明 train-only log-center 没把判别结构直接抹掉

判断：

- 当前更像“次级放大器”，不是主源头

证据强度：

- 低到中

### Node D：z-space 向量化表示

功能：

- 把 SPD 上三角展开到欧式空间，供 PIA / Gate / LinearSVC 使用

最可能引入的失真：

- 高维近邻关系变脆
- 全局方向在局部邻域里失去语义
- 距离与 margin 的小扰动更容易放大

更像：

- 问题放大器
- 在 `seed1` 上会放大 `Step1B / Gate` 的已有偏差

异常信号：

- `seed1` feature_dim=`1953`
- 对照：
  - `har`=`45`
  - `selfregulationscp1`=`21`
  - `natops`=`300`
  - `fingermovements`=`406`

判断：

- `seed1` 的高维风险是明确存在的
- 但 baseline 在同一 z-space 里仍最稳，说明“高维本身”不是唯一根因
- 更准确的表述是：高维让错误方向更难被分辨，让 Gate 更难做可靠局部筛选

证据强度：

- 强

### Node E：Step1B 多方向 PIA

功能：

- 在 z-space 里沿方向库生成训练增强样本

最可能引入的失真：

- 把真实训练分布推向错误方向
- 形成“表面多方向、实际伪多样性”的训练扩张
- 当方向库混入坏方向时，增强越多，偏移越稳定

更像：

- `distribution shift` 主源
- 次级风险是“有效多样性下降”

异常信号：

- `seed1`：
  - baseline `0.7549`
  - Step1B `0.7529`
- `natops`：
  - baseline `0.6512`
  - Step1B `0.6467`
- `fingermovements`：
  - baseline `0.5100`
  - Step1B `0.5000`

判断：

- 当前最可能的问题源就在这里

证据强度：

- 很强

### Node F：Gate1 / Gate2

功能：

- 过滤增强样本，降低坏增强进入训练集的比例

最可能引入的失真：

- 如果筛选不够语义化，只是在删样本，不是在删坏样本
- 若阈值与高维局部结构错位，会出现“机制改善但精度不涨”

更像：

- 次级 `distribution shift` 校正器
- 不是 mode collapse 主源

异常信号：

- `seed1` accept rate 仍高：`0.9061`
- 但 `Step1B+Gate` 反而降到 `0.7445`

判断：

- Gate 不是完全无效
- 但当前不像一个足够强的语义过滤器

证据强度：

- 强

### Node G：LinearSVC

功能：

- 在 z-space 上做最终分类

最可能引入的失真：

- 对增强后混合分布做线性边界拟合时，可能对偏移敏感

更像：

- 末端承受者
- 不是当前主问题源

异常信号：

- baseline 最稳，说明同一个分类器在不增强时没有崩

判断：

- 不应把 LinearSVC 当第一嫌疑点

证据强度：

- 强

### Node H：trial aggregation

功能：

- 把窗口预测聚成 trial 预测

最可能引入的失真：

- 掩盖窗口级问题
- 把小的系统性窗口偏差放大成多数投票偏差

更像：

- 掩盖器 / 放大器
- 不是问题源头

异常信号：

- `seed1` 的 trial-window gap 很大
- fixed split 四组 gap 为 `0`

判断：

- `seed1` 上 aggregation 既在救 baseline，也在放大增强引入的小偏差

证据强度：

- 强

## 5. Step1B 专项结论

### 5.1 当前是否存在“表面多方向，实际少数轴主导”

当前没有证据支持“少数轴主导”。

证据：

- `seed1` direction usage fraction 几乎完全均匀：
  - `0.1978 ~ 0.2017`
  - entropy `1.6094`，接近 5 方向均匀分布理论上限
- `har / fingermovements` 也较均匀

所以当前不是“增强集中堆到一两个方向”的典型 mode collapse。

### 5.2 当前是否存在伪多样性

有高概率存在。

证据：

- 当前 `subset_size = 1`
- `mixing_stats.mean_abs_ai = 1.0`
- `avg_subset_size = 1.0`

这意味着：

- 每个增强样本本质上只沿一个全局方向移动
- 所谓“多方向”主要体现在全局样本集合层面，不是在单样本层面形成真正多轴组合

因此它更像：

- 全局均匀撒向 5 根轴
- 但这些轴本身质量不一致

### 5.3 当前更像方向库错配还是增强强度不足

当前更像：

- 方向库错配为主
- 增强强度固定带来的偏移放大为辅

依据：

1. 方向使用不偏，但方向有效性明显不均匀
   - `seed1` Step1B：5 个方向里只有 `1` 个方向的 `margin_drop_median > 0`
   - 其余 `4` 个方向都是负值
2. `natops` 更极端
   - 5 个方向全部 `margin_drop_median < 0`
3. `seed1` 的最坏方向 `dir=0`
   - `flip_rate ≈ 0.0251`
   - `margin_drop_median ≈ +0.0096`
   同时存在别的方向明显负值

这说明问题不是“方向不够多”，而是：

- 全局方向库里混着好坏方向
- 但当前采样策略对这些方向几乎等权

### 5.4 为什么 seed1 没超过 baseline

高概率原因是：

1. `seed1` 在高维 z-space 中，错误方向更难被局部邻域及时识别
2. Step1B 依然把大约一半训练窗口换成增强窗口
   - Step1B `train_selected_aug_ratio = 0.5`
3. 这些增强方向虽然降低了部分 intrusion，
   但没有稳定增加真实测试所需的判别 margin
4. 结果就是：
   - 几何指标有小改善
   - 但主分类边界没有得到真实收益

### 5.5 fixed split 四组与 seed1 的方向库行为差异

差异不在“是否被少数方向垄断”，而在“高维下坏方向代价更高”。

对比：

- `seed1`
  - 方向使用极均匀
  - 但 Step1B 后 F1 下降
- `har`
  - 使用也均匀
  - Gate 后有小幅收益
- `selfregulationscp1`
  - 方向有效性异质性更强，但 Step1B / Gate 仍有局部正信号

结论：

- 不是 seed1 的方向分布更偏
- 而是 seed1 的高维表示让“坏方向被等权使用”的代价更大

## 6. Gate 专项结论

### 6.1 Gate 是在过滤坏增强，还是在随机减样本

当前更像：

- 不是随机减样本
- 但也不是足够强的语义筛选器

依据：

- `seed1`：
  - `accept_rate_gate1 = 0.9500`
  - `accept_rate_final = 0.9061`
  - `knn_intrusion_rate: 0.2424 -> 0.2269`
- `har`：
  - `flip_rate: 0.005 -> 0.004`
  - `knn_intrusion_rate: 0.1603 -> 0.1585`

所以 Gate 确实会改变机制值，不是纯随机抽样。

但同时：

- `seed1 / natops` 精度没有涨
- 说明它筛掉的是“部分坏增强”，不是“决定边界错误的核心坏增强”

### 6.2 Gate1 / Gate2 谁在起主要作用

当前主要是 `Gate1` 在工作，`Gate2` 很弱。

证据：

- 所有数据集的 `tau_src_y` 都几乎是常数 `0.1`
- accepted / rejected 的 `src_dist` 差别只有数值级微小差异

这说明 `Gate2` 更像：

- 对固定步长 `gamma=0.1` 的长度裁剪
- 不是基于真实局部语义的筛选

反过来，`Gate1` 的 accepted / rejected 中心距离差异明显：

- `seed1`：
  - accepted median `10.74`
  - rejected median `21.47`

所以当前真正有判别力的是 `Gate1`，不是 `Gate2`。

### 6.3 每个已闭环数据集的 Gate 机理归类

- `seed1`
  - 类型：`a) 机制值改善但精度没涨，且精度下降`
  - 表现：
    - intrusion 改善
    - margin 中位数改善
    - flip 不变
    - trial F1 明显下降

- `har`
  - 类型：`c) 机制和精度都改善`
  - 但幅度很小，仍是局部正信号

- `natops`
  - 类型：`a) 机制值轻微改善但精度没涨`

- `fingermovements`
  - 类型：`b/c 之间，更接近 b)`
  - 精度有小回升
  - 但机制值几乎不变，说明收益更像边界偶然纠偏，不是稳定几何改善

- `selfregulationscp1`
  - 类型：`c) 机制和精度都改善`
  - 但仍是单 seed 局部正信号

### 6.4 seed1 中 accept rate 高但性能下降说明了什么

说明：

1. Gate 没有把“决定错误边界的那部分坏增强”足够强地筛掉
2. 当前 `90.6%` 的接受率，对 `seed1` 这样高维数据来说仍然偏松
3. 更重要的是，Gate 过滤后训练集里仍有 `47.7%` 是增强样本

因此 `seed1` 的 Gate 当前更像：

- 轻度纠偏
- 不足以逆转 Step1B 带来的训练分布偏移

## 7. 高维切空间专项结论

### 7.1 seed1 是否存在明显维度诅咒风险

存在。

对照：

- `seed1`: `1953`
- `fingermovements`: `406`
- `natops`: `300`
- `har`: `45`
- `selfregulationscp1`: `21`

`seed1` 明显是当前已闭环数据里最高维的一组。

### 7.2 高维本身是主问题源还是问题放大器

当前更像：

- 问题放大器
- 不是唯一主问题源

理由：

- 如果高维本身就是主问题源，那么 baseline 也应明显不稳
- 但 `seed1` baseline 反而是这组里最稳的 z-space 版本

更合理的解释是：

- 高维让局部近邻和全局方向之间更容易错位
- 于是 Step1B 的全局方向库、Gate 的局部距离阈值，都变得更脆

### 7.3 它如何影响 Step1B 与 Gate

对 Step1B：

- 让“等权使用的坏方向”更难被局部结构抵消
- 让增强后样本更可能远离真实测试分布

对 Gate：

- `Gate1` 的欧式中心距离更容易受高维放大
- `Gate2` 在高维下仍只是固定步长约束，无法补足语义筛选

### 7.4 当前 seed1 baseline 最稳，如何解释

高概率解释是：

- baseline 避免了额外增强偏移
- Step1B / Gate 在高维切空间里放大了伪方向与伪邻域结构

因此 `seed1` 当前更支持这样的判断：

- 高维不是独立主罪魁
- 但它让“方向库错配 + 弱 Gate”从小问题变成实质收益损失

## 8. 表示层专项结论

### 8.1 SPD / log-center / z-space 当前是不是主问题源

当前不支持把它们认定为主问题源。

理由：

- 同一表示链下 baseline 可以稳定闭环
- `seed1` baseline 与官方 MiniROCKET 只差小约 `0.017`
- fixed split 四组也都能跑出合理 z-space baseline

### 8.2 更准确的定位

当前更像：

- 表示层可用
- 中段增强错配是主问题

也就是说，当前最合理的故障定位不是：

- “表示已经坏了”

而是：

- “表示还能工作，但 Step1B 把样本沿错误或混合质量方向推开，Gate 又没有足够强地纠偏”

对 `log-center` 和 SPD 的进一步否定或确认，当前证据仍不足。

## 9. 窗口切分与 trial aggregation 专项结论

### 9.1 seed1 是否提示窗口语义不足

是。

证据：

- baseline 的 `trial_f1 - window_f1 ≈ 0.1057`
- Step1B 的 gap 仍有 `0.1035`
- Step1B+Gate 仍有 `0.0959`

这说明：

- 单窗口并不是稳定判别单元
- trial 聚合在大量相关窗口上起到了“投票补偿”

### 9.2 trial aggregation 是否在掩盖窗口级问题

是，但它不是主问题源。

它的作用更像：

- baseline 的救场器
- 也是增强偏差的放大器

因为在 `seed1`：

- Gate 后 `window_f1` 只比 Step1B 低约 `0.0008`
- 但 `trial_f1` 低约 `0.0084`

说明小的窗口级系统偏移，在多数投票后被放大了。

### 9.3 当前更值得担心的是什么

当前最值得担心的是：

- 长 EEG 被切成大量窗口后，窗口判别语义偏弱
- 再叠加 Step1B 的增强偏移，就更容易在 trial 投票中形成系统偏差

所以这部分当前更像：

- `窗口语义不足 + aggregation 放大偏差`

而不是“aggregation 本身有 bug”。

## 10. 结论汇总

### A. 当前框架有没有整体崩

- 没有整体崩
- 最像中段增强错配
- 最不像表示层或分类器整体失效

### B. 最可能的故障定位

1. 主问题源：`Step1B` 方向库错配
2. 次级问题：`Gate1 / Gate2` 过滤能力不足，尤其 `Gate2` 语义弱
3. 放大器：`seed1` 的高维 z-space
4. 次级背景：长 EEG 窗口语义偏弱，aggregation 在救场同时也会放大偏差

### C. seed1 为什么 baseline 最稳

因为它避开了：

- 高维 z-space 中错误方向带来的训练分布偏移
- Gate 过滤不足导致的残余增强污染

## 11. 下一步最小行动建议

只建议做 3 件诊断导向的小动作：

1. 补一张 5 个已闭环数据集的方向健康度对照表
   - 每方向 `usage / accept / flip / margin`
   - 明确哪些数据集是“方向均匀但质量混杂”

2. 补一张 Gate 前后对照表
   - `accept_rate`
   - `delta intrusion`
   - `delta margin`
   - `delta trial/window f1`
   用来区分“机制改善未转收益”和“机制本身无改善”

3. 补一个 seed1 vs fixed split 的高维风险摘要
   - `feature_dim`
   - `trial-window gap`
   - `aug_ratio`
   - `Gate2 tau_src_y` 常数化现象
   用来支撑“高维是放大器，不是唯一源头”这一结论
