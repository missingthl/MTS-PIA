# 动态流形分类新路径 T0 Promote

更新时间：2026-03-29

## 零、全局战略定位与本轮架构意义

当前静态 SPD 单点主线进入**阶段性冻结**。

本 promote 不是对旧线的局部补丁，也不是一个普通新分类头实验，而是**新表示路径的第一阶段 T0**。

当前要验证的不是：

- 静态点流形还能怎么继续修补
- operator 还能不能再调细

而是：

**把样本对象从 `z_static`（单点 SPD/log-Euclidean 表示）升级为 `z_seq`（窗口化轨迹流形表示）之后，是否能承载更多任务相关结构，并在分类任务上显示出更强或至少更有信息量的潜力。**

当前不预设最终一定是：

- 双流融合架构
- 去除 bridge 的架构
- 保留 bridge 的架构

本轮只做最小表示验证，并要求工程实现具备后续三条路径的兼容性：

1. 单流 trajectory classifier
2. trajectory + raw 时序双流融合
3. trajectory-aware operator 后回接 bridge / raw 空间

---

## 一、任务目标

当前 T0 只做一件事：

**动态流形表示的最小分类验证。**

本轮不是：

- trajectory augmentation
- feedback pool
- M0+ re-basis
- risk-aware controller
- dual-stream 大系统
- bridge-based trajectory 主流程
- DTW / Fréchet 主分类器

本轮只回答：

1. `z_seq` 是否比 `z_static` 更有信息量
2. `z_seq` 在最小分类头下是否比静态流形更适合分类
3. 是否值得进入下一阶段：
   - trajectory-aware operator / augmentation
   - dual-stream fusion
   - 或 trajectory -> bridge -> raw 路线

---

## 二、当前固定前提

1. 当前静态样本对象是：
   - `x_trial -> Σ -> log/tangent -> z_static`
2. 当前静态主线越来越像受到表示瓶颈约束
3. 当前 T0 的重点不是继续优化静态 operator，而是验证“样本对象重定义”是否成立

---

## 三、任务边界

允许做：

- 新增窗口化轨迹流形表示模块
- 对 raw trial 做滑动窗口
- 每窗计算 SPD/log-Euclidean 表示
- 构造 `z_seq`
- 新增最小动态分类头
- 输出分类结果和结构诊断结果

不允许做：

- 修改 TELM2 公式
- 修改当前 PIA Core 主体
- 引入 trajectory augmentation
- 引入 feedback pool / re-basis
- 引入复杂双流系统
- 引入 bridge-based trajectory 主流程
- 引入 DTW / Fréchet 作为第一轮主分类器
- 开模型 zoo
- 扩更多数据集

---

## 四、本轮固定数据集

只做：

- `NATOPS`
- `SelfRegulationSCP1`

---

## 五、样本对象重定义

### 当前静态对象

`x_trial -> Σ -> log/tangent -> z_static ∈ R^D`

### 本轮动态对象

`x_trial -> sliding windows -> {Σ_1, Σ_2, ..., Σ_K} -> {z_1, z_2, ..., z_K}`

最终一个 trial 表示为：

- `z_seq ∈ R^{K × D}`

要求：

- 保留窗口顺序
- 第一轮固定统一窗口策略
- 不做复杂多尺度
- 输出必须具备后续兼容性

硬性约束：

- 滑动窗口长度 `window_len` 必须严格大于特征通道数 `C`
- 必须在代码中加入显式断言：`assert window_len > C`
- 目的不是形式上“保证绝对满秩”，而是避免局部协方差矩阵落入明显奇异或近奇异区，从而降低后续 Log-Euclidean 映射失稳风险
- 如默认窗口策略不满足该条件，必须先调整窗口策略，不允许带病进入轨迹表示阶段

第一轮默认窗口策略建议写死为：

- `window_len = max(C + 4, round(0.20 * T_med_train))`
- `hop_len = max(4, round(0.10 * T_med_train))`
- 并额外保证：
  - `window_len > C`
  - `hop_len < window_len`

---

## 六、架构兼容性要求

本轮虽然只做最小分类验证，但表示层设计必须为后续三条路径留接口：

1. **单流路径**
   - `z_seq -> dynamic manifold classifier`

2. **双流路径**
   - `z_seq + raw temporal stream -> fusion`

3. **回桥路径**
   - `trajectory operator / trajectory augmentation -> bridge back to raw`

因此 `trajectory_representation` 的输出至少必须包含：

- `z_seq`
- `log_matrix_seq`
- `window_meta`
  - `trial_id`
  - `window_index`
  - `start`
  - `end`

这是硬约束。

---

## 七、建议新增模块

建议新增：

- `route_b_unified/trajectory_representation.py`
- `route_b_unified/trajectory_dataset.py`
- `route_b_unified/trajectory_classifier.py`
- `route_b_unified/trajectory_evaluator.py`
- `scripts/run_route_b_dynamic_manifold_classification.py`

职责：

### 1. `trajectory_representation.py`

- 输入 `x_trial`
- 输出：
  - `z_seq`
  - `log_matrix_seq`
  - `window_meta`

### 2. `trajectory_dataset.py`

- 对齐：
  - `trial_id`
  - `label`
  - `z_seq`
  - `z_static`

### 3. `trajectory_classifier.py`

第一轮只允许两个动态头：

- `mean-pool + linear`
- `GRU`

同时必须包含一个静态对照头：

- `static linear`

硬约束：

- `static manifold-only` 必须使用与 `dynamic mean-pool + linear` 尽量等价的最小头口径
- 默认写死为：
  - `z_static -> linear classifier`
- 本轮禁止给 `static manifold-only` 单独加更深的 MLP 头
- 目的：保证主比较聚焦于“样本对象是单点还是序列”，而不是头部复杂度差异

### 4. `trajectory_evaluator.py`

输出：

- `acc`
- `macro-F1`

### 5. `run_route_b_dynamic_manifold_classification.py`

负责：

- `NATOPS + SCP1`
- `3 seeds`
- 小规模正式实验

---

## 八、比较对象分层

### 主比较对象

1. `static manifold-only`
2. `dynamic manifold mean-pool`
3. `dynamic manifold GRU`

### 参考外部强基线

4. `raw + MiniROCKET`

说明：

- 本轮主比较是：
  - `static point manifold` vs `dynamic trajectory manifold`
- `raw + MiniROCKET` 是参考强基线
- 不是本轮内部主逻辑

---

## 九、第一轮不要做的事

第一轮明确禁止：

- dual-stream fusion
- trajectory augmentation
- feedback pool
- re-basis
- risk-aware controller
- bridge-based trajectory 主流程
- DTW / Fréchet 主分类器
- 多 backbone 并行比较
- 大矩阵窗口策略搜索

---

## 十、结构诊断要求

这条新路径不是纯 benchmark。

除了 `acc / macro-F1`，必须同时输出结构诊断：

- `dynamic_manifold_diagnostics_summary.csv`

必须回答：

1. trajectory 是否出现了 static point 没有的状态跃迁/局部高变化模式
2. `NATOPS / SCP1` 哪个更像真正受益于轨迹化
3. `mean-pool` 与 `GRU` 的差异是否说明：
   - 是“顺序信息有用”
   - 而不只是“窗口更多”

建议至少输出：

- `trajectory_len_mean`
- `step_change_mean`
- `local_curvature_proxy`
- `classwise_dispersion`
- `transition_separation_proxy`

硬约束：

- 第一轮这些 proxy 允许采用最小、简单、可复现定义，不要求理论最优
- 当前重点是保证 `static vs trajectory` 比较口径一致，而不是追求最复杂或最漂亮的几何指标

第一轮默认近似定义建议：

- `local_curvature_proxy`
  - `mean(||z_{t+1} - 2z_t + z_{t-1}||_2)`
- `transition_separation_proxy`
  - 各类平均一步差分向量之间的均值距离

---

## 十一、必须输出的文件

1. `dynamic_manifold_config_table.csv`
2. `dynamic_manifold_per_seed.csv`
3. `dynamic_manifold_dataset_summary.csv`
4. `dynamic_manifold_diagnostics_summary.csv`
5. `dynamic_manifold_conclusion.md`

建议字段：

### `dynamic_manifold_per_seed.csv`

- `dataset`
- `seed`
- `model_type`
- `window_len`
- `hop_len`
- `test_acc`
- `test_macro_f1`

### `dynamic_manifold_dataset_summary.csv`

- `dataset`
- `static_manifold_macro_f1_mean/std`
- `dynamic_meanpool_macro_f1_mean/std`
- `dynamic_gru_macro_f1_mean/std`
- `raw_minirocket_macro_f1_mean/std`
- `best_model`

### `dynamic_manifold_diagnostics_summary.csv`

- `dataset`
- `model_type`
- `trajectory_len_mean`
- `step_change_mean`
- `local_curvature_proxy`
- `classwise_dispersion`
- `transition_separation_proxy`
- `notes`

### `dynamic_manifold_conclusion.md`

必须明确回答：

- 动态流形是否优于静态流形
- `GRU` 是否优于 `mean-pool`
- `SCP1` 是否更受益于轨迹化
- 是否值得进入下一阶段：
  - trajectory-aware operator / augmentation
  - dual-stream fusion
  - 或 trajectory -> bridge -> raw 路线

---

## 十二、本轮必须回答的问题

1. 动态流形是否优于静态单点流形分类
2. `GRU` 是否优于 `mean-pool`
3. `SCP1` 是否比静态流形更受益
4. `NATOPS` 上动态流形是否至少不明显退化
5. trajectory 表示是否真的承载了 static point 没有的任务结构
6. 当前下一阶段更值得推进的是：
   - 单流 trajectory classifier
   - 双流融合
   - 还是 trajectory -> bridge -> raw

---

## 十三、成功标准

满足至少两条即可认为 T0 成立：

- `dynamic manifold` 在至少一个主数据集上明确优于 `static manifold-only`
- `GRU > mean-pool`
- `SCP1` 上出现比静态流形更稳定的正向信号
- 结构诊断显示 trajectory 确实携带 static point 没有的状态变化信息

失败标准：

- 动态流形不优于静态流形
- `GRU` 不优于 `mean-pool`
- 结构诊断无法证明轨迹表示带来了新增信息

---

## 十四、建议执行顺序

1. 固定窗口化规则
2. 实现 `z_seq` 提取
3. 先跑 `static linear`
4. 再跑 `dynamic mean-pool`
5. 再跑 `dynamic GRU`
6. 与 `static manifold-only` 比较
7. 最后附上 `raw + MiniROCKET` 参考
8. 输出结构诊断与正式结论

---

## 十五、一句话执行目标

**冻结静态 SPD 单点主线的进一步补丁，启动新表示路径的 T0：把 trial 从“一个 SPD 点”升级为“一个 log-Euclidean 轨迹序列”，先用 `static linear / dynamic mean-pool / dynamic GRU` 的最小比较验证动态流形是否优于静态流形；同时保留它未来接入双流融合或重新回接 bridge/raw 空间的工程兼容性。**
