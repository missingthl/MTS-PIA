# 轨迹感知增强算子 T2a Promote

更新时间：2026-03-29

## 零、战略定位与当前阶段意义

当前 `T0` 已正式证明：

- `dynamic manifold > static manifold`
- `GRU > mean-pool`
- `NATOPS` 上动态流形正式优于当前 `raw + MiniROCKET` 参考
- `SelfRegulationSCP1` 上动态流形显著优于静态流形

因此，当前分类主线的新问题已经不是：

- 静态 SPD 点表示是否成立
- 静态点 operator 还能否继续微调

而是：

**当样本对象从单点升级为轨迹之后，增强算子应该如何作用于轨迹对象。**

本 promote 的目标不是立刻实现最终大系统，而是把 `T2` 收成第一版可执行入口：

- 先做 `T2a: Global Trajectory Operator`
- 先证明“轨迹对象上的最小几何增强算子”是否成立
- 暂不进入：
  - `T2b: Time-aware Local Operator`
  - 双流融合
  - trajectory feedback / re-basis
  - trajectory -> bridge -> raw 主链

一句话说：

> 当前要让 PIA 从“静态点拉伸器”升级为“轨迹几何变形器”，而第一版入口就是 `T2a`。

---

## 一、任务目标

当前 `T2a` 只回答一个问题：

**在当前已成立的 trajectory manifold branch 上，一个全局一致、带最小连续性约束的轨迹增强算子，是否比无增强 trajectory baseline 更有价值。**

本轮不是：

- 双流融合
- trajectory augmentation zoo
- trajectory feedback pool
- trajectory re-basis
- trajectory risk geometry feedback
- DTW / Fréchet 主系统
- 复杂局部 controller / scheduler

本轮只做：

1. 在 pooled train windows 上学习全局 trajectory basis
2. 对整条 `z_seq` 施加统一 operator
3. 用最小时间平滑约束保证轨迹连续性
4. 用 `orig_z_seq + aug_z_seq` 构造训练增强集
5. 在 `dynamic_gru` 终端上验证 operator 是否带来增益

---

## 二、当前固定前提

1. 当前 `T0` 路径已经成立，轨迹顺序信息本身有价值。  
2. 当前第一版不应直接进入 `T2b`，因为还需要先验证：
   - “全局作用版轨迹算子”是否已经成立。  
3. 当前不应把 bridge、双流、feedback/re-basis 混入 `T2a`，否则无法清晰归因。  
4. 当前最合理的第一版对象是：
   - **trajectory baseline**
   - **trajectory + global trajectory operator**
5. 当前第一版的 operator 本质上应诚实理解为：
   - **shared-basis trajectory operator with temporal smoothing**
   - 不是最终形态的局部动力学算子

---

## 三、任务边界

允许做：

- 复用当前 `trajectory_representation.py`
- 复用当前 `trajectory_dataset.py`
- 复用当前 `trajectory_evaluator.py`
- 复用 `TELM2` 模板学习思想
- 在 pooled train windows 上学习 trajectory operator basis
- 对 `z_seq` 定义全局一致 operator
- 对 operator 输出施加最小时间平滑
- 在 `dynamic_gru` 终端上比较 baseline 与 operator

不允许做：

- 修改 `TELM2` 数学公式
- 修改当前静态 `PIA Core` 主体
- 引入 bridge / raw 回流
- 引入 trajectory feedback pool
- 引入 trajectory re-basis
- 引入 trajectory risk geometry feedback
- 引入双流融合
- 引入 `DTW / Fréchet` 主分类器
- 做多 operator zoo
- 做复杂 controller / scheduler
- 同时做 `T2a + T2b`

硬约束：

- `TELM2.fit()` 只允许在 `train windows` 上进行
- operator 只允许对 `train z_seq` 生成增强
- `val/test` 永远保持原始 trajectory 表示
- 不允许任何 `val/test` 信息进入 basis 学习或增强生成

---

## 四、本轮固定数据集

只做：

- `NATOPS`
- `SelfRegulationSCP1`

---

## 五、当前表示与终端固定

### 1. 表示固定

必须直接复用当前 `T0` 已成立的 trajectory 表示：

- `route_b_unified/trajectory_representation.py`

当前窗口策略保持不变，不在 `T2a` 第一版中重新搜索。

### 2. 终端固定

第一版分类终端固定为：

- `dynamic_gru`

原因：

- `T0` 已正式证明 `GRU > mean-pool`
- `T2a` 第一版不再比较分类头
- 当前主问题是 operator 是否有价值，而不是 head 谁更强

### 3. 当前不再主比较

下列对象当前不是 `T2a` 主比较对象：

- `static_linear`
- `dynamic_meanpool`
- `raw + MiniROCKET`

它们仍可作为背景参考，但不进入本轮核心对照。

---

## 六、T2a 第一版的最小数学对象

### 1. basis 学习对象

不要在 whole-trial 单点 `z_static` 上学习 operator basis。  
第一版必须在 **pooled train windows** 上学习。

即：

- 将所有训练 trial 的所有窗口表示 `z_t` 展开成 pooled set
- 记为：
  - `Z_train_win = {z_t^(i)}`

这些 pooled windows 作为第一版 trajectory operator 的 basis 学习对象。

这不是对 trajectory operator 的唯一数学定义，而是当前第一版为了最小化复杂度所采用的**设计选择**：

- 第一版采用全局共享 basis
- 暂不做按时间段学 basis
- 暂不做按类学 basis
- 暂不做按轨迹局部学 basis

### 2. basis 学习方式

第一版优先复用现有静态 PIA 的基本思想：

- 用 `TELM2` 在 `Z_train_win` 上学习模板矩阵 `W`
- 再进行最小的几何化处理

第一版硬约束：

- `r_dimension = 1`
- 即：**第一版只允许单轴 trajectory operator**

原因：

- 当前先回答“轨迹 operator 是否成立”
- 不同时引入多轴身份问题

### 3. operator 作用对象

对每个 trial 的轨迹：

- `z_seq = [z_1, z_2, ..., z_K]`

第一版 operator 先作用于每个时间步：

\[
\delta_t = \gamma \cdot \mathrm{comp}(z_t)
\]

其中：

- `comp(z_t)` 是 `z_t` 在唯一模板轴上的投影分量
- `gamma` 为固定全局强度

### 4. 最小时间连续性约束

第一版必须加入最小平滑层，避免窗口间几何撕裂：

\[
\tilde{\delta}_t = (1-\lambda)\delta_t + \frac{\lambda}{2}(\delta_{t-1} + \delta_{t+1})
\]

边界位置必须写死最小合理近似，避免代码实现时出现越界或口径不一致：

- 对于轨迹起始帧 `t = 0` 与终止帧 `t = K-1`
- 统一采用 **edge padding / 边缘复制**
- 即令：
  - `delta_{-1} = delta_0`
  - `delta_K = delta_{K-1}`

这样第一版的平滑实现可以保持：

- 可复现
- 无数组越界风险
- 不额外引入更复杂的边界插值策略

最终增强：

\[
z_t^{aug} = z_t + \tilde{\delta}_t
\]

当前第一版的真实含义必须写清：

> T2a 第一版本质上是“共享全局 basis 的轨迹增强 + 最小时间连续性约束”，不是完整的局部动力学增强器。

### 5. 当前固定默认值

第一版默认值写死为：

- `axis_count = 1`
- `gamma_main = 0.10`
- `smooth_lambda = 0.50`

并做一个最小 ablation：

- `smooth_lambda = 0.00`
- `smooth_lambda = 0.50`

也就是说，第一版主问题之一是：

**最小平滑约束本身是否有价值。**

同时加入一个最小烟雾测试修正规则：

- 默认先用 `gamma_main = 0.10`
- 若 smoke 明显过弱或过强，允许只追加一个最小修正值：
  - `0.05` 或 `0.15`
- 不允许扩成 `gamma` 网格搜索

也就是说：

- `gamma_main = 0.10` 是默认值
- 不是不可动摇的唯一值

---

## 七、建议新增模块

建议新增：

- `route_b_unified/trajectory_pia_operator.py`
- `route_b_unified/trajectory_pia_evaluator.py`
- `scripts/run_route_b_trajectory_pia_t2a.py`

### 1. `trajectory_pia_operator.py`

职责：

- 在 pooled train windows 上学习单轴 trajectory basis
- 对 `z_seq` 施加全局一致 operator
- 输出：
  - `z_seq_aug`
  - `delta_seq`
  - `operator_meta`

建议接口：

- `fit(train_split)`
- `transform(z_seq)`
- `fit_transform(train_split)`

### 2. `trajectory_pia_evaluator.py`

职责：

- 接收 `trajectory baseline`
- 生成 `trajectory + operator`
- 用 `dynamic_gru` 训练并比较
- 输出分类结果与结构诊断

### 3. `run_route_b_trajectory_pia_t2a.py`

职责：

- 组织 `NATOPS + SCP1`
- 组织 `3 seeds`
- 输出正式小矩阵结果

---

## 八、第一轮实验设计

### 主比较对象

1. `trajectory baseline (dynamic_gru, no operator)`
2. `trajectory + operator train-aug (gamma=0.10, smooth_lambda=0.00)`
3. `trajectory + operator train-aug (gamma=0.10, smooth_lambda=0.50)`

这里必须明确：

- augmentation 的第一版训练口径是：
  - `train_final = orig_z_seq_train + aug_z_seq_train`
- 不是：
  - 用 `aug_z_seq_train` 替换原始训练序列

原因：

- 当前要验证的是 augmentation 的价值
- 不是“变换后的新表示单独是否更好”

同时写死：

- `val/test` 只使用原始 `z_seq`
- 不对 `val/test` 做 operator 变换

### 可选保留字段

可在 summary 中附上背景参考字段：

- `T0 dynamic_gru formal`
- `raw + MiniROCKET formal`

但它们不应成为本轮主比较对象。

### 当前不做

- `gamma` 大网格搜索
- 多轴
- 多窗口策略
- 多终端头

第一版要求：

- 小矩阵
- 低维
- 强归因

---

## 九、当前必须回答的问题

1. `trajectory + operator` 是否优于 `trajectory baseline`
2. `smooth_lambda = 0.50` 是否优于 `smooth_lambda = 0.00`
3. `SCP1` 是否比 `NATOPS` 更受益于 trajectory-aware enhancement
4. trajectory operator 的收益是否来自：
   - 更好的轨迹几何
   - 而不只是训练噪声
5. 当前是否值得进入下一阶段 `T2b`
6. 当前相对：
   - `T0 dynamic_gru formal`
   - `raw + MiniROCKET formal`
   的位置大致在哪里

---

## 十、结构诊断要求

除了分类指标，必须输出结构诊断。

至少输出：

- `step_change_mean`
- `local_curvature_proxy`
- `classwise_dispersion`
- `transition_separation_proxy`
- `continuity_distortion_ratio`

第一轮这些 proxy 允许采用最小、简单、可复现定义，不要求理论最优。  
重点是保证：

- `baseline`
- `unsmoothed operator`
- `smoothed operator`

三者之间口径一致。

### `continuity_distortion_ratio` 建议第一版定义

可定义为：

\[
\frac{\text{mean step-change of augmented trajectory}}{\text{mean step-change of original trajectory} + \epsilon}
\]

用于衡量增强是否显著撕裂轨迹连续性。

---

## 十一、输出文件要求

必须输出：

1. `trajectory_pia_t2a_config_table.csv`
2. `trajectory_pia_t2a_per_seed.csv`
3. `trajectory_pia_t2a_dataset_summary.csv`
4. `trajectory_pia_t2a_diagnostics_summary.csv`
5. `trajectory_pia_t2a_conclusion.md`

建议字段：

### `trajectory_pia_t2a_per_seed.csv`

- `dataset`
- `seed`
- `operator_mode`
- `axis_count`
- `gamma_main`
- `smooth_lambda`
- `test_macro_f1`
- `delta_vs_dynamic_baseline`
- `t0_dynamic_gru_reference`
- `raw_minirocket_reference`

### `trajectory_pia_t2a_dataset_summary.csv`

- `dataset`
- `baseline_macro_f1_mean/std`
- `operator_unsmoothed_macro_f1_mean/std`
- `operator_smoothed_macro_f1_mean/std`
- `t0_dynamic_gru_macro_f1_mean/std`
- `raw_minirocket_macro_f1_mean/std`
- `best_mode`

### `trajectory_pia_t2a_diagnostics_summary.csv`

- `dataset`
- `seed`
- `operator_mode`
- `step_change_mean`
- `local_curvature_proxy`
- `classwise_dispersion`
- `transition_separation_proxy`
- `continuity_distortion_ratio`

### `trajectory_pia_t2a_conclusion.md`

必须明确回答：

- `trajectory operator` 是否成立
- 平滑是否必要
- `SCP1` 是否更受益于 trajectory-aware enhancement
- 是否值得进入 `T2b`
- 当前结果相对：
  - `T0 dynamic_gru`
  - `raw + MiniROCKET`
 处在什么位置

---

## 十二、成功标准

第一版不要求一上来全面超过所有 raw 强基线。  
满足至少两条即可视为 `T2a` 成立：

1. `trajectory + operator > trajectory baseline` 至少在一个主数据集上成立
2. `SCP1` 上出现比 baseline 更明确的提升
3. 平滑版优于不平滑版
4. 结构诊断显示：
   - `transition_separation` 改善
   - 或 `classwise_dispersion` 更合理
   - 且 `continuity_distortion_ratio` 没有明显恶化

失败标准：

- operator 不优于 baseline
- 平滑无意义
- 增强明显撕裂轨迹连续性
- 无法区分“更好几何”与“随机训练波动”

---

## 十三、当前明确不做的事

第一版明确不扩成：

- 双流融合主系统
- trajectory feedback pool
- trajectory re-basis
- trajectory risk geometry feedback
- DTW / Fréchet 大系统
- 多 operator zoo
- 大规模窗口策略搜索
- 复杂 controller / scheduler
- `T2b` 局部时间感知 operator

---

## 十四、建议执行顺序

### Step 1

复用当前 `trajectory_representation.py`，固定 T0 默认窗口策略。

### Step 2

实现 `trajectory_pia_operator.py`：

- pooled train windows 上 `TELM2.fit`
- `r_dimension = 1`
- 输出单轴 trajectory basis

### Step 3

实现最小 trajectory operator：

- `gamma_main = 0.10`
- `smooth_lambda ∈ {0.00, 0.50}`

### Step 4

实现 `trajectory_pia_evaluator.py` 与 runner：

- `dynamic_gru`
- `NATOPS + SCP1`
- `3 seeds`

### Step 5

输出正式结果与结论文档，明确判断：

- operator 是否成立
- 平滑是否必要
- 是否值得进入 `T2b`

---

## 十五、一句话执行目标

**在已成立的 trajectory manifold branch 上，先落一个最小的全局轨迹作用版 PIA operator：用 pooled train windows 学单轴全局 basis，对整条 `z_seq` 施加统一但连续性受约束的几何增强，并在 `dynamic_gru` 终端上验证“轨迹几何变形器”是否比无增强 trajectory baseline 更有价值。**
