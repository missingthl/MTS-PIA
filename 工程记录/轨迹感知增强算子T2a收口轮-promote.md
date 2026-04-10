# 轨迹感知增强算子 T2a 收口轮 Promote

更新时间：2026-03-29

## 零、战略定位与当前收口目标

当前 `T2a` 首轮 formal 已经给出清晰结论：

- `NATOPS` 上，`trajectory operator` 已明确成立
- `SelfRegulationSCP1` 上，仅呈现边际正信号，尚未正式站稳
- 当前 `T2a` 的主命题已经不再是：
  - “轨迹 operator 是否存在”
- 而是：
  - **“为什么 `SCP1` 没有把这条线正式站稳”**

因此，这一轮 promote 的目标不是继续扩张 `T2a`，也不是直接进入 `T2b`，而是做一次**解释性收口轮**：

- 固定 `T2a` 的其余一切
- 只围绕两个旋钮做很窄的小矩阵：
  - `gamma_main`
  - `smooth_lambda`
- 以 `SCP1` 为主
- `NATOPS` 只保留少量锚点，防止把已成立的那条线打坏

一句话说：

> 当前收口轮的任务不是“再证明 T2a 成立”，而是判清 `SCP1` 的限制因素究竟主要来自增强强度，还是来自连续性约束。

---

## 一、任务目标

当前收口轮只回答三个问题：

1. `SCP1` 的限制因素主要来自：
   - `gamma_main`
   - 还是 `smooth_lambda`
2. 是否存在一个：
   - `SCP1` 更优
   - 同时不明显损伤 `NATOPS`
   的 `T2a` 收口点
3. 如果没有，是否足以判断：
   - 当前问题已经超出 `T2a` 的“共享全局 basis + 全局平滑”范式
   - 因而值得进入 `T2b`

本轮不是：

- 再证明 trajectory manifold branch 是否成立
- 多轴 trajectory operator
- trajectory feedback / re-basis
- 双流融合
- 大规模窗口策略搜索
- 大规模超参搜索

---

## 二、当前固定前提

以下内容在本轮中全部冻结：

- 当前窗口策略
- 当前 `trajectory_representation.py`
- 当前 pooled-train-windows 学 basis 的方法
- `axis_count = 1`
- 当前 `dynamic_gru` 终端
- 训练轮数 / 优化器 / batch / 学习率等训练超参
- 训练构造方式：
  - `train_final = orig_z_seq_train + aug_z_seq_train`
- `val/test` 永远保持原始 trajectory 表示

也就是说：

> 本轮不再讨论新的 operator 结构，只讨论当前 `T2a` 结构下两个旋钮的敏感性。

---

## 三、任务边界

允许做：

- 复用当前 `trajectory_pia_operator.py`
- 复用当前 `trajectory_pia_evaluator.py`
- 固定 basis 学习逻辑
- 扫描：
  - `gamma_main`
  - `smooth_lambda`
- 用当前 `T2a` 的结构诊断口径做收口判断

不允许做：

- 改 basis 学习对象
- 改 `TELM2` 数学公式
- 改 `axis_count`
- 改终端分类器
- 改窗口策略
- 引入 bridge / raw 回流
- 引入 feedback / re-basis
- 引入双流
- 引入 `T2b` 局部时间感知 operator

硬约束：

- 对于每个 `dataset × seed`，basis 只允许 `fit` 一次，并在整轮 sweep 中固定复用
- 不允许因为不同 `gamma/smooth` 配置而重新学习 basis
- 本轮要测的是：
  - **固定 basis 下的 operator sensitivity**
  - 不是 basis 重学波动
- 本轮 sweep 比较的是 **fixed-basis 下的 operator sensitivity**，而不是 basis 重学后的联合变化；因此任何配置变化都不允许触发 basis 重拟合

---

## 四、数据集与比较权重

### 1. `SCP1` 为主矩阵

当前主矩阵只围绕：

- `SelfRegulationSCP1`

展开。

### 2. `NATOPS` 为锚点

`NATOPS` 不再做全矩阵，只保留少量锚点配置用于 sanity check：

1. `trajectory baseline`
2. 当前 `T2a formal` 最优点
3. `SCP1` 收口后得到的候选点

目的只有一个：

> 确认为 `SCP1` 找到的收口点，没有把已经成立的 `NATOPS` 线明显打坏。

---

## 五、实验设计原则

### 1. 只动两个参数

本轮唯一可变参数：

- `gamma_main`
- `smooth_lambda`

其余一切固定。

### 2. 先两阶段，不一次性全铺二维网格

最稳的执行顺序是：

- `Phase A：单因素定位`
- `Phase B：小联动收口`

原因：

- 先判断主矛盾在哪个旋钮上
- 再决定是否需要二维小网格
- 避免一上来把搜索做脏

---

## 六、Phase A：单因素定位

Phase A 以当前 `T2a formal` 第一版默认点为中心，不再追溯其他旧点。  
在当前实现口径下，这个默认中心点固定为：

- `gamma_main = 0.10`
- `smooth_lambda = 0.50`

### A1. 固定平滑，只扫 gamma

固定：

- `smooth_lambda = 0.50`

扫描：

- `gamma_main ∈ {0.05, 0.08, 0.10, 0.12, 0.15}`

目的：

- 判断 `SCP1` 是否对增强强度过敏

### A2. 固定 gamma，只扫平滑

固定：

- `gamma_main = 0.10`

扫描：

- `smooth_lambda ∈ {0.00, 0.25, 0.50, 0.75}`

目的：

- 判断 `SCP1` 是否被平滑过度抹平
- 或是否反而需要更强平滑

---

## 七、Phase A 的判断逻辑

看哪一条曲线更敏感：

1. 如果改 `gamma_main` 变化大、改 `smooth_lambda` 变化小  
   -> 主矛盾在增强强度

2. 如果改 `smooth_lambda` 变化大、改 `gamma_main` 变化小  
   -> 主矛盾在连续性约束

3. 如果两边都敏感  
   -> 再进入 `Phase B`

硬约束：

- 只有当某个旋钮在 `3 seeds` 上表现出更大的稳定敏感性，或两者都明显敏感时，才进入 `Phase B`
- 如果差异只落在噪声范围里，不允许硬开二维网格

---

## 八、Phase B：小联动收口

只有在 `Phase A` 结束后才允许进入。

### 情况 1：`gamma_main` 更敏感

则只围绕较优 gamma 附近，配 2 到 3 档平滑：

- `gamma_main ∈ {0.06, 0.08, 0.10}`
- `smooth_lambda ∈ {0.25, 0.50}`

### 情况 2：`smooth_lambda` 更敏感

则只围绕较优 smooth 附近，配 2 到 3 档 gamma：

- `gamma_main ∈ {0.08, 0.10}`
- `smooth_lambda ∈ {0.00, 0.25, 0.50}`

### 情况 3：两者都敏感

则做一个最小 `3 × 3` 网格：

- `gamma_main ∈ {0.06, 0.08, 0.10}`
- `smooth_lambda ∈ {0.00, 0.25, 0.50}`

硬约束：

- `Phase B` 不允许超过这个规模
- 当前目标不是全局最优搜索，而是**解释限制因素**

---

## 九、NATOPS 锚点执行顺序

`NATOPS` 锚点不应和 `SCP1 Phase A` 同时跑。

最稳的执行顺序是：

1. 先完成 `SCP1 Phase A`
2. 必要时完成 `SCP1 Phase B`
3. 在 `SCP1` 确认候选点后，再跑 `NATOPS` 的 3 个锚点

也就是：

- `baseline`
- 当前 `T2a formal` 最优点
- `SCP1` 收口候选点

硬约束：

- 若 `SCP1` 收口后存在多个近似候选点，只保留 **一个最稳候选点** 进入 `NATOPS` 锚点校验
- 不允许把 `NATOPS` 锚点校验扩成多候选并行对照

这样做的好处：

- 省计算
- 不把锚点校验提前混入主判断

---

## 十、需要重点观察的指标

本轮不能只看 `macro-F1`，必须分两层看。

### 1. 终点指标

至少记录：

- `test_macro_f1`
- `delta_vs_t2a_baseline`
- 对于 `NATOPS` 锚点：
  - `delta_vs_best_t2a`

### 2. 轨迹结构指标

沿用当前 `T2a` 已有诊断口径，重点观察：

- `step_change_mean`
- `local_curvature_proxy`
- `transition_separation_proxy`
- `continuity_distortion_ratio`

硬约束：

- 结构诊断指标用于解释 `SCP1` 的失败模式与限制因素
- 不单独作为候选点录取依据
- 候选点仍以：
  - 终点表现
  - 跨 seed 稳定性
  为主

---

## 十一、当前判读逻辑

### 情况 A：降低 gamma 后更好

如果：

- `SCP1` 分数上升
- `continuity_distortion_ratio` 下降
- `transition_separation_proxy` 不明显恶化

则说明：

> `SCP1` 主要是被增强强度打伤。

### 情况 B：减弱平滑后更好

如果：

- `SCP1` 分数上升
- `transition_separation_proxy` 上升
- `continuity_distortion_ratio` 没明显爆炸

则说明：

> `SCP1` 主要是被过强平滑抹掉了局部状态跃迁。

### 情况 C：增强平滑后更好

如果：

- 更强的 `smooth_lambda` 让 `SCP1` 更好

则说明：

> `SCP1` 的问题主要是轨迹增强撕裂，需要更强连续性约束。

---

## 十二、最小实验矩阵建议

### 1. `SCP1` 主矩阵

`Phase A`：

- gamma sweep：`5` 点
- smooth sweep：`4` 点

如果需要进入 `Phase B`：

- 最多再加 `6` 点

也就是：

- `9 ~ 15` 个配置
- `× 3 seeds`

### 2. `NATOPS` 锚点

- `3` 个配置
- `× 3 seeds`

整体工作量是可控的，不是大矩阵。

---

## 十三、输出文件要求

必须输出：

1. `trajectory_pia_t2a_closure_config_table.csv`
2. `trajectory_pia_t2a_closure_per_seed.csv`
3. `trajectory_pia_t2a_closure_dataset_summary.csv`
4. `trajectory_pia_t2a_closure_diagnostics_summary.csv`
5. `trajectory_pia_t2a_closure_conclusion.md`

建议字段：

### `trajectory_pia_t2a_closure_per_seed.csv`

- `dataset`
- `phase`
- `config_id`
- `seed`
- `gamma_main`
- `smooth_lambda`
- `test_macro_f1`
- `delta_vs_t2a_baseline`
- `delta_vs_best_t2a`

说明：

- `delta_vs_t2a_baseline` 中的 `baseline` 固定指：
  - `dynamic_gru, no operator`
- 不允许把该字段解释成：
  - `current best T2a`
  - 或 `SCP1` 当前最优点

### `trajectory_pia_t2a_closure_dataset_summary.csv`

- `dataset`
- `phase`
- `config_id`
- `macro_f1_mean/std`
- `delta_vs_t2a_baseline_mean`
- `delta_vs_best_t2a_mean`
- `is_candidate`

说明：

- `delta_vs_t2a_baseline_mean` 的 `baseline` 仍固定指：
  - `dynamic_gru, no operator`

### `trajectory_pia_t2a_closure_diagnostics_summary.csv`

- `dataset`
- `phase`
- `config_id`
- `step_change_mean`
- `local_curvature_proxy`
- `transition_separation_proxy`
- `continuity_distortion_ratio`

### `trajectory_pia_t2a_closure_conclusion.md`

必须明确回答：

1. `SCP1` 的限制因素主要来自：
   - `gamma_main`
   - 还是 `smooth_lambda`
2. 是否存在一个：
   - `SCP1` 更优
   - 同时不明显损伤 `NATOPS`
   的收口点
3. 如果没有，当前是否值得进入 `T2b`

---

## 十四、成功标准

这轮收口不是要求全面赢。

满足以下任一条，即可视为收口成功：

1. 找到一个稳定优于当前 `T2a baseline` 的 `SCP1` 点
2. 找到一个不明显损伤 `NATOPS` 的 `SCP1` 改善点
3. 明确判清主限制因素主要来自：
   - `gamma_main`
   - 或 `smooth_lambda`

其中第 3 条单独成立也算成功，因为它足以决定是否进入 `T2b`。

---

## 十五、当前明确不做的事

这一轮明确不扩成：

- `T2b`
- 多轴 trajectory operator
- basis 学习方式重构
- trajectory feedback / re-basis
- 双流融合
- 大规模窗口策略搜索
- 大网格超参搜索

---

## 十六、建议执行顺序

### Step 1

固定当前 `T2a` 的 basis 学习方式、终端与训练口径。

### Step 2

对 `SCP1` 执行 `Phase A`：

- gamma sweep
- smooth sweep

### Step 3

根据 `Phase A` 判断是否进入 `Phase B`。

### Step 4

若有候选点，再执行 `NATOPS` 3 个锚点。

### Step 5

输出正式结果与结论，明确判断：

- 主限制因素
- 是否存在收口点
- 是否值得进入 `T2b`

---

## 十七、一句话执行目标

**冻结 `T2a` 的其余一切，只围绕 `gamma_main` 与 `smooth_lambda` 两个旋钮做窄矩阵收口；以 `SCP1` 为主定位当前失败模式，再用少量 `NATOPS` 锚点确认收口点不会明显打坏已成立主线，并据此判断下一步应继续停留在 `T2a` 收口，还是正式进入 `T2b`。**
