# 动态末端统一 T8-pre Promote

更新时间：2026-03-30

## 一、任务定位

这一轮不是双流，不是新表示，也不是新增强。

这一轮只回答一个问题：

> 在不改变当前动态表示与增强主链的前提下，把动态末端从 `GRU` 换成 `MiniROCKET` 后，动态线本身能否明显变强。

---

## 二、当前固定前提

以下内容全部冻结：

- 当前 `trajectory_representation`
- 当前窗口策略
- 当前 train/val/test split
- 当前 `z_seq` 表示构造
- 不做任何 operator / rebasis / feedback
- 不做双流
- 不做 bridge
- 不做 raw 回写

也就是说：

**本轮只改动态末端 classifier。**

---

## 三、主比较对象

1. `static_linear`
2. `dynamic_gru`
3. `dynamic_minirocket`
4. `raw_minirocket`

说明：

- `static_linear` 保留为静态对象基线
- `dynamic_gru` 是当前动态末端基线
- `dynamic_minirocket` 是本轮唯一新增对象
- `raw_minirocket` 是外部参考强基线

---

## 四、硬约束

### 1. dynamic_minirocket 只吃 z_seq

- `dynamic_minirocket` 只允许读取 `z_seq`
- 不允许把 `z_static` 伪装成序列喂 MiniROCKET

### 2. static 不做伪序列 MiniROCKET

- 静态线仍然是 `static_linear`
- 当前不做 `static_minirocket`

### 3. 变长序列处理固定

第一版固定：

- 以每个 `dataset × seed` 的 **train 内最大 trajectory length** 作为目标长度
- 若该长度小于 `9`，则固定抬到 `9`
- 对 `z_seq` 统一做 **edge pad**
- 若 `val/test` 长于 train max len，则按 train max len 截断
- 不搜索 padding 策略

### 4. padding 只发生在 z_seq 表示层

- 不允许回写 raw
- 不允许引入 raw-level stitching

---

## 五、数据集

第一版只做：

- `NATOPS`
- `SelfRegulationSCP1`

---

## 六、输出文件

必须输出：

1. `dynamic_terminal_probe_config_table.csv`
2. `dynamic_terminal_probe_per_seed.csv`
3. `dynamic_terminal_probe_dataset_summary.csv`
4. `dynamic_terminal_probe_padding_summary.csv`
5. `dynamic_terminal_probe_diagnostics_summary.csv`
6. `dynamic_terminal_probe_conclusion.md`

---

## 七、成功标准

### 弱成立

- `dynamic_minirocket > dynamic_gru` 至少在一个主数据集上成立

### 中等成立

- `dynamic_minirocket > dynamic_gru` 在两站都成立
- 或一站明显成立，另一站不明显回退

### 强成立

- `dynamic_minirocket` 明显缩小与 `raw_minirocket` 的差距
- 且足以支持后续 `T8` 双流优先考虑 `MiniROCKET` 作为动态流终端

---

## 八、一句话执行目标

**冻结当前动态表示与增强主链，不上双流、不改容器，只新增一个 `dynamic_z_seq + MiniROCKET` 末端，先判断当前动态线的瓶颈里到底有多少来自终端本身。**
