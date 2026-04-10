# Unified Layer Refactor Note

更新时间：2026-03-27

当前 Route B 的正确层次，不应再写成：

`representation -> unified policy -> bridge -> evaluator`

而应收紧为：

`representation -> PIA Core -> scheduler/controller -> bridge -> raw evaluator`

更严格地说，在执行路径上应理解为：

`representation -> PIA Core(template learner) -> scheduler -> PIA operator -> bridge -> raw evaluator`

各层职责：

- `representation`
  - 负责 raw -> z-space / 几何表示
- `PIA Core`
  - 负责模板学习与模板响应
- `scheduler/controller`
  - 只负责外层调度
  - 不负责模板学习本体
- `bridge`
  - 负责 raw 注回与 fidelity 诊断
- `raw evaluator`
  - 负责最终 raw-side 验证

当前工程判断：

- `policy/controller` 不是方法学核心
- `PIA/TELM2` 才是方法学核心
- 后续任何统一化都不得再吞掉 `PIA Core`
