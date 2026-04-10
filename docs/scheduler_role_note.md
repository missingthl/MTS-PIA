# Scheduler Role Note

更新时间：2026-03-27

当前 `scheduler/controller` 的定位必须降级为：

- 模板选择器
- 轮次推进器
- budget / stop rule 管理器
- outer-loop feedback 承载器

它当前不允许承担：

- 模板学习
- 反激活目标构造
- 岭回归回写
- 偏置更新
- 行正交化

也就是说：

- `PIA Core` 负责学 `W, b`
- `scheduler` 只负责决定“用哪些模板、用多大 gamma、何时停”

`LRAES`、`multiround`、`feedback` 也都应视为 scheduler 的外层部件，而不是 `PIA Core` 本身。
