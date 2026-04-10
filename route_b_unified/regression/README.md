# Route B Regression Probe

更新时间：2026-03-28

这里存放的是**并行验证分支**：

`raw trial -> z-space -> regressor`

当前目的不是替代分类主线，而是验证：

- 当前协方差 / log-Euclidean 几何前端
- 在不经过 `bridge` 和 `raw MiniROCKET` 的前提下
- 是否能直接支撑连续变量回归 baseline

当前文件：

- `representation.py`
  - 回归版几何表示构造
- `regressor.py`
  - 最小回归头：`Ridge / ElasticNet`
- `evaluator.py`
  - 输出 `RMSE / MAE / R2`

说明：

- 当前主线仍然是分类
- 这里只有回归探针，不包含 `PIA augmentation in z-space`
