# Forecast Scripts

这一层对应时间序列预测副线。

当前主要入口：

- `run_forecast_probe_ett.py`
- `run_pia_dynamics_probe.py`

这条线和当前分类二阶段解耦，不建议和 `Tensor-CSPNet / ResNet1D + DLCR` 主线混读。
