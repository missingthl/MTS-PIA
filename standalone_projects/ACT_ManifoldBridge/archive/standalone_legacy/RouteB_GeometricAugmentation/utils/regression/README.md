# Regression Dataset Adapters

更新时间：2026-03-28

这里存放的是**回归探针分支**专用的数据集接入层。

当前只保留：

- `ieeeppg_trials.py`
  - `IEEEPPG` 官方 train/test 回归数据
  - 输出统一的 trial 风格字典：
    - `trial_id_str`
    - `x_trial`
    - `y_value`
    - `split`

说明：

- 这里不是当前分类主线的数据入口
- 当前分类数据入口仍以 `datasets/trial_dataset_factory.py` 和各分类 trial loader 为主
