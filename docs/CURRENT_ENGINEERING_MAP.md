# Current Engineering Map

更新时间：2026-04-19

这份文件只回答三件事：

1. 当前仓库默认应该先看什么
2. 当前结果层有哪些已经闭环
3. 哪些目录应该默认后放

## 1. 当前仓库的四层结构

### A. 当前代码层

- `models/`
- `scripts/`

### B. 当前结果层

- `out/_active/`

### C. 工程解释层

- `docs/`
- `工程记录/`

### D. 后放层

- `archive/`
- `standalone_projects/`
- `route_b_unified/`
- `PIA/`
- `manifold_raw/`

## 2. 默认阅读顺序

1. [CURRENT_ENGINEERING_MANIFEST.json](CURRENT_ENGINEERING_MANIFEST.json)
2. [../ASSISTANT_ENTRY.md](../ASSISTANT_ENTRY.md)
3. [../工程记录/分类/00-入口/分类工程现状.md](../工程记录/分类/00-入口/分类工程现状.md)
4. [../out/_active/README.md](../out/_active/README.md)
5. [../models/README.md](../models/README.md)
6. [../scripts/README.md](../scripts/README.md)

## 3. 当前默认主线与宿主层

### 默认主验证线

- `ResNet1D + DLCR`

当前已经闭环的主结果层：

- 行为变量正式矩阵
- `tangent probe` 全量谱证据
- `center_tangent(k=4)` 全量正式对照

### 外部宿主验证线

- `Tensor-CSPNet + DLCR`

当前角色：

- EEG / SPD 外部宿主验证
- 不是默认主结果入口

### 已接入但未形成主结果层的宿主

- `PatchTST + DLCR`
- `TimesNet + DLCR`

当前角色：

- 代码接入已完成
- 还没有 21 数据集级别的主结果层

## 4. 当前结果层分布

### 4.1 行为变量正式矩阵

目录：

- [../out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418](../out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418)

状态：

- `21` 数据集
- `126` 条条件
- 已完成

主要回答：

- `center_only` 是否值得长期保留
- `center_subproto` 是否真的带来额外结构收益
- `subproto_temperature` 是否是数据集相关变量

### 4.2 `tangent probe`

目录：

- [../out/_active/verify_resnet1d_tangent_probe_20260418](../out/_active/verify_resnet1d_tangent_probe_20260418)

状态：

- `21` 数据集
- 已完成

主要回答：

- 当前子原型偏移谱是否支持低秩切空间读法

当前稳妥结论：

- `rank95_mean = 4.0`
- 更像接近满秩 `4` 维，而不是明显低秩

### 4.3 `center_tangent(k=4)` 全量对照

目录：

- [../out/_active/verify_resnet1d_center_tangent_fullscale_20260419](../out/_active/verify_resnet1d_center_tangent_fullscale_20260419)

状态：

- `21` 数据集
- 已完成

主要回答：

- `center_tangent(k=4)` 是否已经足够强，可以升格成统一默认主线

当前稳妥结论：

- 它不是无效分支
- 但还不是统一默认主线

### 4.4 `Tensor-CSPNet + DLCR`

目录：

- [../out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412](../out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412)

状态：

- 外部宿主验证已完成

当前稳妥结论：

- 宿主可靠
- `E2` 尚未在该宿主上建立稳定优势

## 5. 当前辅助层

这些目录重要，但不是默认第一阅读层：

- [../out/_active/verify_resnet1d_subproto_temp_sweep_20260414](../out/_active/verify_resnet1d_subproto_temp_sweep_20260414)
- [../out/_active/verify_resnet1d_dataflow_probe_20260414](../out/_active/verify_resnet1d_dataflow_probe_20260414)
- [../out/_active/verify_e2_tau02_fullscale_20260414](../out/_active/verify_e2_tau02_fullscale_20260414)
- [../out/_active/verify_3baseline_minirocket_20260414](../out/_active/verify_3baseline_minirocket_20260414)

## 6. 默认后放区

### 历史结构证据层

- `verify_route_b_*`
- `verify_axis_pullback_*`
- `verify_pia_core_*`
- `route_b_dynamic_*`
- `verify_logeuclidean_*`

这些目录当前仍保留在仓库里，但不应作为默认结果入口。

### 本地临时层

- `out/_active/_tmp*`
- `out/_active/_scratch`

### standalone

- `standalone_projects/MBA_ManifoldBridge`
- 其他 `standalone_projects/*`

## 7. 当前最简总读法

**先从 `ResNet1D` 的行为矩阵、probe、center_tangent 全量对照接管当前主线；再看 `Tensor-CSPNet` 外部宿主验证；最后才按需回到辅助层、历史层和 standalone。**
