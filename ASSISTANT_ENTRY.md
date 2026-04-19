# Assistant Entry

更新时间：2026-04-19

这份文件给语言模型一个**最短、最不容易读偏**的入口。

## 先记住这 6 条

1. 默认先读的是 `ResNet1D + DLCR`，不是历史 `route_b`。
2. `Tensor-CSPNet + DLCR` 仍然有效，但它是外部宿主验证线，不是默认主结果入口。
3. 当前 `ResNet1D` 相关有两层都已经是 **21 数据集全量**：
   - `tangent probe`
   - `center_tangent(k=4)` 正式性能对照
4. `PatchTST / TimesNet` 宿主已经接进代码，但还没有形成主结果层。
5. `MiniRocket + DLCR` 仍是边界 / 诊断线，`MBA_ManifoldBridge` 是 standalone。
6. 默认跳过：
   - `archive/`
   - `route_b_unified/`
   - `PIA/`
   - `manifold_raw/`
   - `out/_active/` 里的 `_tmp* / _scratch / 旧 route_b 目录`

## 最短阅读路径

1. [docs/CURRENT_ENGINEERING_MANIFEST.json](docs/CURRENT_ENGINEERING_MANIFEST.json)
2. [docs/CURRENT_ENGINEERING_MAP.md](docs/CURRENT_ENGINEERING_MAP.md)
3. [工程记录/分类/00-入口/分类工程现状.md](工程记录/分类/00-入口/分类工程现状.md)
4. [out/_active/README.md](out/_active/README.md)
5. [models/README.md](models/README.md)
6. [scripts/README.md](scripts/README.md)

## 当前结果层怎么读

### 1. 行为变量正式矩阵

- [out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418](out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418)

这是当前 `ResNet1D + DLCR` 的默认主结果入口。

关键结论：

- `center_only vs E0`：`8 胜 / 2 平 / 11 负`
- `best center_subproto vs E0`：`14 胜 / 1 平 / 6 负`
- `best center_subproto vs center_only`：`14 胜 / 4 平 / 3 负`
- `tau` 没有统一默认值

### 2. `tangent probe`

- [out/_active/verify_resnet1d_tangent_probe_20260418](out/_active/verify_resnet1d_tangent_probe_20260418)

这是当前的**谱证据层**。

关键结论：

- `21 / 21` 数据集已完成
- `rank95_mean` 全部为 `4.0`
- 当前并不支持“明显低秩切空间”的强结论

### 3. `center_tangent(k=4)` 全量对照

- [out/_active/verify_resnet1d_center_tangent_fullscale_20260419](out/_active/verify_resnet1d_center_tangent_fullscale_20260419)

这是当前的**正式性能层**。

关键结论：

- `21 / 21` 数据集已完成
- 相对 `E0`：`11 胜 / 1 平 / 9 负`
- 相对 `best center_subproto`：`5 胜 / 3 平 / 13 负`
- `center_tangent(k=4)` 现在更像“选择性高上限分支”，还不是统一默认主线

### 4. `Tensor-CSPNet + DLCR`

- [out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412](out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412)

当前读法：

- 外部宿主复现成立
- `E2` 尚未在该宿主上建立稳定优势

## 默认代码入口

### 模型

1. [models/local_closed_form_residual_head.py](models/local_closed_form_residual_head.py)
2. [models/resnet1d_local_closed_form.py](models/resnet1d_local_closed_form.py)
3. [models/patchtst_local_closed_form.py](models/patchtst_local_closed_form.py)
4. [models/timesnet_local_closed_form.py](models/timesnet_local_closed_form.py)
5. [models/tensor_cspnet_adapter.py](models/tensor_cspnet_adapter.py)

### 脚本

1. [scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py](scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py)
2. [scripts/hosts/run_resnet1d_dlcr_behavioral_matrix.py](scripts/hosts/run_resnet1d_dlcr_behavioral_matrix.py)
3. [scripts/hosts/run_resnet1d_tangent_probe_matrix.py](scripts/hosts/run_resnet1d_tangent_probe_matrix.py)
4. [scripts/hosts/run_resnet1d_center_tangent_fullscale.py](scripts/hosts/run_resnet1d_center_tangent_fullscale.py)
5. [scripts/hosts/run_tsl_local_closed_form_fixedsplit.py](scripts/hosts/run_tsl_local_closed_form_fixedsplit.py)

## 一句话版

**默认先读 `ResNet1D + DLCR` 的行为矩阵、probe 和 center_tangent 全量对照；把 `Tensor-CSPNet` 当外部宿主验证线，把 `PatchTST / TimesNet` 当已接入但未完成主结果层的宿主，把 `route_b / archive / standalone` 默认后放。**
