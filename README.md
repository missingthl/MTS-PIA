# MTS-PIA Workspace

更新时间：2026-04-19

这个仓库现在应该按“**当前工程状态**”来读，而不是按历史阶段线索倒推。

默认读法是：

- 先接住 **当前主验证线**
- 再看 **外部宿主验证线**
- 最后再按需回到历史结构证据、standalone 和 archive

## 最短入口

如果你想最快接住当前仓库，只看下面 5 份：

1. [docs/CURRENT_ENGINEERING_MANIFEST.json](docs/CURRENT_ENGINEERING_MANIFEST.json)
2. [docs/CURRENT_ENGINEERING_MAP.md](docs/CURRENT_ENGINEERING_MAP.md)
3. [ASSISTANT_ENTRY.md](ASSISTANT_ENTRY.md)
4. [工程记录/分类/00-入口/分类工程现状.md](工程记录/分类/00-入口/分类工程现状.md)
5. [out/_active/README.md](out/_active/README.md)

运行资源与服务器隔离说明看：

- [SERVER_RESOURCE_GUIDE.md](SERVER_RESOURCE_GUIDE.md)

## 当前默认阅读口径

### 1. 默认主验证线

- `ResNet1D + DLCR`

这是当前默认先读的线。当前最重要的三层结果都已经齐了：

- 行为变量正式矩阵：
  - [verify_resnet1d_dlcr_behavioral_matrix_20260418](out/_active/verify_resnet1d_dlcr_behavioral_matrix_20260418)
- `tangent probe` 全量谱证据：
  - [verify_resnet1d_tangent_probe_20260418](out/_active/verify_resnet1d_tangent_probe_20260418)
- `center_tangent(k=4)` 全量正式对照：
  - [verify_resnet1d_center_tangent_fullscale_20260419](out/_active/verify_resnet1d_center_tangent_fullscale_20260419)

### 2. 外部宿主验证线

- `Tensor-CSPNet + DLCR`

这是 EEG / SPD 外部宿主验证线，不是废线，但不是当前默认先读的通用 MTS 主线。

结果入口：

- [verify_tensor_cspnet_local_closed_form_holdout_20260412](out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412)

### 3. 已接入但尚未形成主结果层的宿主

- `PatchTST + DLCR`
- `TimesNet + DLCR`

它们的代码接入已经完成，但当前 GitHub 里还没有对应的全量正式主结果层。

## 当前最重要的已核实结论

### `ResNet1D + DLCR`

- `center_only vs E0`：`8 胜 / 2 平 / 11 负`
- `best center_subproto vs E0`：`14 胜 / 1 平 / 6 负`
- `best center_subproto vs center_only`：`14 胜 / 4 平 / 3 负`
- `center_tangent(k=4) vs E0`：`11 胜 / 1 平 / 9 负`
- `center_tangent(k=4) vs best center_subproto`：`5 胜 / 3 平 / 13 负`
- `tangent probe` 21 数据集上 `rank95_mean` 全部为 `4.0`

当前稳妥口径：

- `center_only` 是必须保留的强基线
- `center_subproto` 是当前最稳的增强主分支
- `center_tangent(k=4)` 已证明自己不是无效分支，但还不是统一默认主线
- `subproto_temperature` 是真实行为变量，但明显数据集相关

### `Tensor-CSPNet + DLCR`

- `BCIC-IV-2a holdout` 外部宿主复现已完成
- 当前公开读法仍是：
  - 宿主可靠
  - `E2` 尚未在该宿主上建立稳定优势

## 目录怎么读

默认优先进入：

- `models/`
- `scripts/`
- `docs/`
- `工程记录/`
- `out/_active/`

默认后读或跳过：

- `archive/`
- `standalone_projects/`
- `route_b_unified/`
- `PIA/`
- `manifold_raw/`
- `out/_active/` 里的 `_tmp* / _scratch / 旧 route_b 目录`

## 当前这份仓库不该再怎么读

不要再把当前仓库理解成：

- 仍以 `route_b` 为默认主线
- 只有 `tau=0.2 corrected` 这一轮旧 sweep，没有新的行为变量矩阵
- 只有 `tangent probe` 有 21 数据集，而 `center_tangent` 没有
- `MBA` 已并入当前主仓库主线 ranking

当前更准确的总读法是：

**`ResNet1D + DLCR` 是默认主验证线；`Tensor-CSPNet + DLCR` 是外部宿主验证线；`PatchTST / TimesNet` 已完成宿主接入；主结果先看行为矩阵、probe 和 center_tangent 全量对照。**
