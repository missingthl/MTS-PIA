# 分类工程记录

更新时间：2026-04-19

这份目录现在只承担一件事：

- 给当前分类工程提供**解释层入口**

如果你是 agent，不要先从这里倒推历史；先接住主结果，再回来看这里。

## 推荐阅读顺序

1. [../../docs/CURRENT_ENGINEERING_MANIFEST.json](../../docs/CURRENT_ENGINEERING_MANIFEST.json)
2. [../../docs/CURRENT_ENGINEERING_MAP.md](../../docs/CURRENT_ENGINEERING_MAP.md)
3. [00-入口/分类工程现状.md](00-入口/分类工程现状.md)
4. [00-入口/分类调试记录.md](00-入口/分类调试记录.md)
5. [../../ASSISTANT_ENTRY.md](../../ASSISTANT_ENTRY.md)

## 当前口径

- 默认主验证线：`ResNet1D + DLCR`
- 外部宿主验证线：`Tensor-CSPNet + DLCR`
- 已接入宿主线：`PatchTST / TimesNet + DLCR`
- `MiniRocket + DLCR`：边界 / 诊断线
- `MBA_ManifoldBridge`：standalone

## 这个目录里各入口的职责

### `00-入口/分类工程现状.md`

唯一权威状态页，负责：

- 当前主线定义
- 当前已核实结果快照
- 当前未闭环问题
- 当前不写进权威结论的边界

### `00-入口/分类调试记录.md`

只负责：

- 调试过程
- 阶段性转折
- 排障结论

### `01-阶段二-宿主实验`

宿主专项任务单与阶段记录。

### `02-阶段一-PIA-Operator`

历史结构证据层。默认不作为当前主结果入口。

## 默认不从这里直接得出排名

不要从这个目录里直接推出：

- `MiniRocket` 已经是主结果线
- `MBA` 已并入主线 ranking
- 历史 `route_b` 结果仍然等价于当前主线
