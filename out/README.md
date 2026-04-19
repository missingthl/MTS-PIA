# Output Layout

更新时间：2026-04-19

`out/` 现在应该按下面方式来读：

1. `out/_active/`
   - 当前结果层
   - 默认先读这里
2. 其他 `out/*`
   - 历史复现、专项批次、阶段性产物
   - 默认后放

## 默认入口

- [./_active/README.md](./_active/README.md)

## 默认规则

- 先看上层 `md / csv / json` 汇总文件
- 不要默认直接扎进深层逐 run 目录
- 本地 `_tmp* / _scratch` 产物不作为权威结论层
