# Route B Legacy-to-Unified Mapping

更新时间：2026-03-22

## 旧逻辑 -> 新逻辑

- 旧 `LRAES`
  - 新 `policy.score_base`
- 旧 `multiround gamma schedule`
  - 新 `policy action / target feedback`
- 旧 `best round logic`
  - 新 `policy stop rule + inner-val best round selection`
- 旧 `bridge fidelity summary`
  - 新 `bridge global constraints`
- 新增 `classwise bridge summary`
  - 新 `bridge task constraints`
- 旧 `raw MiniROCKET final score`
  - 新 `evaluator posterior + final objective`

## 当前边界

当前 unified framework 不做：

- seed 家族
- Gate3 / Controller Lite / upgrade manifold
- Fisher/C0 主体竞争
- 黑箱 RL / controller

当前 unified framework 保留：

- 旧 runner 作为 legacy/reference
- 旧结果作为历史对照
