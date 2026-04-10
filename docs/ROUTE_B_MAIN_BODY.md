# Route B 主体骨架说明

更新时间：2026-03-22

## 1. 当前主体重排

从 2026-03-22 起，当前分类工程的主体口径调整为：

1. `Freeze` 主线
   - 降级为历史失败线 / 诊断线
   - 仅用于说明旧 `z-space -> Step1B -> Gate -> LinearSVC` 为什么不足
2. `Route B`
   - 升格为当前主体骨架
   - 定义为：
     - `target upgrade -> multiround curriculum -> bridge -> raw MiniROCKET validation`
3. `LRAES`
   - 升格为 Route B 的优先前端方向生成器
   - 目标是替换旧方向源 / TELM2 方向库
4. `Fisher/C0`
   - 降级为历史升级尝试 / 对照净化器
   - 不再作为主体前端优先选择

## 2. 主体骨架

当前主体骨架记为：

`LRAES front-end -> multiround curriculum -> bridge -> raw MiniROCKET`

当前已经开始向统一主链收口：

- [ROUTE_B_UNIFIED_POLICY.md](/home/THL/project/MTS-PIA/docs/ROUTE_B_UNIFIED_POLICY.md)
- [ROUTE_B_UNIFIED_INTERFACES.md](/home/THL/project/MTS-PIA/docs/ROUTE_B_UNIFIED_INTERFACES.md)
- [ROUTE_B_UNIFIED_MAPPING.md](/home/THL/project/MTS-PIA/docs/ROUTE_B_UNIFIED_MAPPING.md)

新的工程方向不是继续把 `LRAES / multiround / bridge` 当成并列增强件，而是把它们收口到：

`representation -> unified policy -> task-constrained bridge -> raw evaluator`

职责拆分：

- `LRAES`
  - 决定“往哪走”
  - 用 `M = (S_expand + lambda I) - beta * (S_risk + lambda I)` 的局部特征分解直接生成新方向
- `curriculum`
  - 决定“走多远”
  - 通过跨轮步长预算更新扩张 / 维持 / 收缩 / 冻结各方向
- `bridge`
  - 把 target 结构注回 raw 域
- `raw MiniROCKET`
  - 作为 raw 端统一外部验证器

## 3. 当前证据

### 3.1 Route B 已成立

simple-set bridge 结果见：

- [HAR / NATOPS / SCP1 / FM 总结](/home/THL/project/MTS-PIA/out/bridge_curriculum_fm_pilot_20260321/bridge_curriculum_all_simple_sets_compare.md)

当前结论：

- `HAR`
  - 完整正闭环
- `NATOPS`
  - 完整正闭环，而且更强
- `SelfRegulationSCP1`
  - 边界但非退化
- `FingerMovements`
  - high-volatility edge case

这说明：

- bridge 层整体干净
- 当前主矛盾更像上游 target 是否足够 task-aligned

### 3.2 LRAES 是当前优先前端候选

当前权威结果以 clean rerun 为准：

- [lraes_curriculum_performance_summary.csv](/home/THL/project/MTS-PIA/out/phase15_lraes_curriculum_core_20260322_rerun/lraes_curriculum_performance_summary.csv)
- [lraes_solver_summary.csv](/home/THL/project/MTS-PIA/out/phase15_lraes_curriculum_core_20260322_rerun/lraes_solver_summary.csv)
- [lraes_curriculum_conclusion.md](/home/THL/project/MTS-PIA/out/phase15_lraes_curriculum_core_20260322_rerun/lraes_curriculum_conclusion.md)

主报告 `beta=0.5` 下：

- `NATOPS`
  - 相对纯 curriculum 基本持平，微正
- `SelfRegulationSCP1`
  - 明显优于纯 curriculum
- `FingerMovements`
  - 明显负增益

因此当前更准确的判断是：

- `LRAES + curriculum` 已经证明对边界集 `SCP1` 有价值
- 但还没有强到可以无条件替换纯 curriculum
- 它已经值得作为 Route B 的优先前端候选继续推进

### 3.3 LRAES 已经接回 Route B bridge 骨架

第一轮 bridge-coupled LRAES pilot 已经在 `SelfRegulationSCP1` 上闭环：

- [bridge_lraes_scp1_pilot_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_scp1_pilot_20260322/bridge_lraes_scp1_pilot_summary.csv)
- [bridge_lraes_scp1_target_health_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_scp1_pilot_20260322/bridge_lraes_scp1_target_health_summary.csv)
- [bridge_lraes_scp1_fidelity_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_scp1_pilot_20260322/bridge_lraes_scp1_fidelity_summary.csv)
- [bridge_lraes_scp1_solver_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_scp1_pilot_20260322/bridge_lraes_scp1_solver_summary.csv)
- [bridge_lraes_scp1_pilot_conclusion.md](/home/THL/project/MTS-PIA/out/bridge_lraes_scp1_pilot_20260322/bridge_lraes_scp1_pilot_conclusion.md)

核心结论：

- `raw_only`
  - `macro_f1 = 0.8018 +/- 0.0111`
- `bridge_single_round`
  - `macro_f1 = 0.7577 +/- 0.0288`
- `bridge_multiround`
  - `macro_f1 = 0.7518 +/- 0.0289`
- `bridge_lraes`
  - `macro_f1 = 0.7755 +/- 0.0181`

这意味着：

- `LRAES bridge` 已经明确优于 `single-round bridge`
- `LRAES bridge` 也明确优于 `multiround bridge`
- 但当前仍未超过 `raw_only`

因此这轮更准确的定位是：

- `LRAES` 已从纯 target-side 候选，升级为 `Route B` 的 bridge-coupled 强前端候选
- `SCP1` 上它更像“边界修复升级正在转成更强 raw 收益”
- 但还不能写成已经形成完整正闭环

随后在 `NATOPS` 上做的第二站 bridge-coupled pilot 表明：

- [bridge_lraes_natops_pilot_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_natops_pilot_20260322/bridge_lraes_natops_pilot_summary.csv)
- [bridge_lraes_natops_target_health_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_natops_pilot_20260322/bridge_lraes_natops_target_health_summary.csv)
- [bridge_lraes_natops_fidelity_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_natops_pilot_20260322/bridge_lraes_natops_fidelity_summary.csv)
- [bridge_lraes_natops_solver_summary.csv](/home/THL/project/MTS-PIA/out/bridge_lraes_natops_pilot_20260322/bridge_lraes_natops_solver_summary.csv)
- [bridge_lraes_natops_pilot_conclusion.md](/home/THL/project/MTS-PIA/out/bridge_lraes_natops_pilot_20260322/bridge_lraes_natops_pilot_conclusion.md)

`NATOPS` 聚合结果为：

- `raw_only`
  - `macro_f1 = 0.7239 +/- 0.0350`
- `bridge_single_round`
  - `macro_f1 = 0.7379 +/- 0.0325`
- `bridge_multiround`
  - `macro_f1 = 0.7471 +/- 0.0120`
- `bridge_lraes`
  - `macro_f1 = 0.7247 +/- 0.0140`

这说明：

- `LRAES bridge` 没有复制 `SCP1` 上“优于 multiround”的结构
- 但它的 fidelity 仍然优于 `single-round` 与 `multiround`
- solver 侧也没有出现明显 `fully_risk_dominated`

因此当前对 `LRAES` 的 bridge-coupled 总判断应更新为：

- 它已经不是只停留在 target-side 的候选
- 但也还不能正式升格为 `Route B` 的默认前端
- 当前更准确的定位是：
  - `SCP1`：边界修复升级正在转成 raw 收益
  - `NATOPS`：target 更干净，但 raw 转化没有超过 `multiround`
- 所以下一步不应继续盲扩 bridge，而应先把 `LRAES` 视为“强前端候选，待进一步确认”

## 4. 当前不再作为主体的线

### 4.1 Freeze

`Freeze` 主线当前保留为：

- 历史失败线
- 诊断线
- 旧主体为何不足的证据来源

不再作为论文主体，不再作为工程优先推进方向。

### 4.2 Fisher/C0

`Fisher/C0 + curriculum` 当前保留为：

- 对照净化器
- 历史升级尝试
- 用于说明上游净化曾经有一定帮助，但不是当前首选前端

结果入口：

- [fisher_curriculum_performance_summary.csv](/home/THL/project/MTS-PIA/out/phase15_fisher_curriculum_core_20260321/fisher_curriculum_performance_summary.csv)

## 5. 当前建议阅读顺序

1. [docs/ROUTE_B_MAIN_BODY.md](/home/THL/project/MTS-PIA/docs/ROUTE_B_MAIN_BODY.md)
2. [工程记录/分类工程现状.md](/home/THL/project/MTS-PIA/工程记录/分类工程现状.md)
3. [工程记录/分类调试记录.md](/home/THL/project/MTS-PIA/工程记录/分类调试记录.md)
4. [lraes_curriculum_performance_summary.csv](/home/THL/project/MTS-PIA/out/phase15_lraes_curriculum_core_20260322_rerun/lraes_curriculum_performance_summary.csv)
5. [bridge_curriculum_all_simple_sets_compare.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_fm_pilot_20260321/bridge_curriculum_all_simple_sets_compare.md)
