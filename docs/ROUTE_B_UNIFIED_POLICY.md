# Route B Unified Policy-Driven 闭环框架

更新时间：2026-03-27

说明：

- 这份文档描述的是当前 `unified shell`
- 从 `2026-03-27` 起，工程口径进一步收紧为：
  - `PIA Core` 必须恢复为一级方法学核心
  - `unified policy` 只应视为外层调度壳
- 对应补充说明见：
  - [pia_operator_note.md](/home/THL/project/MTS-PIA/docs/pia_operator_note.md)
  - [unified_layer_refactor_note.md](/home/THL/project/MTS-PIA/docs/unified_layer_refactor_note.md)
  - [scheduler_role_note.md](/home/THL/project/MTS-PIA/docs/scheduler_role_note.md)

## 1. 新主体链

当前 Route B 不再被理解为：

- `LRAES`
- `multiround curriculum`
- `bridge`
- `raw MiniROCKET`

这几个增强件的顺序拼接。

新的主体链固定为：

`representation -> unified policy -> task-constrained bridge -> raw evaluator`

其中：

- `representation`
  - 固定 raw -> SPD/log-center/z-space 表示
- `unified policy`
  - 合并原 `LRAES + multiround`
  - 统一输出方向选择、权重、步长、stop rule
- `task-constrained bridge`
  - 保留现有 bridge 主协议
  - 新增 classwise fidelity 与 inter-class margin proxy
- `raw evaluator`
  - 固定为 raw MiniROCKET
  - 只提供 final objective 与 inner-validation posterior

## 2. 当前实现原则

- 新主链放在 `route_b_unified/` 新命名空间
- 旧 `multiround / lraes / bridge` runner 保留为 legacy/reference
- feedback 只来自训练内 inner validation
- 最终 test 只做最后评估，不参与 policy update

## 3. 第一版 unified policy

第一版仍保持显式、轻量、闭式、可解释：

- `score_base(d)` 来自 LRAES 谱解或 legacy uniform prior
- `policy_step` 负责：
  - direction ranking
  - top-K selection
  - per-direction weights
  - step-size allocation
- `apply_target_feedback` 负责：
  - target-side budget update
- `update_policy` 负责：
  - posterior -> next-round score adjustment

第一版更新律：

`score_{t+1}(d) = score_t(d) + eta_1 * reward_t(d) - eta_2 * penalty_t(d)`

当前 feedback 只允许吸收：

- direction usage entropy
- worst-dir signal
- classwise bridge distortion
- round gain proxy

## 4. 当前 runner

统一 runner：

- [run_route_b_unified_probe.py](/home/THL/project/MTS-PIA/scripts/route_b/run_route_b_unified_probe.py)

当前第一批主判断集固定为：

- `SelfRegulationSCP1`
- `NATOPS`

## 5. 当前必须落盘的日志

- `policy_summary.csv`
- `classwise_bridge_summary.csv`
- `feedback_update_summary.csv`
- `final_coupling_summary.csv`

这些日志对应的职责是：

- policy 内部动作
- bridge 的 classwise / task-risk 诊断
- posterior 如何改变下一轮策略
- final raw-side objective 如何回收整个主链
