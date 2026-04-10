# Route B Unified Interfaces

更新时间：2026-03-22

## 1. Representation

`build_representation(split_cfg) -> RepresentationState`

最小输出：

- `train/val/test` raw trial dicts
- `Z0_train / Z0_val / Z0_test`
- `y / tid`
- `mean_log_train`
- split meta

## 2. Policy

`init_policy(rep_state, policy_cfg) -> PolicyState`

`policy_step(rep_state, policy_state, posterior_prev | None, policy_cfg) -> PolicyAction, PolicyState`

`apply_target_feedback(policy_state, margin_by_dir, flip_by_dir, intrusion_by_dir, policy_cfg)`

`update_policy(policy_state, posterior_t, policy_cfg) -> PolicyState, PolicyUpdateSummary`

`PolicyAction` 当前至少包含：

- selected directions
- direction weights
- per-direction step sizes
- entropy
- stop flag / reason

## 3. Bridge

`apply_bridge(rep_state, target_state, bridge_cfg, variant=...) -> BridgeResult`

`BridgeResult` 当前至少包含：

- bridged train trials
- passthrough val/test trials
- global fidelity
- classwise fidelity
- inter-class margin proxy
- task-risk comment

## 4. Evaluator

`evaluate_bridge(bridge_result, eval_cfg, split_name, target_state, round_gain_proxy) -> EvaluatorPosterior`

`EvaluatorPosterior` 当前至少包含：

- val/test acc
- val/test macro_f1
- round gain proxy
- worst-dir summary
- direction metrics
- classwise distortion summary
- inter-class margin proxy

## 5. Unified Logs

当前统一主链至少要维护：

- `policy_summary.csv`
- `classwise_bridge_summary.csv`
- `feedback_update_summary.csv`
- `final_coupling_summary.csv`

旧脚本产生的单独 target / bridge / solver summary 不再视为主体终点，而是 legacy/reference。
