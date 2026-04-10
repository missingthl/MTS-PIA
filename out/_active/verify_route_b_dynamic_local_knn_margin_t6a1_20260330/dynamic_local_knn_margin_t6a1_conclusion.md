# T6a-1 Local kNN Margin Unified Conclusion

This first-pass T6a-1 probe freezes generator, shared rebasis, and classifier, and upgrades only the informative gate from global radial/global margin to local kNN margin under a unified-role policy.

Interpretation guardrails:

- `local_knn_margin` is computed only against `orig-train-only windows`; same-class neighbors must exclude the current window itself.
- local kNN margin is a local discriminative proxy, not a proven final answer.
- admitted windows drive both rebasis and final z_seq-level writeback; T6a-1 therefore measures a unified-role policy, not dual-role sample separation.
- final writeback happens only at the `z_seq` level; no raw-level stitching is used.
- low-coverage classes are tracked explicitly via `max(8, ceil(0.05 * safe_window_count_class))` and cannot be over-interpreted as fully triggered gate success.

## selfregulationscp1 (main)

- baseline: 0.5144 +/- 0.0000
- t2a_default: 0.5620 +/- 0.0000
- t3_shared_rebasis: 0.5620 +/- 0.0000
- t4b_window_radial_gate: 0.5620 +/- 0.0000
- t4b_window_margin_gate: 0.5620 +/- 0.0000
- t6a1_local_knn_margin_unified: 0.5286 +/- 0.0000

## natops (anchor)

- t3_shared_rebasis: 0.7335 +/- 0.0000
- t4b_window_margin_gate: 0.7402 +/- 0.0000
- t6a1_local_knn_margin_unified: 0.7214 +/- 0.0000

## Success Layers

- Weak success: `t6a1_local_knn_margin_unified > t3_shared_rebasis` and no obvious stitching pathology.
- Medium success: `t6a1_local_knn_margin_unified > t4b_window_margin_gate`, indicating local discriminative structure is more useful than the current global margin proxy.
- Strong success: SCP1 improves in both basis movement and end performance while major classes avoid low-coverage collapse.
