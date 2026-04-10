# T6b Dual-role Sample Policy Conclusion

This first-pass T6b probe freezes generator, shared rebasis, and classifier, and separates constructive windows from discriminative windows.

Interpretation guardrails:

- `window_safety_only` is used as the first constructive-side scaffold because it is safer and less biased than the current stronger geometry-moving gates; this does not prove it is the uniquely optimal constructive gate.
- `local_kNN_margin` is used as the first discriminative-side local proxy because T6a-1 suggests it is not suitable as a unified gate; this does not prove it is the uniquely optimal discriminative gate.
- all kNN queries use `orig-train-only windows`, and same-class local neighbors exclude the current window itself.
- final stitched trajectories are constructed only at the `z_seq` representation level; no raw-level stitching is used.
- low-coverage interpretation is class-level first, then dataset summary; average coverage alone is not sufficient to claim the gate has been effectively triggered.

## selfregulationscp1 (main)

- baseline: 0.5144 +/- 0.0000
- t2a_default: 0.5620 +/- 0.0000
- t5_dual_role_policy: 0.5665 +/- 0.0000
- t3_shared_rebasis: 0.5620 +/- 0.0000
- t4b_window_radial_gate: 0.5620 +/- 0.0000
- t4b_window_margin_gate: 0.5620 +/- 0.0000
- t6a1_local_knn_margin_unified: 0.5286 +/- 0.0000
- t6b_dual_role_policy: 0.5937 +/- 0.0000

## natops (anchor)

- t3_shared_rebasis: 0.7335 +/- 0.0000
- t4b_window_radial_gate: 0.7389 +/- 0.0000
- t4b_window_margin_gate: 0.7402 +/- 0.0000
- t6a1_local_knn_margin_unified: 0.7214 +/- 0.0000
- t6b_dual_role_policy: 0.7266 +/- 0.0000

## Success Layers

- Weak success: `t6b_dual_role_policy > t3_shared_rebasis` and no obvious stitching pathology.
- Medium success: `t6b_dual_role_policy > max(t4b_window_margin_gate, t6a1_local_knn_margin_unified)` and role overlap is not near-total.
- Strong success: SCP1 improves in both basis movement and end performance while class-level low-coverage does not dominate key classes.
