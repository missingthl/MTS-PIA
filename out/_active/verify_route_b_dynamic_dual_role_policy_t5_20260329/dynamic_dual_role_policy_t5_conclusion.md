# T5 Dual-role Sample Policy Conclusion

This first-pass T5 probe freezes generator, window policy, shared rebasis, and classifier, and separates constructive samples from discriminative samples.

Interpretation guardrails:

- `radial_gain` is fixed as the constructive-side proxy because T4b showed stronger basis-moving behavior under radial gating; this does not prove radial is the uniquely optimal constructive gate.
- `margin_gain` is fixed as the discriminative-side proxy because it is more directly aligned with class separation; this does not prove margin is the uniquely optimal discriminative gate.
- final stitched trajectories are constructed only at the `z_seq` representation level; no raw-level stitching is used.
- T5 measures the overall effect of a dual-role sample policy, not a strict causal decomposition of radial versus margin.

## selfregulationscp1 (main)

- baseline: 0.5144 +/- 0.0000
- t2a_default: 0.5620 +/- 0.0000
- t3_shared_rebasis: 0.5620 +/- 0.0000
- t4b_window_radial_gate: 0.5620 +/- 0.0000
- t4b_window_margin_gate: 0.5620 +/- 0.0000
- t5_dual_role_policy: 0.5665 +/- 0.0000

## natops (anchor)

- t3_shared_rebasis: 0.7335 +/- 0.0000
- t4b_window_radial_gate: 0.7389 +/- 0.0000
- t4b_window_margin_gate: 0.7402 +/- 0.0000
- t5_dual_role_policy: 0.7253 +/- 0.0000

## Success Layers

- Weak success: `t5_dual_role_policy > t3_shared_rebasis` and no obvious stitching pathology.
- Medium success: `t5_dual_role_policy > max(t4b_window_radial_gate, t4b_window_margin_gate)` and role overlap is not near-total.
- Strong success: SCP1 improves in both basis movement and end performance while stitching continuity remains healthy.
