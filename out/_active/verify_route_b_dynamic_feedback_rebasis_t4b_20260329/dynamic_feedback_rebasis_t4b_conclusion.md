# T4b Window-Conditioned Rebasis-Informative Feedback Pool Conclusion

This first-pass T4b probe freezes generator, shared rebasis, window policy, and classifier, and upgrades only the feedback-pool object from whole trajectories to window-level objects.

Interpretation guardrails:

- `radial_gain` is treated as an outward-expansion proxy, not as a proven rebasis-optimal signal.
- `margin_gain` is treated as a discriminative-gain proxy, not as a proven final answer.
- admitted windows enter rebasis as length-1 pseudo sequences; T4b therefore measures window-conditioned rebasis signal, not full segment-aware rebasis geometry.

## selfregulationscp1 (main)

- baseline: 0.5144 +/- 0.0000
- t2a_default: 0.5620 +/- 0.0000
- t3_shared_rebasis: 0.5620 +/- 0.0000
- t4b_window_safety_only: 0.5620 +/- 0.0000
- t4b_window_radial_gate: 0.5620 +/- 0.0000
- t4b_window_margin_gate: 0.5620 +/- 0.0000

## natops (anchor)

- t3_shared_rebasis: 0.7335 +/- 0.0000
- t4b_window_safety_only: 0.7389 +/- 0.0000
- t4b_window_radial_gate: 0.7389 +/- 0.0000
- t4b_window_margin_gate: 0.7402 +/- 0.0000

## Success Layers

- Weak success: `window_safety_only > t3_shared_rebasis`.
- Medium success: `window_radial_gate` or `window_margin_gate > window_safety_only`.
- Strong success: SCP1 performance improves and basis shift changes from near-static to non-trivial and interpretable.
