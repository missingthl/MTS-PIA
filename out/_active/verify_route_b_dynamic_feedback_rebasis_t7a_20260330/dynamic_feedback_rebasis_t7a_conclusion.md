# T7a Window-level Constructive Pool + Class-Conditioned Rebasis Conclusion

## Key Findings
- `selfregulationscp1`
  - baseline: 0.5144 +/- 0.0000
  - t2a_default: 0.5620 +/- 0.0000
  - t3_shared_rebasis: 0.5620 +/- 0.0000
  - t4b_window_radial_gate: 0.5800 +/- 0.0000
  - t7a_class_conditioned_rebasis: 0.5657 +/- 0.0000
  - best_mode: `t4b_window_radial_gate`
  - inter_basis_cosine_mean: 1.0000
  - inter_basis_cosine_min: 1.0000
  - inter_basis_cosine_max: 1.0000
- `natops`
  - baseline: 0.6857 +/- 0.0000
  - t2a_default: 0.7335 +/- 0.0000
  - t3_shared_rebasis: 0.7335 +/- 0.0000
  - t4b_window_radial_gate: 0.7337 +/- 0.0000
  - t7a_class_conditioned_rebasis: 0.7335 +/- 0.0000
  - best_mode: `t4b_window_radial_gate`
  - inter_basis_cosine_mean: 0.2015
  - inter_basis_cosine_min: 0.0016
  - inter_basis_cosine_max: 0.8584

## Reading Notes
- T7a routes class-conditioned containers only during training-time augmentation generation.
- Validation/test always consume original trajectories only; there is no class-conditioned test-time routing.
- Each class-conditioned axis uses train-only one-vs-rest fitting so the container remains discriminative rather than collapsing into class-only unsupervised PCA-like axes.
