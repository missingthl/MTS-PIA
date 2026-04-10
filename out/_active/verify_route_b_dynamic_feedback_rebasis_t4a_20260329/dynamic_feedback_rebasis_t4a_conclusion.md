# T4a Class-Conditioned Basis Family Re-basis Conclusion

This first-pass T4a probe freezes the T3 feedback-pool protocol and upgrades only the rebasis organization from a shared single axis to a class-conditioned single-axis family.

Important scope note:

- T4a measures the overall effect of a class-conditioned basis-family generator during training-time augmentation.
- It does not isolate basis organization as the only causal source, because train-time basis selection also uses true class labels.

## natops

- baseline: 0.6857 +/- 0.0000
- t2a_default: 0.7335 +/- 0.0000
- t3_shared_rebasis: 0.7335 +/- 0.0000
- t4a_class_conditioned_rebasis: 0.7375 +/- 0.0000
- inter-basis cosine mean: 0.0928
- inter-basis cosine min: 0.0117
- inter-basis cosine max: 0.3041

## selfregulationscp1

- baseline: 0.5144 +/- 0.0000
- t2a_default: 0.5620 +/- 0.0000
- t3_shared_rebasis: 0.5620 +/- 0.0000
- t4a_class_conditioned_rebasis: 0.5620 +/- 0.0000
- inter-basis cosine mean: 0.9891
- inter-basis cosine min: 0.9891
- inter-basis cosine max: 0.9891

## Judgment

- T4a only counts as successful if class-conditioned basis-family rebasis improves end performance or yields clearer SCP1 gains than T3 shared rebasis.
- Basis family differences alone do not count as success; they must be non-degenerate, non-runaway, and aligned with better downstream behavior.
- If T4a fails, the next likely bottleneck is feedback-pool object quality rather than another round of operator parameter tuning.
