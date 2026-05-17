# Internal Controls

These arms are useful mechanism controls, but they are **not** external paper
baselines.

Included arms:

- `random_cov_state`
- `pca_cov_state`

Use them in mechanism and ablation tables, not in the external-baseline main
table.

Rationale:

- `random_cov_state` tests whether covariance-state perturbation alone is
  strong.
- `pca_cov_state` tests whether global train covariance-state PCA directions are
  enough.
- Neither is an external augmentation method from another paper.
