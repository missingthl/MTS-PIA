# Direction Specificity Stress Report

- out_root: `standalone_projects/ACT_ManifoldBridge/results/csta_direction_specificity_stress_v1/resnet1d_s123`
- phase1_root: `standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123`
- phase2_root: `standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase2/resnet1d_s123`
- extra_roots: `(none)`
- rows_with_refs: `147`

Generated files:
- `per_seed_direction_specificity_with_refs.csv`
- `dataset_summary_external.csv`
- `overall_summary_external.csv`
- `direction_specificity_diagnostics.csv`
- `direction_specificity_completion.csv`
- `direction_specificity_win_tie_loss.csv`

Interpretation guardrail: distinguish full random covariance (`random_cov_state`) from random template sampling inside the PIA dictionary (`csta_template_random_within_bank`).

## Direction Specificity Questions

1. Bank-random close to U5? U5=0.665242, bank-random=0.653546, delta=-0.011696.
2. Bank-random better than full random covariance? bank-random=0.653546, full-random=0.64813, delta=0.00541572.
3. Top1 underperforms U5? top1=0.648579, U5=0.665242, delta_top1_minus_u5=-0.0166631.
4. Response gaps flat? U5 mean top1-top5 response gap=0.42607; inspect `direction_specificity_diagnostics.csv` before making geometric claims.
5. Safe-step shrinkage comparable? post-safe norms: U5=0.0994325, bank-random=0.0994325, full-random=nan.
6. Relaxed eta exposes direction specificity? Compare eta-tagged rows in `direction_specificity_diagnostics.csv` if stress roots were supplied.
7. Claim boundary: this report should not claim manifold flatness or direction superiority unless performance, response-gap, and post-safe diagnostics jointly support it.
