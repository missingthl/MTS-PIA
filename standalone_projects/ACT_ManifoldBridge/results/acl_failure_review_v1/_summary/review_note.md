# ACL v1 Failure Review

- frozen_root: `/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results/acl_small_matrix_v1`
- review_root: `/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results/acl_failure_review_v1`

## handwriting
- label: `acl_no_clear_advantage_yet`
- rationale: at least one follow-up variant recovers against continue_ce; best_variant=gcg_acl_align1p0
- best_variant: `mba_ref`
- best_variant_mean_f1: `0.45725025728860796`
- best_variant_delta_vs_continue_ce: `0.03182362130770833`

## atrialfibrillation
- label: `objective_mismatch_issue`
- rationale: all ACL variants remain below continue_ce; best_variant=gcg_acl_temp0p10
- best_variant: `mba_ref`
- best_variant_mean_f1: `0.2512820512820513`
- best_variant_delta_vs_continue_ce: `-0.010541310541310559`

## motorimagery
- label: `objective_mismatch_issue`
- rationale: all ACL variants remain below continue_ce; best_variant=gcg_acl_align1p0
- best_variant: `mba_ref`
- best_variant_mean_f1: `0.4929651554258781`
- best_variant_delta_vs_continue_ce: `-0.01881005936490691`
