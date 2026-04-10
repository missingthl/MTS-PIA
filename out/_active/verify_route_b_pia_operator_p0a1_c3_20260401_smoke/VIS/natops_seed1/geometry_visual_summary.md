# C3 Geometry Summary: natops seed=1

## High-Level Read

- `C3` has rotated the single template much closer to prototype-pair normal directions than `C0/C2`.
- This rotation is not obviously explained by same/opp pool imbalance, because both count ratio and weight-mass ratio stay near `1.0`.
- The remaining bottleneck is not primarily axis discovery anymore; it is the force/readout coupling after the axis has rotated.

## Pair-Axis Alignment

- `pair cosine mean`: C0=0.0112, C2=0.0324, C3=0.1592
- `pair cosine median`: C0=0.0167, C2=0.0634, C3=0.1786
- `positive pair count`: C0=14/24, C2=14/24, C3=16/24
- `same-only fit-row local cosine mean`: C0=0.0111, C2=0.0232

- Figure: [pair_axis_cosine_comparison.png](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/pair_axis_cosine_comparison.png)

## Pool Balance Audit

- `same_pool_count`: 192
- `opp_pool_count`: 192
- `same_weight_mass`: 24.0000
- `opp_weight_mass`: 24.0000
- `same_opp_count_ratio`: 1.0000
- `same_opp_weight_mass_ratio`: 1.0000

## Representative Pair

- `pair`: s1p2|o2p0
- `pair cosine`: C0=0.0109, C2=0.0013, C3=0.1689
- Figure: [representative_pair_pca.png](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/representative_pair_pca.png)

## Force-Field Readout

- `train response_vs_margin_correlation`: C0=0.1077, C2=0.1098, C3=-0.0209
- Geometric reading: the axis has rotated, but the deployed response field has not yet become margin-monotone under the current A2r force rule.
- Figure: [train_response_margin_scatter.png](/home/THL/project/MTS-PIA/out/_active/verify_route_b_pia_operator_p0a1_c3_20260401_smoke/VIS/natops_seed1/train_response_margin_scatter.png)
