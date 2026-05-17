# Paper Experiment Evidence Status

## 1. Backbone Robustness Matrix

Canonical governance entry:

```text
docs/BACKBONE_U5_MATRIX.md
results/backbone_u5_matrix_v1/backbone_u5_summary.csv
```

| Backbone | Status | Coverage | Evidence Tier | Results |
| :--- | :--- | :--- | :--- | :--- |
| ResNet1D | Done | 20 datasets / 60 pairs | canonical Final20 | Main host, +0.0405 vs no_aug |
| ModernTCN | Done | 20 datasets / 60 pairs | rebuilt Final20 | Robustness, +0.0696 vs no_aug |
| MiniRocket | Done with caveat | 19 datasets / 57 pairs | best-available recovery | Robustness, +0.0089 vs no_aug |
| PatchTST | Done with caveat | 20 datasets / 59 pairs | best-available recovery | Robustness, +0.0243 vs no_aug |
| TimesNet | Done with caveat | 19 datasets / 55 pairs | best-available recovery | Robustness, +0.0906 vs no_aug |
| MPTSNet | Pilot/probe only | Pilot7 / 21 pairs | pilot_only_u5 | Not Final20 paper evidence |

## 2. Mechanism Evidence (Covariance Control)
- [x] MiniRocket (CSTA vs Random/PCA): Done
- [ ] ModernTCN (CSTA vs Random/PCA): TBD
