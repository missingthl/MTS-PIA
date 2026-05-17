# P0.5 Backbone Table Freeze Audit Report

## 1. Quantitative Integrity
| Backbone | Datasets | Pairs (N) | Mean Delta | W/T/L | Math Consistency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet1D | 20 | 60 | +0.0413 | 39/7/14 | PASS |
| ModernTCN | 20 | 60 | +0.0696 | 46/2/12 | PASS |
| MiniRocket | 18 | 54 | +0.0100 | 25/14/15 | PASS |

## 2. Compliance Checklist
- [x] **ResNet1D Boundary**: Strictly filtered to Final20. (Current: 20/20 datasets)
- [x] **ModernTCN Atomic Reconstruction**: Verified 60/60 aligned pairs from physical shards.
- [x] **MiniRocket Labeling**: Explicitly identified as 18/20 Subset (N=54).
- [x] **Method Unification**: All backbones mapped to `csta_topk_uniform_top5`.
- [x] **Pairwise Calculation**: ΔF1 derived from `mean(csta - no_aug)` per sample.

## 3. Claim Evidence Matrix
### Supported Claims
- **Model-Agnosticism**: CSTA-U5 provides positive gain across ResNet, ModernTCN, and MiniRocket (all ΔF1 > 0).
- **Deep Learning Priority**: The gain in neural networks (4-7%) is significantly higher than in linear kernels (1%), supporting the manifold-regularization theory.
- **Robustness**: 76.7% win-rate on ModernTCN confirms stability in SOTA architectures.

### NOT Supported / Limitations
- **Variable-Length Compatibility**: Cannot claim support for variable-length series on MiniRocket (excluded).
- **Mechanism Supremacy on Linear Models**: CSTA is NOT significantly better than PCA/Random on MiniRocket; mechanism advantage is limited to deep non-linear models.
