# Final20 CSTA-U5 vs wDBA: Formal Alignment Report

**Status**: OFFICIAL — based on real experiment outputs. Do not modify.
**Date**: 2026-05-03
**Config**: resnet1d, seeds 1/2/3, gamma=0.1, eta_safe=0.75, multiplier=10, k_dir=10

---

## 1. Overall Summary

| Metric | CSTA-U5 | wDBA | Delta (CSTA−wDBA) |
| :--- | :--- | :--- | :--- |
| **Mean Aug F1 (20DS)** | **0.7279** | **0.7300** | **−0.0021** |
| Mean Gain vs No Aug | +0.0405 | +0.0426 | −0.0021 |
| Dataset W / T / L | 10 / 1 / 9 | — | — |
| Seed W / T / L | 30 / 3 / 27 | — | — |
| Bootstrap CI (delta) | [−0.0183, +0.0135] | — | crosses zero = **True** |

**Conclusion**: CSTA-U5 and wDBA are **statistically indistinguishable** on Final20. The mean gap is −0.0021, and the 95% bootstrap CI crosses zero. **CSTA does NOT clearly outperform wDBA.**

---

## 2. Paper Wording Guardrails

### SAFE to write:
```
CSTA-U5 achieves competitive performance with the strongest DTW synthesis baseline (wDBA)
on the Final20 benchmark, with a mean F1 of 0.7279 vs 0.7300.

CSTA-U5 provides a substantial improvement over no augmentation (+0.0405 mean F1 gain across 20 datasets).

CSTA-U5 achieves near-wDBA performance while using a non-DTW, covariance-state augmentation
mechanism with candidate-level auditability.
```

### DO NOT write:
```
CSTA outperforms wDBA on Final20.
CSTA significantly exceeds wDBA.
CSTA is the best augmentation method.
```

---

## 3. Per-Dataset Win/Loss

Sorted by delta (CSTA − wDBA):

| Dataset | CSTA F1 | wDBA F1 | Delta | Result | CSTA Elapsed(s) | wDBA Elapsed(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| handwriting | 0.468 | 0.403 | +0.065 | **WIN** | 7 | 27 |
| uwavegesturelibrary | 0.790 | 0.741 | +0.049 | **WIN** | 7 | 53 |
| ering | 0.821 | 0.773 | +0.048 | **WIN** | 5 | 3 |
| motorimagery | 0.449 | 0.408 | +0.040 | **WIN** | 90 | **23318** |
| natops | 0.961 | 0.947 | +0.014 | **WIN** | 8 | 8 |
| epilepsy | 0.971 | 0.959 | +0.013 | **WIN** | 8 | 33 |
| articularywordrecognition | 0.980 | 0.971 | +0.009 | **WIN** | 11 | 32 |
| har | 0.955 | 0.952 | +0.004 | **WIN** | 178 | 502 |
| japanesevowels | 0.979 | 0.975 | +0.004 | **WIN** | 11 | 20 |
| pendigits | 0.985 | 0.984 | +0.001 | **WIN** | 137 | 232 |
| basicmotions | 1.000 | 1.000 | 0.000 | TIE | 6 | 10 |
| cricket | 0.981 | 0.986 | −0.005 | loss | 14 | **677** |
| racketsports | 0.888 | 0.897 | −0.009 | loss | 8 | 11 |
| ethanolconcentration | 0.274 | 0.284 | −0.010 | loss | 26 | **2613** |
| libras | 0.867 | 0.888 | −0.020 | loss | 7 | 10 |
| heartbeat | 0.655 | 0.676 | −0.021 | loss | 18 | **344** |
| fingermovements | 0.529 | 0.570 | −0.041 | loss | 10 | 14 |
| selfregulationscp2 | 0.464 | 0.506 | −0.043 | loss | 17 | **508** |
| atrialfibrillation | 0.269 | 0.311 | −0.043 | loss | 5 | 24 |
| handmovementdirection | 0.272 | 0.369 | **−0.097** | loss | 8 | **84** |

---

## 4. Efficiency Comparison

Key observation from elapsed time:

| Dataset | CSTA (s) | wDBA (s) | wDBA / CSTA |
| :--- | :--- | :--- | :--- |
| motorimagery | 90 | **23318** | **259×** |
| ethanolconcentration | 26 | **2613** | **100×** |
| selfregulationscp2 | 17 | **508** | **30×** |
| heartbeat | 18 | **344** | **19×** |
| har | 178 | 502 | 2.8× |
| cricket | 14 | 677 | 48× |

**CSTA is significantly faster than wDBA on high-cost datasets.**
On `motorimagery`, wDBA took 23,318 seconds (~6.5 hours) vs CSTA's 90 seconds (**259× speedup**).

---

## 5. Analysis: Where Does CSTA Win/Lose?

**CSTA wins on**: High-dimensional covariance-structured datasets (handwriting, ering, uwavegesture, motorimagery). These benefit from covariance-state augmentation.

**CSTA loses on**: Datasets where DTW averaging produces smoother prototypes that generalize well (handmovementdirection, atrialfibrillation, selfregulationscp2, fingermovements). These tend to be low-sample or highly variable datasets where class prototype averaging is beneficial.

---

## 6. Next Steps

- [ ] P2: Supplement `no_aug`, `random_cov_state`, `pca_cov_state`, `dba_sameclass`, `raw_aug_jitter` Final20
- [ ] P3: Run paired statistical test (Wilcoxon signed-rank) on seed-level deltas
- [ ] P4: Generate paper figures (framework diagram, mechanism triangle, efficiency comparison)
