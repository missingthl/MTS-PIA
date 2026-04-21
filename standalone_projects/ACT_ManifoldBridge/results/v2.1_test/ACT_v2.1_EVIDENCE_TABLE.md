# ACT v2.1 Analysis: Evidence Table

> [!NOTE]
> This table summarizes the performance of ACT v2.1 across different regimes. 
> `v2.1 (τ=0)` represents the proposed 'Augmentation Feedback Control' strategy.

| Dataset | Arm | Base F1 (avg) | ACT F1 (avg) | Gain % (avg) | Mean w_aug | Zero Weight % |
| --- | --- | --- | --- | --- | --- | --- |
| atrialfibrillation | Pure | 0.1973 | 0.1732 | -1.76% | 1.000 | 0.0% |
| atrialfibrillation | Hybrid (v1) | 0.2644 | 0.2146 | -18.80% | 1.000 | 0.0% |
| atrialfibrillation | v2.1 (τ=0) | 0.2602 | 0.1707 | -33.69% | 0.007 | 94.4% |
| atrialfibrillation | v2.1 (τ=0.1) | 0.1907 | 0.1429 | -24.92% | 0.007 | 94.4% |
| heartbeat | Hybrid (v1) | 0.6679 | 0.6474 | -3.03% | 1.000 | 0.0% |
| heartbeat | v2.1 (τ=0) | 0.6334 | 0.6141 | -3.04% | 0.002 | 91.4% |