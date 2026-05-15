# E1 Data Audit

Generated: 2026-05-15T14:19:15
Experiment tag: `E1_main_v1`

## Method Coverage Matrix

| Method | Final20 datasets with 3 seeds | Total rows | Missing runs |
| --- | ---: | ---: | --- |
| `no_aug` | 20/20 | 60 |  |
| `raw_aug_jitter` | 20/20 | 60 |  |
| `raw_aug_timewarp` | 7/20 | 21 | articularywordrecognition:s1,2,3; basicmotions:s1,2,3; cricket:s1,2,3; epilepsy:s1,2,3; ethanolconcentration:s1,2,3; fingermovements:s1,2,3; har:s1,2,3; heartbeat:s1,2,3 ... |
| `raw_mixup` | 20/20 | 60 |  |
| `timegan_classwise` | 0/20 | 0 | articularywordrecognition:s1,2,3; atrialfibrillation:s1,2,3; basicmotions:s1,2,3; cricket:s1,2,3; epilepsy:s1,2,3; ering:s1,2,3; ethanolconcentration:s1,2,3; fingermovements:s1,2,3 ... |
| `diffusionts_classwise` | 7/20 | 21 | articularywordrecognition:s1,2,3; basicmotions:s1,2,3; cricket:s1,2,3; epilepsy:s1,2,3; ethanolconcentration:s1,2,3; fingermovements:s1,2,3; har:s1,2,3; heartbeat:s1,2,3 ... |
| `dba_sameclass` | 20/20 | 60 |  |
| `wdba_sameclass` | 20/20 | 60 |  |
| `rgw_sameclass` | 7/20 | 21 | articularywordrecognition:s1,2,3; basicmotions:s1,2,3; cricket:s1,2,3; epilepsy:s1,2,3; ethanolconcentration:s1,2,3; fingermovements:s1,2,3; har:s1,2,3; heartbeat:s1,2,3 ... |
| `dgw_sameclass` | 7/20 | 21 | articularywordrecognition:s1,2,3; basicmotions:s1,2,3; cricket:s1,2,3; epilepsy:s1,2,3; ethanolconcentration:s1,2,3; fingermovements:s1,2,3; har:s1,2,3; heartbeat:s1,2,3 ... |
| `csta_topk_uniform_top5` | 20/20 | 60 |  |

## Dataset Coverage Matrix

| Dataset | Methods complete at 3 seeds |
| --- | ---: |
| `articularywordrecognition` | 6/11 |
| `atrialfibrillation` | 10/11 |
| `basicmotions` | 6/11 |
| `cricket` | 6/11 |
| `epilepsy` | 6/11 |
| `ering` | 10/11 |
| `ethanolconcentration` | 6/11 |
| `fingermovements` | 6/11 |
| `handmovementdirection` | 10/11 |
| `handwriting` | 10/11 |
| `har` | 6/11 |
| `heartbeat` | 6/11 |
| `japanesevowels` | 10/11 |
| `libras` | 6/11 |
| `motorimagery` | 6/11 |
| `natops` | 10/11 |
| `pendigits` | 6/11 |
| `racketsports` | 10/11 |
| `selfregulationscp2` | 6/11 |
| `uwavegesturelibrary` | 6/11 |

## Missing Runs

- `raw_aug_timewarp` / `articularywordrecognition` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `basicmotions` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `cricket` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `epilepsy` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `ethanolconcentration` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `fingermovements` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `har` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `heartbeat` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `libras` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `motorimagery` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `pendigits` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `selfregulationscp2` missing seeds: 1, 2, 3
- `raw_aug_timewarp` / `uwavegesturelibrary` missing seeds: 1, 2, 3
- `timegan_classwise` / `articularywordrecognition` missing seeds: 1, 2, 3
- `timegan_classwise` / `atrialfibrillation` missing seeds: 1, 2, 3
- `timegan_classwise` / `basicmotions` missing seeds: 1, 2, 3
- `timegan_classwise` / `cricket` missing seeds: 1, 2, 3
- `timegan_classwise` / `epilepsy` missing seeds: 1, 2, 3
- `timegan_classwise` / `ering` missing seeds: 1, 2, 3
- `timegan_classwise` / `ethanolconcentration` missing seeds: 1, 2, 3
- `timegan_classwise` / `fingermovements` missing seeds: 1, 2, 3
- `timegan_classwise` / `handmovementdirection` missing seeds: 1, 2, 3
- `timegan_classwise` / `handwriting` missing seeds: 1, 2, 3
- `timegan_classwise` / `har` missing seeds: 1, 2, 3
- `timegan_classwise` / `heartbeat` missing seeds: 1, 2, 3
- `timegan_classwise` / `japanesevowels` missing seeds: 1, 2, 3
- `timegan_classwise` / `libras` missing seeds: 1, 2, 3
- `timegan_classwise` / `motorimagery` missing seeds: 1, 2, 3
- `timegan_classwise` / `natops` missing seeds: 1, 2, 3
- `timegan_classwise` / `pendigits` missing seeds: 1, 2, 3
- `timegan_classwise` / `racketsports` missing seeds: 1, 2, 3
- `timegan_classwise` / `selfregulationscp2` missing seeds: 1, 2, 3
- `timegan_classwise` / `uwavegesturelibrary` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `articularywordrecognition` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `basicmotions` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `cricket` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `epilepsy` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `ethanolconcentration` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `fingermovements` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `har` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `heartbeat` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `libras` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `motorimagery` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `pendigits` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `selfregulationscp2` missing seeds: 1, 2, 3
- `diffusionts_classwise` / `uwavegesturelibrary` missing seeds: 1, 2, 3
- `rgw_sameclass` / `articularywordrecognition` missing seeds: 1, 2, 3
- `rgw_sameclass` / `basicmotions` missing seeds: 1, 2, 3
- `rgw_sameclass` / `cricket` missing seeds: 1, 2, 3
- `rgw_sameclass` / `epilepsy` missing seeds: 1, 2, 3
- `rgw_sameclass` / `ethanolconcentration` missing seeds: 1, 2, 3
- `rgw_sameclass` / `fingermovements` missing seeds: 1, 2, 3
- `rgw_sameclass` / `har` missing seeds: 1, 2, 3
- `rgw_sameclass` / `heartbeat` missing seeds: 1, 2, 3
- `rgw_sameclass` / `libras` missing seeds: 1, 2, 3
- `rgw_sameclass` / `motorimagery` missing seeds: 1, 2, 3
- `rgw_sameclass` / `pendigits` missing seeds: 1, 2, 3
- `rgw_sameclass` / `selfregulationscp2` missing seeds: 1, 2, 3
- `rgw_sameclass` / `uwavegesturelibrary` missing seeds: 1, 2, 3
- `dgw_sameclass` / `articularywordrecognition` missing seeds: 1, 2, 3
- `dgw_sameclass` / `basicmotions` missing seeds: 1, 2, 3
- `dgw_sameclass` / `cricket` missing seeds: 1, 2, 3
- `dgw_sameclass` / `epilepsy` missing seeds: 1, 2, 3
- `dgw_sameclass` / `ethanolconcentration` missing seeds: 1, 2, 3
- `dgw_sameclass` / `fingermovements` missing seeds: 1, 2, 3
- `dgw_sameclass` / `har` missing seeds: 1, 2, 3
- `dgw_sameclass` / `heartbeat` missing seeds: 1, 2, 3
- `dgw_sameclass` / `libras` missing seeds: 1, 2, 3
- `dgw_sameclass` / `motorimagery` missing seeds: 1, 2, 3
- `dgw_sameclass` / `pendigits` missing seeds: 1, 2, 3
- `dgw_sameclass` / `selfregulationscp2` missing seeds: 1, 2, 3
- `dgw_sameclass` / `uwavegesturelibrary` missing seeds: 1, 2, 3

## Budget Consistency

- No target/actual augmentation-ratio mismatch detected in available rows.

## Cost Field Availability

| Method | Rows | aug_cost | generator_fit | sample_gen | dtw_alignment | cov_state | bridge | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_aug` | 60 | 60 | 0 | 0 | 0 | 0 | 0 | 39 |
| `raw_aug_jitter` | 60 | 60 | 0 | 0 | 0 | 0 | 0 | 60 |
| `raw_aug_timewarp` | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `raw_mixup` | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 60 |
| `timegan_classwise` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `diffusionts_classwise` | 21 | 21 | 0 | 0 | 0 | 0 | 0 | 21 |
| `dba_sameclass` | 60 | 60 | 0 | 0 | 60 | 0 | 0 | 60 |
| `wdba_sameclass` | 60 | 60 | 0 | 0 | 60 | 0 | 0 | 60 |
| `rgw_sameclass` | 21 | 21 | 0 | 0 | 21 | 0 | 0 | 21 |
| `dgw_sameclass` | 21 | 21 | 0 | 0 | 21 | 0 | 0 | 21 |
| `csta_topk_uniform_top5` | 60 | 60 | 0 | 0 | 0 | 0 | 0 | 60 |

## W/T/L Consistency

- No-Aug: W/T/L=0/20/0, paired dataset count=20
- Jitter: W/T/L=12/2/6, paired dataset count=20
- Mixup: W/T/L=13/1/6, paired dataset count=20
- DBA: W/T/L=15/1/4, paired dataset count=20
- wDBA: W/T/L=17/1/2, paired dataset count=20
- CoSTA-U5: W/T/L=17/1/2, paired dataset count=20

## Claim Support Checklist

- RandomCov and PCACov are excluded from E1 and remain internal controls.
- TimeVAE is excluded from E1.
- Mixup is marked as training-time soft-label vicinal training; offline augmentation cost can be N/A.
- TimeGAN currently has method metadata but no completed E1 rows in this workspace.
- Diffusion-TS rows are subset coverage and must not be described as Final20 full unless completed.
- RGW/DGW rows are subset coverage and clean-room adapters; do not call them official reproductions.
- CoSTA-U5 uses canonical Final20 rows when available.
