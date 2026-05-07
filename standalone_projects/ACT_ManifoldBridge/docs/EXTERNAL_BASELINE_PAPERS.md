# External Baseline Paper Sources

PDFs were downloaded locally under:

```text
references/external_baselines/
```

That directory is intentionally ignored by git, so large PDFs do not enter
release commits.  This tracked index records the source mapping.

| Method arm(s) | Local PDF name | Source URL | Implementation note |
| --- | --- | --- | --- |
| `jobda_cleanroom` | `jobda_aaai2021.pdf` | https://cdn.aaai.org/ojs/17071/17071-13-20565-1-2-20210518.pdf | Clean-room TSW + joint-label implementation; no confirmed official code found. |
| `rgw_sameclass`, `dgw_sameclass` | `guided_warping_icpr2020.pdf` | https://arxiv.org/pdf/2004.08780.pdf | Clean-room RGW/DGW adapter based on guided warping. |
| `dba_sameclass` | `dba_petitjean2011.pdf` | https://francois-petitjean.com/Research/Petitjean2011-PR.pdf | Uses `tslearn` DBA. |
| `wdba_sameclass` | `wdba_forestier2017_icdm.pdf` | https://germain-forestier.info/publis/icdm2017.pdf | Weighted DBA / sparse time-series augmentation reference. |
| `spawner_sameclass_style` | `spawner_sensors2020.pdf` | https://pdfs.semanticscholar.org/9c64/0819865d83bfc24547ff1af563e5616d04ec.pdf | SPAWNER-style clean-room DTW aligned average + noise adapter. |
| `raw_smote_flatten_balanced` | `smote_chawla2002.pdf` | https://arxiv.org/pdf/1106.1813.pdf | Class-balancing baseline, not budget-matched CSTA augmentation. |
| `raw_mixup` | `mixup_zhang2017.pdf` | https://arxiv.org/pdf/1710.09412.pdf | Soft-label mixup baseline. |
| `timevae_classwise_optional` | `timevae_desai2021.pdf` | https://arxiv.org/pdf/2111.08095.pdf | PyTorch-style classwise adapter; not the official Keras pipeline. |
| `diffusionts_classwise` | `diffusionts_iclr2024.pdf` | https://proceedings.iclr.cc/paper_files/paper/2024/file/b5b66077d016c037576cc56a82f97f66-Paper-Conference.pdf | Wrapper around Diffusion-TS-style classwise generator. |
| `timevqvae_classwise` | `timevqvae_aistats2023.pdf` | https://arxiv.org/pdf/2303.04743.pdf | Wrapper hook for TimeVQVAE-style generation. |
| `raw_aug_jitter`, `raw_aug_scaling`, `raw_aug_timewarp`, `raw_aug_magnitude_warping`, `raw_aug_window_warping`, `raw_aug_window_slicing` | `time_series_aug_survey_ijcai2021.pdf` | https://www.ijcai.org/proceedings/2021/0631.pdf | General survey source for common raw-domain time-series augmentation transforms. |

Do not cite local paths in papers; cite the publication/source URL or canonical
bibliographic entry.  Local PDFs are only for project reading and audit support.
