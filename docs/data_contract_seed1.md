# SEED_1 Data Contract

This document defines the canonical data roots and asset usage for SEED_1. All future runs must use these paths to avoid data drift.

## Source Of Truth

- seed_root: `data/SEED`

## Asset Contract

| Asset | Root | Sample Unit | Labels | Primary Use |
| --- | --- | --- | --- | --- |
| Raw CNT | `data/SEED/SEED_EEG/SEED_RAW_EEG` | trial -> window | trial label from `SEED_stimulation.xlsx` + `time.txt` | raw -> cov_spd -> TSM pipeline |
| Raw FIF | `data/SEED/SEED_EEG/SEED_RAW_FIF` | trial -> window | same as CNT | speed/IO for raw pipeline |
| Official MAT (1s) | `data/SEED/SEED_EEG/ExtractedFeatures_1s` | trial | trial label | subject-split baseline (LogReg/SVM) |
| Official MAT (4s) | `data/SEED/SEED_EEG/ExtractedFeatures_4s` | trial | trial label | subject-split baseline (LogReg/SVM) |
| Multimodal NPZ (1s) | `data/SEED/SEED_Multimodal/Chinese/02-EEG-DE-feature/eeg_used_1s` | author protocol (clip/window) | embedded in npz | feature alignment sanity tests |
| Multimodal NPZ (4s) | `data/SEED/SEED_Multimodal/Chinese/02-EEG-DE-feature/eeg_used_4s` | author protocol (clip/window) | embedded in npz | feature alignment sanity tests |

## Default Roots (Use These)

- raw_root = `data/SEED/SEED_EEG/SEED_RAW_EEG`
- raw_fif_root = `data/SEED/SEED_EEG/SEED_RAW_FIF`
- mat_root_1s = `data/SEED/SEED_EEG/ExtractedFeatures_1s`
- mat_root_4s = `data/SEED/SEED_EEG/ExtractedFeatures_4s`
- npz_root_1s = `data/SEED/SEED_Multimodal/Chinese/02-EEG-DE-feature/eeg_used_1s`
- npz_root_4s = `data/SEED/SEED_Multimodal/Chinese/02-EEG-DE-feature/eeg_used_4s`

## Prohibited / Guardrails

- Do not read raw EEG from `data/SEED/SEED_Multimodal/Chinese/01-EEG-raw` unless coverage is explicitly validated and a switch is documented.
- Training scripts must not mix data sources across different roots without updating this contract.

## Notes

- Trial slicing uses `time.txt` + `SEED_stimulation.xlsx`.
- Any change to offset or smoothing parameters must be recorded in run_config.json and linked back to this contract.
