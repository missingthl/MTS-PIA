# ACL v1 Frozen Conclusion

## Status

`acl_small_matrix_v1` is the frozen result package for `ACT_ManifoldBridge` ACL v1.

- Frozen main configuration:
  - `pipeline=gcg_acl`
  - `model=resnet1d`
  - `algo=lraes`
  - `acl_candidates_per_anchor=4`
  - `acl_positives_per_anchor=1`
  - `acl_alignment_weight=0.7`
  - `acl_loss_weight=0.2`
  - `acl_temperature=0.07`
- `gcg_acl_n8` is retained only as a sensitivity arm and is not the default experimental setting.
- `continue_ce` is treated as the logical control arm derived from `gcg_acl_n4/base_f1`; it is not rerun as a separate experiment.

## Frozen Readout

- Official result entrypoints:
  - `_summary/arm_long.csv`
  - `_summary/dataset_summary.csv`
  - `_summary/gate_report.json`
- Matrix scope: `8 datasets x 3 seeds`
- Execution summary from `_summary/gate_report.json`:
  - `expected_actual_runs = 24`
  - `observed_seed_rows = 72`
  - `all_status_success = true`
  - `audit_files_complete = true`
  - `structure_pass_count = 3`
  - `volatility_stability_pass_count = 2`
  - `full_matrix_gate_pass = true`

## Stage Position

GCG + ACL v1 已完成工程闭环和首轮机制验证，默认配置可冻结为 n4 + top1 positive；方法在 NATOPS、JapaneseVowels、Heartbeat 上给出明确正向信号，在 Libras 上显示中间态，在 Handwriting、AtrialFibrillation、MotorImagery 上仍需机制复盘。

## Dataset-Level Readout

- Positive signal:
  - `natops`: `gcg_acl=0.9411`, `continue_ce=0.8986`, `mba=0.9358`
  - `japanesevowels`: `gcg_acl=0.9779`, `continue_ce=0.9616`, `mba=0.9708`
  - `heartbeat`: `gcg_acl=0.6560`, `continue_ce=0.6070`, `mba=0.6395`
- Intermediate:
  - `libras`: `gcg_acl=0.8737`, `continue_ce=0.8476`, `mba=0.8993`
- Needs mechanism review:
  - `handwriting`: `gcg_acl=0.4216`, `continue_ce=0.4254`, `mba=0.4573`
  - `atrialfibrillation`: `gcg_acl=0.1394`, `continue_ce=0.2618`, `mba=0.2513`
  - `motorimagery`: `gcg_acl=0.4537`, `continue_ce=0.5118`, `mba=0.4930`

## Freeze Rule

- This package is the sole frozen reference source for ACL v1 follow-up work.
- Follow-up review and stability confirmation should read from this package instead of rewriting its schema or rerunning its default matrix.
