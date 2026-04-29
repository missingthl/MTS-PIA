# Scripts

Only the release-maintained scripts are kept here.

- `run_mba_vs_rc4_matrix.py`: queue runner for the supported comparison arms:
  `mba_core_lraes`, `mba_core_rc4_fused_concat`, and
  `mba_core_zpia_top1_pool`.
- `summarize_mba_vs_rc4_matrix.py`: produces per-seed, per-dataset, and overall
  comparison summaries.
- `run_external_baselines_phase1.py`: isolated Phase 1 external-baseline runner
  for raw time-domain augmentation, Mixup, DBA, SMOTE, naive covariance-state
  controls, and the maintained CSTA arms. It writes its own summary files under
  `results/csta_external_baselines_phase1/` and does not add external baseline
  logic to `run_act_pilot.py`.

The canonical single-experiment entrypoint is
[`../run_act_pilot.py`](../run_act_pilot.py).

Historical one-off sweep, visualization, and paper-asset scripts were removed
from the release folder to keep the project focused and readable.
