# Scripts

This directory contains the runnable experiment and summary entrypoints for the
ACT/CSTA project.  The historical names are kept for compatibility, but the
current script roles are broader than the original release-only tree.

## Canonical Single Run

- `../run_act_pilot.py`: single CSTA/PIA/MBA experiment entrypoint.

## Internal MBA/CSTA Release Matrix

- `run_mba_vs_rc4_matrix.py`: queue runner for internal comparison arms.
- `summarize_mba_vs_rc4_matrix.py`: summary table generator for that matrix.

## External Baseline Matrix

- `run_external_baselines_phase1.py`: general external baseline runner.  Despite
  the historical name, this now dispatches Phase 1, Phase 2, Phase 3, and CSTA
  sampling arms.
- `list_external_baselines.py`: prints the external baseline catalog from
  `utils/external_baseline_manifest.py`.

Useful lookup:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py
```

## Protocol And Diagnostics

- `build_csta_protocol_summary.py`: unified external/CSTA protocol summaries.
- `build_step3_diagnostic_report.py`: Step3 gamma/eta diagnostic report.
- `final_audit_report.py`: compact audit reports used during CSTA sampling.
- `audit_safe_step.py`, `audit_step3_results.py`, `generate_advantage_map.py`:
  targeted diagnostic helpers.

## Shell Launchers

- `run_csta_sampling_v1.sh`
- `run_csta_step3_diagnostic_sweep.sh`
- `run_csta_pia_final20.sh`
- `run_csta_neurips_ablation.sh`
- `run_wdba_final20.sh`

Keep smoke/debug outputs outside locked result roots.  Prefer `/tmp/...` or a
clearly named local-only root for probes.

The external runner refuses to write to locked Phase 1/2 roots unless
`--allow-locked-root-overwrite` is supplied.  Use that flag only when
intentionally regenerating locked reference summaries.
