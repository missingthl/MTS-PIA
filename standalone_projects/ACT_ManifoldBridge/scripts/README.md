# Scripts

This directory contains the runnable experiment and summary entrypoints for the
ACT/CSTA project.  The historical names are kept for compatibility, but the
current script roles are broader than the original release-only tree.

For the full operational map, start with:

```text
docs/WORKFLOW.md
```

## Workflow Readiness

- `check_workflow_readiness.py`: read-only preflight check for canonical
  entrypoints, locked Phase1/Phase2 roots, Final20 roots, and E1 coverage
  artifacts.

Run:

```bash
python standalone_projects/ACT_ManifoldBridge/scripts/check_workflow_readiness.py
```

## Canonical Single Run

- `../run_act_pilot.py`: single CSTA/PIA/MBA experiment entrypoint.

## Archived MBA/CSTA Release Matrix

The pre-U5 MBA/RC4 release-era runner has moved to:

```text
archive/release_legacy/scripts/
```

It is kept for provenance only and is not part of the current workflow.

## External Baseline Matrix

- `run_external_baselines_phase1.py`: general external baseline runner.  Despite
  the historical name, this now dispatches Phase 1, Phase 2, Phase 3, and CSTA
  sampling arms.
- `list_external_baselines.py`: prints the external baseline catalog from
  `utils/external_baseline_manifest.py`.  By default it shows only active E1
  methods, active controls, and the SPG-CFM future branch; pass
  `--include-archived` to show frozen probes.

Useful lookup:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py
```

Archived/probe lookup:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py --include-archived
```

## Protocol And Diagnostics

- `build_e1_main_artifacts.py`: builds the E1 atomic run table, method registry,
  dataset registry, cost audit, artifact audit, main table, and data audit doc.
- `build_csta_protocol_summary.py`: unified external/CSTA protocol summaries.
- `build_step3_diagnostic_report.py`: Step3 gamma/eta diagnostic report.
- `final_audit_report.py`: compact audit reports used during CSTA sampling.
- `audit_safe_step.py`, `audit_step3_results.py`, `generate_advantage_map.py`:
  targeted diagnostic helpers.

Post-U5 exploratory mechanism report builders and launchers have moved to:

```text
archive/mechanism_probes/scripts/
```

That archive includes AG-PIA, CS-Flow, latent residual, task-guided latent,
LC latent, SPG, ECL/RN-ECL, SPG-CFM, AO, selector-ablation, and direction
specificity probes.

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
