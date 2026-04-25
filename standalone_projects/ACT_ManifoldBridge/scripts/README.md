# Scripts

Scripts in this folder support the canonical entrypoint
[`../run_act_pilot.py`](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/run_act_pilot.py).

## Current Active Scripts

- `run_v2_grand_sweep.sh`: ACT-V2 RC grand sweep launcher.
- `run_mba_vs_rc4_matrix.py`: 4-GPU queue runner for the 20-dataset MBA vs RC-4 census.
- `summarize_mba_vs_rc4_matrix.py`: normalize the 3 actual arms into 4 logical arms and emit reproduction / drift / gap reports.
- `aggregate_v2_grand_sweep.py`: aggregate per-dataset result files into one summary.
- `analyze_v2_taxonomy.py`: summarize emerging regime taxonomy.
- `generate_paper_assets.py`: generate paper-facing assets.
- `gen_paper_tables.py`: produce paper tables.
- `gamma_scan.py`: gamma sensitivity utility.
- `viz_manifold.py`: latent-space visualization.
- `viz_theory_stats.py`: theory-stat plots.
- `viz_time_series.py`: waveform visualization.

## Historical / Legacy

`legacy/` contains older one-off sweep and evidence helpers kept only for
reproducibility. They are not the current project API.
