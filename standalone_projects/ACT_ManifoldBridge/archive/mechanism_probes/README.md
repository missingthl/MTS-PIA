# Mechanism Probe Archive

This archive keeps report builders and launchers for exploratory mechanisms
that were tested after CSTA-U5 was selected.  They are not part of the current
paper-facing workflow.

Current main workflow:

```text
docs/WORKFLOW.md
scripts/check_workflow_readiness.py
scripts/run_external_baselines_phase1.py
scripts/build_e1_main_artifacts.py
scripts/run_csta_pia_final20.sh
```

## Candidate Summary

| Candidate | Best signal | Final decision |
| --- | --- | --- |
| `task_guided_latent_residual_flow` | Pilot3: `0.945059`, above U5 `0.942395` on NatOps/JapaneseVowels/RacketSports | Pilot7: `0.646855`, below U5 `0.665242` and random-cov `0.648130`; archive as near-mainline negative probe. |
| `cs_flow_single_step` | Pilot3: `0.950717`, strongest Pilot3 signal | Pilot7 fresh: `0.636248`, below U5/random; archive as deterministic-flow negative probe. |
| `spg_pia_zhead` | Pilot3: `0.942051`, close to U5 `0.942395` | Pilot7: `0.639050`, below U5/random; archive. |
| `latent_residual_flow` | Fixed CS-Flow direction collapse | Pilot7: `0.647500`, below U5 and near random-cov; archive. |
| `lc_latent_residual_flow` | Reduced task-guidance label-breaking mass | Pilot3 lower than task-guided/U5; archive. |
| AG/ECL/RN-ECL/SPG-CFM variants | Useful diagnostics | Did not beat U5 on their locked comparison scopes; archive. |

## Rule

Do not restart these probes from the main `scripts/` directory.  If a probe must
be reproduced, run it from this archive with a new local result root and record
why the frozen U5 decision is being revisited.
