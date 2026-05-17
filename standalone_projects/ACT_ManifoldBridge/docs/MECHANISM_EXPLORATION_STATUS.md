# Mechanism Exploration Status

更新时间：2026-05-13

This page separates the frozen CSTA-U5 mainline from exploratory generation
mechanisms.  It is an engineering guardrail: code may remain available for
audit and reproduction, but availability does not mean a method is canonical.

## Canonical Mainline

```text
csta_topk_uniform_top5
```

Frozen configuration:

```text
gamma = 0.1
eta_safe = 0.75
multiplier = 10
k_dir = 10
backbone = resnet1d
```

Paper-facing result entrypoints:

```text
results/CANONICAL_RESULTS.md
docs/EXPERIMENT_MATRIX_INDEX.md
docs/BACKBONE_U5_MATRIX.md
```

Do not replace this mainline without a new locked result document.

## CSTA Controls

These controls are useful for mechanism interpretation and should remain
available:

| Method | Purpose |
| --- | --- |
| `csta_top1_current` | Greedy highest-response PIA template control. |
| `csta_template_random_within_bank` | Random direction inside TELM2/PIA support bank. |
| `random_cov_state` | Full covariance-state random direction control. |
| `pca_cov_state` | Global PCA covariance-state control. |

The distinction between bank-random and full random is important.  Do not merge
their interpretation.

## Exploratory Generation Engines

The following modules are exploratory and should not be read as canonical
methods unless promoted by a locked result table.

| Family | Main Code | Current Status |
| --- | --- | --- |
| AG-PIA | `core/csta/ag_pia.py` | Direction-generation probe; not promoted over U5. |
| CS-Flow | `core/csta/cs_flow.py` | Pilot3 positive, Pilot7 failed to beat U5/wDBA; retained as diagnostic. |
| Latent Residual Flow | `core/csta/latent_residual_flow.py` | Fixed CS-Flow direction collapse, but did not beat U5. |
| Task-Guided Latent Residual | `core/csta/task_guided_latent_residual_flow.py` | Pilot3 promising; Pilot7 under U5/random, task utility assigned high label-breaking mass. |
| Label-Consistent Latent Residual | `core/csta/lc_latent_residual_flow.py` | Reduced wrong-pred mass but became too conservative; not promoted. |
| SPG-PIA | `core/csta/spg_pia.py` | Support-projected gradient signal is real, but deterministic version collapses directions; ECL/RN variants are exploratory. |
| SPG-CFM | `core/csta/spg_cfm.py` | Conditional-flow mechanism exploration; not canonical. |

## Runner And Registry Rule

Exploratory methods are still registered so old pilot roots can be reproduced.
The runner path is:

```text
scripts/run_external_baselines_phase1.py
  -> utils/csta_method_commands.py
  -> run_act_pilot.py
  -> core/csta/pipelines.py
```

Generation-engine methods must not be routed through template selection:

```text
ag_*
cs_flow_*
latent_residual_*
task_guided_*
lc_*
spg_*
ecl_spg_*
rn_ecl_spg_*
gi_spg_*
spg_cfm_*
```

Their candidate rows use placeholder template fields:

```text
template_id = -1
template_rank = -1
template_response_abs = NaN
direction_id = -1
```

## When Adding A New Mechanism

Do not add new method logic directly to `run_external_baselines_phase1.py`.

Update these locations instead:

```text
core/csta/<new_method>.py
core/csta/pipelines.py
core/csta/result_schema.py
utils/external_runner_registry.py
utils/csta_method_commands.py
scripts/build_<method>_report.py
docs/MECHANISM_EXPLORATION_STATUS.md
```

Before claiming a method is a mainline candidate, it must have:

```text
1. /tmp smoke success;
2. Pilot3 summary with diagnostics;
3. clear comparison against U5, random_cov_state, bank-random, and wDBA when relevant;
4. explicit collapse/risk diagnostics;
5. a documented decision on whether Pilot7 is warranted.
```

## Non-Goals

Current engineering cleanup should not:

```text
1. tune U5;
2. tune failed exploratory mechanisms;
3. add new selectors;
4. add new generation engines;
5. overwrite locked Phase1/Phase2 roots;
6. mix smoke/recovery roots into paper-facing claims.
```
