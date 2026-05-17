# ACT/CSTA Workflow

This document is the operational map for the stabilized ACT/CSTA project.  It
does not introduce new methods.  It explains which entrypoints are canonical,
which artifacts support paper claims, and how to keep local experiments
auditable.

## Scope

The workflow is scoped to:

```text
standalone_projects/ACT_ManifoldBridge/
```

Adjacent standalone projects and root-level scripts are out of scope unless a
future migration explicitly pulls them into ACT.

## Workflow Layers

| Layer | Purpose | Primary entrypoint | Primary artifacts |
| --- | --- | --- | --- |
| W0 Readiness | Verify key entrypoints and locked roots before work | `scripts/check_workflow_readiness.py` | `results/workflow_readiness_v1/workflow_readiness_report.md` |
| W1 Single CSTA Run | One dataset/seed smoke or debug run | `run_act_pilot.py` | local `*_results.csv`, candidate audit |
| W2 External/E1 Matrix | External baselines and CSTA arms through one runner | `scripts/run_external_baselines_phase1.py` | `per_seed_external.csv`, summaries |
| W3 E1 Table Build | Convert run atoms into E1 paper/audit tables | `scripts/build_e1_main_artifacts.py` | `results/e1_main/*`, `docs/E1_DATA_AUDIT.md` |
| W4 Canonical U5 Final20 | Proposed method canonical table component | `scripts/run_csta_pia_final20.sh` | `results/csta_pia_final20/resnet1d_s123/per_seed_external.csv` |
| W5 External Final20 Addenda | Fill external method coverage gaps | dedicated shell/script roots | `final20_*`, `wdba_final20`, local generative roots |
| W6 Backbone Robustness | Host-model robustness evidence | backbone-specific launchers and builders | `docs/BACKBONE_U5_MATRIX.md`, `results/backbone_u5_matrix_v1/*` |
| W7 Mechanism Evidence | Non-mainline diagnostics/probes | `build_*_report.py`, audit scripts | mechanism reports; never main E1 rows |

## W0 Readiness

Run this before formal matrix work:

```bash
cd /home/THL/project/MTS-PIA
python standalone_projects/ACT_ManifoldBridge/scripts/check_workflow_readiness.py
```

The check is read-only.  It validates:

```text
key scripts and docs exist
locked Phase1/Phase2 row counts are unchanged
canonical U5/wDBA/no_aug-style roots have expected row counts
E1 artifacts exist and expose method coverage
known subset methods are explicitly marked as incomplete
```

## W1 Single CSTA Run

Use this only for smoke/debug.  Keep output under `/tmp` or a clearly local
root.

```bash
source /home/THL/miniconda3/etc/profile.d/conda.sh
conda activate pia
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops \
  --pipeline act \
  --algo zpia_top1_pool \
  --model resnet1d \
  --seeds 1 \
  --epochs 1 \
  --batch-size 64 \
  --lr 1e-3 \
  --patience 1 \
  --val-ratio 0.2 \
  --k-dir 10 \
  --pia-gamma 0.1 \
  --eta-safe 0.75 \
  --multiplier 1 \
  --template-selection topk_uniform_top5 \
  --out-root /tmp/csta_u5_smoke
```

Canonical U5 means:

```text
algo = zpia_top1_pool
template_selection = topk_uniform_top5
gamma = 0.1
eta_safe = 0.75
multiplier = 10 for formal runs
k_dir = 10
```

## W2 External/E1 Matrix Runner

Despite its historical name, this is the unified matrix runner for:

```text
external baselines
CSTA internal arms
generation-engine probes
backbone variants
```

Smoke:

```bash
python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
  --datasets natops \
  --seeds 1 \
  --arms no_aug,raw_mixup,csta_topk_uniform_top5 \
  --epochs 1 \
  --batch-size 64 \
  --patience 1 \
  --multiplier 1 \
  --out-root /tmp/e1_runner_smoke \
  --device cuda \
  --fail-fast
```

Do not write formal probes into locked roots:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
results/csta_external_baselines_phase2/resnet1d_s123/
```

The runner guards those roots unless explicitly overridden.

## W3 E1 Table Build

Build auditable E1 atoms and derived tables from existing run outputs:

```bash
python standalone_projects/ACT_ManifoldBridge/scripts/build_e1_main_artifacts.py
```

Outputs:

```text
results/e1_main/per_seed_e1_runs.csv
results/e1_main/e1_method_registry.csv
results/e1_main/e1_dataset_registry.csv
results/e1_main/e1_aug_artifacts.csv
results/e1_main/e1_cost_audit.csv
results/e1_main/e1_main_table.csv
results/e1_main/e1_main_table.md
docs/E1_DATA_AUDIT.md
```

E1 excludes internal controls such as `random_cov_state` and `pca_cov_state`.
They belong in mechanism/control sections, not the main external-baseline table.

## W4 Canonical U5 Final20

Canonical U5 evidence root:

```text
results/csta_pia_final20/resnet1d_s123/per_seed_external.csv
```

Expected coverage:

```text
20 datasets x 3 seeds = 60 rows
method = csta_topk_uniform_top5
```

If a new run is needed, use a new local root first.  Only replace canonical
roots after a dedicated audit.

## W5 External Final20 Addenda

Current E1 status:

```text
Full Final20:
  no_aug
  raw_aug_jitter
  raw_mixup
  dba_sameclass
  wdba_sameclass
  csta_topk_uniform_top5

Subset coverage:
  raw_aug_timewarp
  diffusionts_classwise
  rgw_sameclass
  dgw_sameclass

Registered but missing E1 rows:
  timegan_classwise
```

Subset methods must stay marked as subset until they reach `20 datasets x 3
seeds`.

## W6 Backbone Robustness

Use this only after ResNet1D canonical claims are stable.

Primary evidence:

```text
docs/BACKBONE_U5_MATRIX.md
results/backbone_u5_matrix_v1/backbone_u5_summary.csv
```

Backbone adapters are ACT-local under `core/` and `utils/`; they are not
guaranteed mirrors of root-level `models/`.

## W7 Mechanism Evidence

Mechanism exploration is useful, but it is not E1.

Examples:

```text
local tangent audit
direction specificity stress
AG/CS-Flow/latent/SPG probes
RandomCov/PCACov controls
```

Most post-U5 mechanism probe launchers and report builders now live under:

```text
archive/mechanism_probes/scripts/
```

The strongest non-U5 probe was `task_guided_latent_residual_flow`: it beat U5
on Pilot3 but failed on Pilot7, so it remains an archived near-mainline negative
result rather than an active workflow branch.

The only retained future-branch family is SPG-CFM:

```text
spg_cfm_one_step
spg_cfm_align_one_step
```

It remains non-canonical, but it is the most complete next-generation skeleton:
support-projected task gradients plus conditional flow matching.  All other
post-U5 generation-engine probes are hidden from the default method listing and
require `list_external_baselines.py --include-archived`.

Rules:

```text
do not mix mechanism controls into E1
do not treat debug probes as paper baselines
do not overwrite canonical roots with mechanism roots
always report subset coverage and negative diagnostics
```

## Result Hygiene

Use these conventions:

```text
/tmp/...                         smoke and disposable checks
results/*_v1/...                 local experiment root
results/e1_main/...              generated E1 audit artifacts
results/csta_pia_final20/...     canonical U5 component
results/wdba_final20/...         canonical wDBA component
```

Avoid writing large `_csta_runs`, `_shards`, or raw logs into release-oriented
docs.  Keep them local unless a paper claim requires a specific artifact.
