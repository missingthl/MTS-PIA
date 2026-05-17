# Method Registry Governance

## Runtime Source of Truth

`utils/external_runner_registry.py` owns the runtime method registry:

- `METHOD_INFO`
- default phase arm lists
- locked-root guards
- CSTA method to `run_act_pilot.py --template-selection` mapping
- CSTA result passthrough fields

Runner scripts must import this registry instead of maintaining a second copy.

## Documentation Catalog

`utils/external_baseline_manifest.py` is a documentation and discovery layer. It stores conceptual fields such as method family, implementation file, runner entrypoint, and notes. Runtime fields are projected from `external_runner_registry.METHOD_INFO`, including:

- `source_space`
- `label_mode`
- `budget_matched`

If a method appears in the manifest but not in the runtime registry, importing the manifest raises an error.

## Adding A CSTA Internal Arm

Add the method to `METHOD_INFO` with `source_space="covariance_template"` and `label_mode="hard"`. Add its policy mapping in `csta_policy_for_method()` if it needs a specific `--template-selection`.

For example, `csta_template_random_within_bank` maps to `--template-selection random`. This means random sampling inside the train-only TELM2/PIA template bank, not full covariance-state random perturbation.

Then add a manifest row for documentation and run a smoke test through `scripts/run_external_baselines_phase1.py`.

AG-PIA arms are the exception to the template-policy rule. `ag_target_direct`,
`ag_pia_single`, and `ag_pia_multihead5` use
`source_space="covariance_state_operator"` and must not pass any
`--template-selection` flag. They are dedicated direction operators, not PIA
template selectors.

CS-Flow arms follow the same no-template-policy rule. `cs_flow_target_direct`
and `cs_flow_single_step` use `source_space="covariance_state_flow"` and must
not pass any `--template-selection` flag. `cs_flow_target_direct` is a
debug-only probe, not a competing paper baseline.

## Adding A True External Baseline

Add runtime metadata to `METHOD_INFO`, implement or dispatch the augmentation under `utils/external_baseline_methods/` or `utils/external_aug_dispatch.py`, and add a manifest row documenting the paper/source and implementation status.

Do not implement external algorithms in `run_act_pilot.py`; that file remains the public CSTA/ACT CLI.

## Protected Roots

Do not use locked reference roots for smoke or probe runs:

- `results/csta_external_baselines_phase1/resnet1d_s123/`
- `results/csta_external_baselines_phase2/resnet1d_s123/`

Use `/tmp/...` or a clearly named local result root unless intentionally regenerating locked references with the explicit overwrite flag.
