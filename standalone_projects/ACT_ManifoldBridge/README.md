# ACT_ManifoldBridge

`ACT_ManifoldBridge` is organized around a two-layer view of augmentation for multivariate time series:

- `MBA-Core`: the primary geometric generator.
- `ACT-Lite`: the minimal host-response feedback layer.
- `ACT-Heavy`: the historical exploratory ACL protocol retained for analysis, not as the default path.

## Core Narrative

### MBA-Core

`MBA-Core` is the method body of the repository. It keeps the clean geometric chain:

`x -> z(x) -> z_cand -> x_cand`

In code, this means:

- trial records are embedded as static SPD / Log-Euclidean points
- local directions come from `PIA` or `LRAES`
- candidate points are generated in latent space
- `bridge_single()` realizes candidates back into raw signals

This is the default protocol exposed through:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops --pipeline mba --algo lraes --model resnet1d
```

### ACT-Lite

`ACT-Lite` does not replace the generator. It only controls how much the host should use each MBA candidate during training.

For each augmented sample `x_cand`, the current host computes a true-class margin:

`margin(x_cand) = logit_y - max_{c!=y} logit_c`

The augmentation utilization weight is then:

`w_aug = sigmoid(margin / temperature)`

Training stays in the plain classification regime:

`L = L_ce(orig) + lambda_aug * mean(w_aug * CE(x_cand, y))`

This path is exposed through:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops --pipeline mba_feedback --algo lraes --model resnet1d
```

Current scope:

- `mba_feedback` supports `ResNet1D`
- no projection head in the main path
- no SupCon in the main path
- no selected-positive admission logic in the main path

### ACT-Heavy

`gcg_acl` is kept as the historical exploratory protocol for contrastive candidate selection and ACL-style utilization. It remains runnable for auditing and retrospective comparison, but it is not the default experimental narrative.

## Project Structure

- [run_act_pilot.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/run_act_pilot.py): protocol entrypoint for `mba`, `mba_feedback`, and `gcg_acl`
- [core/pia.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core/pia.py): direction-bank construction
- [core/curriculum.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core/curriculum.py): latent candidate generation
- [core/bridge.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core/bridge.py): whitening-coloring bridge
- [act_lite_feedback.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/act_lite_feedback.py): minimal margin-based feedback utilities
- [utils/evaluators.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/utils/evaluators.py): host training and evaluation

## Practical Notes

- `mba` remains the default CLI pipeline.
- `mba_feedback` keeps the original supervision stream intact and adds only a weighted augmentation loss branch.
- in `mba_feedback`, augmented batches are forwarded with frozen batch-stat updates so BatchNorm statistics continue to be driven by the original supervision stream only.
- frozen ACL result packages under `results/acl_small_matrix_v1` and its follow-up directories are retained as historical reference and are not rewritten by the refactor.

## Quick Start

Run the baseline geometric generator:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset basicmotions --pipeline mba --algo lraes --model resnet1d \
  --seeds 1 --epochs 30 --device cuda
```

Run the minimal feedback version:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset basicmotions --pipeline mba_feedback --algo lraes --model resnet1d \
  --seeds 1 --epochs 30 --feedback-margin-temperature 1.0 \
  --feedback-aug-weight 1.0 --device cuda
```

Run the historical ACL protocol:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset basicmotions --pipeline gcg_acl --algo lraes --model resnet1d \
  --seeds 1 --epochs 30 --acl-warmup-epochs 10 --device cuda
```
