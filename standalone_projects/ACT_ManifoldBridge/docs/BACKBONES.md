# Backbone Map

This page maps ACT/CSTA host backbones to their model files, trainer dispatch
layers, and current support status.

Backbones are **classification hosts**.  They are not augmentation baselines and
they are not external augmentation methods.  They answer a different question:

```text
Does CSTA/PIA remain useful when the downstream classifier changes?
```

## Directory Layout

Current project-native backbone model files live directly under `core/`:

```text
core/resnet1d.py
core/patchtst.py
core/timesnet.py
core/mptsnet.py
core/moderntcn.py
```

`minirocket` is the exception: it is a sklearn/sktime-style feature pipeline
routed through evaluator utilities rather than a `core/minirocket.py` model file.

Training/evaluation dispatch is split by experiment owner:

```text
core/csta/training.py
  Training dispatch used by run_act_pilot.py and CSTA/MBA/RC4 internal runs.

utils/backbone_trainers.py
  Training dispatch used by external baseline matrix runs.

utils/evaluators.py
  Low-level fit/evaluate implementations and shared training loops.
```

## Supported Backbones

| Backbone | Model location | Main dispatch | Current role |
| --- | --- | --- | --- |
| `resnet1d` | `core/resnet1d.py` | `core/csta/training.py`, `utils/backbone_trainers.py` | Canonical CSTA/PIA and external-baseline host. |
| `minirocket` | `utils/evaluators.py` | `core/csta/training.py`, `utils/backbone_trainers.py` | Hard-label robustness host; no soft-label support. |
| `patchtst` | `core/patchtst.py` | `core/csta/training.py`, `utils/backbone_trainers.py` | Hard-label robustness host. |
| `timesnet` | `core/timesnet.py` | `core/csta/training.py`, `utils/backbone_trainers.py` | Hard-label robustness host. |
| `mptsnet` | `core/mptsnet.py` | `core/csta/training.py`, `utils/backbone_trainers.py` | Backbone robustness host, not an augmentation competitor. |
| `moderntcn` | `core/moderntcn.py` | `utils/backbone_trainers.py` | External-runner hard-label host; CSTA path support is limited. |

## Support Boundaries

Hard-label training is the broadest path:

```text
resnet1d / minirocket / patchtst / timesnet / mptsnet / moderntcn
```

Special training modes are intentionally narrower:

| Mode | Supported backbone | Reason |
| --- | --- | --- |
| `raw_mixup` soft labels | `resnet1d` only | Soft-label training has a ResNet1D-specific trainer. |
| `manifold_mixup` | `resnet1d` only | Hidden-state mixup is implemented for ResNet1D. |
| `jobda_cleanroom` joint labels | `resnet1d` only | Joint-label inference wrapper is ResNet1D-specific. |
| weighted aug-CE in CSTA internals | `resnet1d`, `patchtst`, `timesnet` | Implemented in `core/csta/training.py`; not generalized to all hosts. |

Non-supported combinations should fail fast.  They should not silently fall back
to ResNet1D or a different loss path.

## Experiment Flow

Canonical CSTA/PIA runs:

```text
run_act_pilot.py --model <backbone>
  ↓
core/csta/cli.py
  ↓
core/csta/experiment.py
  ↓
core/csta/training.py
  ↓
utils/evaluators.py
```

External baseline matrix runs:

```text
scripts/run_external_baselines_phase1.py --backbone <backbone>
  ↓
utils/backbone_trainers.py
  ↓
utils/evaluators.py
```

MPTSNet pilot helper:

```text
scripts/run_pilot7_mptsnet.sh
  ↓
run_act_pilot.py --model mptsnet
```

## Naming Guidance

Use backbone names only for downstream host classifiers:

```text
Correct:
  CSTA-PIA with ResNet1D
  CSTA-PIA with MPTSNet
  wDBA baseline evaluated with ResNet1D

Avoid:
  MPTSNet as an augmentation baseline
  ResNet1D as an external method
```

MPTSNet is useful for robustness evidence, but it should be described as a
backbone check, not as a competitor to DBA/wDBA/RGW/DGW/JobDA/TimeVAE.

## Future Cleanup

The current layout keeps model imports stable.  If the backbone list continues
to grow, the clean next step is:

```text
core/backbones/
  resnet1d.py
  patchtst.py
  timesnet.py
  mptsnet.py
  moderntcn.py
```

and then preserve compatibility shims:

```text
core.resnet1d
core.patchtst
core.timesnet
core.mptsnet
core.moderntcn
```

This should be a separate compatibility refactor, not mixed with CSTA algorithm
or external-baseline changes.
