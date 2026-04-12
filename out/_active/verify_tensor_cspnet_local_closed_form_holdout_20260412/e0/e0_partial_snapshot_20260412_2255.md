# E0 Partial Snapshot

Snapshot time: `2026-04-12 22:55` (Asia/Shanghai)

This note records the current confirmed state of the long-running `E0` full holdout experiment before partially interrupting it to free one GPU for follow-up work.

## Run groups

- `GPU1`:
  - run tag: `seed1_s3_e0_sub1_4_seed1`
  - command:
    - `python scripts/run_tensor_cspnet_local_closed_form_holdout.py --arm e0 --start-no 1 --end-no 4 --epochs 100 --seed 1 --num-workers 8 --log-every 10 --run-tag s3_e0_sub1_4_seed1`
  - run dir:
    - `/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e0/seed1_s3_e0_sub1_4_seed1`
  - pid at snapshot:
    - `1413761`

- `GPU2`:
  - run tag: `seed1_s3_e0_sub5_9_seed1`
  - command:
    - `python scripts/run_tensor_cspnet_local_closed_form_holdout.py --arm e0 --start-no 5 --end-no 9 --epochs 100 --seed 1 --num-workers 8 --log-every 10 --run-tag s3_e0_sub5_9_seed1`
  - run dir:
    - `/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e0/seed1_s3_e0_sub5_9_seed1`
  - pid at snapshot:
    - `1413760`

## Confirmed completed subjects

These are the latest confirmed subject-level results observed before the snapshot.

- `subject 1`: `acc=0.8368`, `loss=0.393698`, `wallclock=6045.6s`
- `subject 5`: `acc=0.6076`, `loss=1.090845`, `wallclock=6188.9s`

## Last confirmed in-progress states

The training script only writes `per_subject.csv` and `summary.json` after a run group finishes, so partially completed subjects were not yet flushed to disk at the time of snapshot.

- `GPU1 / subject 2`: previously observed progressing past epoch checkpoints and known to be mid-run.
- `GPU2 / subject 6`: previously observed progressing past epoch checkpoints and known to be mid-run.

## Process state at snapshot

- `pid 1413761`: elapsed `03:25:47`, `%CPU=100`, `STAT=Ssl`
- `pid 1413760`: elapsed `03:25:47`, `%CPU=100`, `STAT=Rsl`

## Interruption plan

- Keep `GPU1` `E0 sub1-4` running.
- Stop `GPU2` `E0 sub5-9` to free one GPU for the next experiment wave.

