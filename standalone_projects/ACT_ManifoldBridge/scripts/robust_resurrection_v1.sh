#!/bin/bash

# Configuration
PYTHON="/home/THL/miniconda3/envs/pia/bin/python"
SCRIPT="scripts/run_external_baselines_phase1.py"

# --- 1. PatchTST Recovery (GPU 0 & 1) ---
# motorimagery (Exclusive on GPU 0, BS=2)
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --backbone patchtst --out-root results/patchtst_final20_v1/motor_rec --datasets "motorimagery" --seeds "1,2,3" --arms "no_aug,csta_topk_uniform_top5" --group-size 1 --batch-size 2 >> results/rec_patch_motor.log 2>&1 &

# har, heartbeat, pendigits (GPU 1)
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT --backbone patchtst --out-root results/patchtst_final20_v1/misc_rec --datasets "har,heartbeat,pendigits" --seeds "1,2,3" --arms "no_aug,csta_topk_uniform_top5" --group-size 1 >> results/rec_patch_misc.log 2>&1 &

# --- 2. TimesNet Recovery (GPU 2 & 3) ---
# TimesNet was hit hard, let's split the incomplete ones
T_DS1="epilepsy,articularywordrecognition,racketsports,handwriting,japanesevowels"
T_DS2="libras,uwavegesturelibrary,natops,fingermovements,heartbeat"

CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT --backbone timesnet --out-root results/timesnet_final20_v1/rec_split1 --datasets "$T_DS1" --seeds "1,2,3" --arms "no_aug,csta_topk_uniform_top5" --group-size 1 >> results/rec_times1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PYTHON $SCRIPT --backbone timesnet --out-root results/timesnet_final20_v1/rec_split2 --datasets "$T_DS2" --seeds "1,2,3" --arms "no_aug,csta_topk_uniform_top5" --group-size 1 >> results/rec_times2.log 2>&1 &

# --- 3. MiniRocket Recovery (GPU 2) ---
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT --backbone minirocket --out-root results/minirocket_final20_recovery_v3 --datasets "pendigits" --seeds "1,2,3" --arms "no_aug,csta_topk_uniform_top5" --group-size 1 --eta-safe 0.5 >> results/rec_mr_pen.log 2>&1 &

echo "Resurrection launched with Power-Safe mode (Max 1 task per GPU)."
