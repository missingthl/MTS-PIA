#!/bin/bash

# Configuration
DATASETS="atrialfibrillation ering handmovementdirection handwriting japanesevowels natops racketsports"
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_neurips_ablation_pilot7/resnet1d_sharedbudget_s123"
SEEDS="1,2,3"

# Common Args
COMMON="--model resnet1d --seeds $SEEDS --epochs 30 --batch-size 64 --lr 1e-3 --patience 10 --val-ratio 0.2 --k-dir 10 --pia-gamma 0.1 --eta-safe 0.5"

mkdir -p $OUT_ROOT

for DS in $DATASETS; do
    echo "Processing Dataset: $DS"
    
    # 1. csta_top1_repeated
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_top1_pool --multiplier 10 --out-root "$OUT_ROOT/csta_top1_repeated"
    
    # 2. csta_top1_unique
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_top1_pool --multiplier 2 --out-root "$OUT_ROOT/csta_top1_unique"
    
    # 3. csta_random1
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_top1_pool --template-selection random --multiplier 10 --out-root "$OUT_ROOT/csta_random1"
    
    # 4. csta_fixed1
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_top1_pool --template-selection fixed --multiplier 10 --out-root "$OUT_ROOT/csta_fixed1"
    
    # 5. csta_multidir
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_multidir_pool --multi-template-pairs 5 --multiplier 10 --out-root "$OUT_ROOT/csta_multidir"
    
    # 6. csta_no_safe
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_top1_pool --disable-safe-step --multiplier 10 --out-root "$OUT_ROOT/csta_no_safe"
    
    # 7. csta_pca_template
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_top1_pool --template-source pca --multiplier 10 --out-root "$OUT_ROOT/csta_pca_template"
    
    # 8. csta_random_orth_template
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py $COMMON --dataset $DS --algo zpia_top1_pool --template-source random_orth --multiplier 10 --out-root "$OUT_ROOT/csta_random_orth_template"
done

echo "Ablation sweep complete."
