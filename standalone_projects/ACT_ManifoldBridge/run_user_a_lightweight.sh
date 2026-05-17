#!/bin/bash
# USER A LIGHTWEIGHT RE-RUN: GPU 0 (NUMA 0)

echo "Dispatching User A: GPU 0 (NUMA 0) -> Lightweight Models..."
export CUDA_VISIBLE_DEVICES=0
taskset -c 0-51,104-155 conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,InsectWingbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PEMS-SF,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary --arms raw_aug_jitter,raw_aug_timewarp,raw_mixup --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_USER_A_LIGHTWEIGHT > gpu0_lightweight.log 2>&1 &
echo "Lightweight baselines successfully dispatched on GPU 0."
