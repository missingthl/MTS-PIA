#!/bin/bash
# USER A CONFIGURATION: GPU 0 & GPU 1, NUMA Node 0 (Cores 0-51,104-155)

echo "Killing old master bash and python processes..."
pkill -f "run_baselines_single_gpu.sh"
pkill -f "run_external_baselines_phase1.py"
sleep 3

echo "Dispatching User A: GPU 0 (NUMA 0) -> Generative Models..."
export CUDA_VISIBLE_DEVICES=0
(
    taskset -c 0-51,104-155 conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary --arms timegan_classwise,diffusionts_classwise --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_USER_A_GPU0 &&     taskset -c 0-51,104-155 conda run -n pia python scripts/run_external_baselines_phase1.py --datasets InsectWingbeat,PEMS-SF --arms timegan_classwise,diffusionts_classwise --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 2 --out-root results/E1_BASELINES_USER_A_GPU0_HEAVY
) > gpu0_user_a.log 2>&1 &

echo "Dispatching User A: GPU 1 (NUMA 0) -> Lightweight + DTW Models..."
export CUDA_VISIBLE_DEVICES=1
(
    taskset -c 0-51,104-155 conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms raw_aug_jitter,raw_aug_timewarp,raw_mixup --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_USER_A_GPU1 &&     taskset -c 0-51,104-155 conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms dba_sameclass,wdba_sameclass,rgw_sameclass,dgw_sameclass --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_USER_A_GPU1
) > gpu1_user_a.log 2>&1 &

echo "User A baselines successfully dispatched on NUMA 0 (taskset)."
