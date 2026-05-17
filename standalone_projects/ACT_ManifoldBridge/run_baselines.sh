#!/bin/bash
# GPU 3: Lightweight
export CUDA_VISIBLE_DEVICES=3
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms raw_aug_jitter,raw_aug_timewarp,raw_mixup --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_GPU3 > gpu3_baselines.log 2>&1 &

# GPU 1: DTW Group 1
export CUDA_VISIBLE_DEVICES=1
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms dba_sameclass,wdba_sameclass --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_GPU1 > gpu1_baselines.log 2>&1 &

# GPU 2: DTW Group 2
export CUDA_VISIBLE_DEVICES=2
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms rgw_sameclass,dgw_sameclass --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_GPU2 > gpu2_baselines.log 2>&1 &

# GPU 0: Generative Models (Split into Standard and Heavy)
export CUDA_VISIBLE_DEVICES=0
(
    conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary --arms timegan_classwise,diffusionts_classwise --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_GPU0 &&     conda run -n pia python scripts/run_external_baselines_phase1.py --datasets InsectWingbeat,PEMS-SF --arms timegan_classwise,diffusionts_classwise --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 2 --out-root results/E1_BASELINES_GPU0_HEAVY
) > gpu0_baselines.log 2>&1 &

echo "All baselines dispatched."
