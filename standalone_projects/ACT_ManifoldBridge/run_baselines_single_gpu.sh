#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "Starting Lightweight Baselines..."
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms raw_aug_jitter,raw_aug_timewarp,raw_mixup --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_SINGLE_GPU >> gpu0_single.log 2>&1

echo "Starting DTW Group 1..."
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms dba_sameclass,wdba_sameclass --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_SINGLE_GPU >> gpu0_single.log 2>&1

echo "Starting DTW Group 2..."
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary,InsectWingbeat,PEMS-SF --arms rgw_sameclass,dgw_sameclass --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_SINGLE_GPU >> gpu0_single.log 2>&1

echo "Starting Generative Models (Standard Datasets)..."
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary --arms timegan_classwise,diffusionts_classwise --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 10 --out-root results/E1_BASELINES_SINGLE_GPU >> gpu0_single.log 2>&1

echo "Starting Generative Models (Heavy Datasets - Multiplier 2)..."
conda run -n pia python scripts/run_external_baselines_phase1.py --datasets InsectWingbeat,PEMS-SF --arms timegan_classwise,diffusionts_classwise --seeds 1,2,3 --backbone resnet1d --epochs 100 --multiplier 2 --out-root results/E1_BASELINES_SINGLE_GPU_HEAVY >> gpu0_single.log 2>&1

echo "All 9 Baselines completed on single GPU." >> gpu0_single.log
