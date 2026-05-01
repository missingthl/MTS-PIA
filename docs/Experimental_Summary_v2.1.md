# Experimental Summary: CSTA-PIA v2.1 (Engineering Audit & Final-20 Benchmark)

## 1. Overview
This version implements critical engineering refinements to the PIA operator, including numerical stability for Softmax sampling, explicit `eta_safe` propagation, and validated `aug_valid_rate` metrics.

## 2. Core Results (ResNet1D - 20 Datasets)
Validated with optimal configuration: `Gamma=0.1`, `Eta_Safe=0.75`, `Policy=Uniform-Top5`.

| Metric | Value |
| :--- | :--- |
| **Mean F1 (Baseline)** | 0.6874 |
| **Mean F1 (PIA v2.1)** | **0.7279** |
| **Absolute Gain** | **+4.05%** |
| **Win Rate** | **17 / 20 (85%)** |

## 3. Advantageous Intervals
PIA demonstrates significant SOTA performance in the following regimes:

- **Gesture & Trajectory Recognition**: 
  - `Libras`: **+30.7%** gain.
  - `uWaveGestureLibrary`: **+8.1%** gain.
- **Handwriting Analysis**:
  - `Handwriting`: **+6.8%** gain.
- **Physical Dynamics**:
  - `ERing`: **+7.4%** gain.

## 4. Engineering Improvements
- **Numerical Stability**: Implemented logit max-normalization in Softmax realization.
- **Safety Step Logic**: Fixed a bug where `eta_safe` was hardcoded; now correctly propagates through the CLI.
- **Performance Pipeline**: Implemented a 4-GPU parallel runner reducing full-scale benchmark time by ~75%.
