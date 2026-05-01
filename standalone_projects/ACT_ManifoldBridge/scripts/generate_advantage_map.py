import pandas as pd
import numpy as np
from pathlib import Path

def analyze_regimes(summary_df):
    """
    Implements the Advantageous Interval Analysis logic.
    """
    df = summary_df.copy()
    
    # Derived Metrics
    df['sample_scarcity'] = df['N_train'] / df['n_classes']
    df['covariance_dim'] = df['C'] * (df['C'] + 1) / 2
    df['cov_to_sample_ratio'] = df['covariance_dim'] / df['N_train']
    df['baseline_difficulty'] = 1.0 - df['no_aug_f1']
    
    # Sensitivities
    df['raw_aug_sensitivity'] = df['best_rawaug_f1'] - df['no_aug_f1']
    df['dtw_sensitivity'] = df['wdba_f1'] - df['no_aug_f1']
    df['cov_state_sensitivity'] = df[['random_cov_f1', 'pca_cov_f1']].max(axis=1) - df['no_aug_f1']
    df['pia_extra_gain'] = df['csta_pia_f1'] - df[['random_cov_f1', 'pca_cov_f1']].max(axis=1)
    
    # Comparison Deltas
    df['csta_gain'] = df['csta_pia_f1'] - df['no_aug_f1']
    df['csta_vs_wdba'] = df['csta_pia_f1'] - df['wdba_f1']
    df['csta_vs_bestrawaug'] = df['csta_pia_f1'] - df['best_rawaug_f1']
    
    # Regime Labeling
    def label_regime(row):
        if row['csta_vs_wdba'] > 0.005 and row['csta_vs_bestrawaug'] > 0.005:
            return "CSTA-dominant"
        if abs(row['csta_vs_wdba']) <= 0.005:
            return "CSTA-competitive"
        if row['wdba_f1'] - row['csta_pia_f1'] > 0.01:
            return "DTW-dominant"
        if row['best_rawaug_f1'] - row['csta_pia_f1'] > 0.01:
            return "RawAug-dominant"
        return "Mixed/Unstable"
    
    df['regime_label'] = df.apply(label_regime, axis=1)
    return df

# Script main logic for later execution...
