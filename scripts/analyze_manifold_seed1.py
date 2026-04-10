
import sys
import os
import numpy as np

# Adjust path to import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.seed_official_de import load_seed_official_de
from datasets.types import TrialFoldData
from runners.manifold_runner import ManifoldRunner
from tools.metrics import accuracy_from_proba

def reconstruct_trials(X, y, trial_indices):
    """Reconstruct list of trials from concatenated features based on trial metadata."""
    trials = []
    y_trials = []
    ids = []
    
    cursor = 0
    for meta in trial_indices:
        n = meta['n_windows']
        if n == 0: continue
        
        # Access segment
        segment = X[cursor : cursor + n]
        label = int(meta['label'])
        tid = meta['trial_id']
        
        trials.append(segment)
        y_trials.append(label)
        ids.append(tid)
        
        cursor += n
        
    return trials, np.array(y_trials), np.array(ids)

def main():
    # Configuration matches what was used in pia_unified_demo logs
    SEED_DE_ROOT = "data/SEED/SEED_EEG/ExtractedFeatures_1s"
    SEED_DE_VAR = "de_LDS1"
    MANIFEST = "logs/seed_raw_trial_manifest_full.json"
    
    print(f"Loading SEED1 data from {SEED_DE_ROOT}...")
    try:
        trainx, trainy, testx, testy, trial_index, skipped = load_seed_official_de(
            seed_de_root=SEED_DE_ROOT,
            seed_de_var=SEED_DE_VAR,
            manifest_path=MANIFEST,
            freeze_align=True 
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Filter indices by split
    train_meta = [m for m in trial_index if m['split'] == 'train']
    test_meta = [m for m in trial_index if m['split'] == 'test']
    
    print("Reconstructing trials...")
    trials_train, y_trial_train, id_train = reconstruct_trials(trainx, trainy, train_meta)
    trials_test, y_trial_test, id_test = reconstruct_trials(testx, testy, test_meta)
    
    print(f"Train trials: {len(trials_train)}, Test trials: {len(trials_test)}")
    
    # Construct Fold Data
    fold = TrialFoldData(
        trials_train=trials_train,
        y_trial_train=y_trial_train,
        trials_test=trials_test,
        y_trial_test=y_trial_test,
        trial_id_train=id_train,
        trial_id_test=id_test
    )
    
    # Configure Manifold Runner
    # Using 'band' mode (splits 310 -> 5x62), 'ra' alignment, 'logeuclidean' metric
    print("Initializing ManifoldRunner...")
    runner = ManifoldRunner(
        mode="band",
        classifier="svm",
        C=1.0,
        align_mode="ra",         # Riemannian Alignment
        ra_mode="euclidean",     # Reference mean
        mdm_metric="logeuclidean",
        debug=True 
    )
    
    print("Running Fit/Predict...")
    result = runner.fit_predict(fold)
    
    acc = accuracy_from_proba(result['y_trial_test'], result['trial_proba_test'])
    print("\n[Manifold Stream Results]")
    print(f"Trial-level Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
