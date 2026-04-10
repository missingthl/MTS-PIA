
import sys
import os
sys.path.append(os.getcwd())
from datasets.adapters import Seed1Adapter
import numpy as np

def inspect_ids():
    adapter = Seed1Adapter()
    print("Loading folds...")
    folds = adapter.get_manifold_trial_folds()
    fold = folds['fold1']
    
    tr_ids = fold.trial_id_train
    te_ids = fold.trial_id_test
    
    print(f"Train IDs ({len(tr_ids)}): {tr_ids[:5]} ... {tr_ids[-5:]}")
    print(f"Test IDs ({len(te_ids)}): {te_ids[:5]} ... {te_ids[-5:]}")
    
    # Check type
    print(f"Type: {type(tr_ids[0])}")
    
    # Check intersection
    intersect = np.intersect1d(tr_ids, te_ids)
    print(f"Intersection content: {intersect}")
    print(f"Intersection count: {len(intersect)}")

if __name__ == "__main__":
    inspect_ids()
