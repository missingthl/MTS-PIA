
import os
import sys

# Add project root to sys.path BEFORE local imports
print(f"CWD: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from datasets.adapters import get_adapter
from runners.spatial_dcnet_torch import DCNetTorch, SpatialDCNetRunnerTorch

def measure_window_acc(seed=0):
    ckpt_path = f"experiments/checkpoints/seedv_spatial_torch_seed{seed}_phase13a_spatial_fallback.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    adapter = get_adapter("seed1")
    folds = adapter.get_spatial_folds_for_cnn(
        seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s",
        seed_de_var="de_LDS1"
    )
    fold = folds['fold1']
    X_te = fold.X_test
    # Need to verify if y_test is window labels or trial labels expanded
    # The adapter documentation or code would clarify.
    # Usually X_test is [N_windows, 310] and y_test is [N_windows]
    y_te = fold.y_test.ravel()
    
    print(f"Test Data Shape: {X_te.shape}, Labels: {y_te.shape}")
    
    # Load Model
    model = DCNetTorch(310, 3).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Inference
    X_te_tensor = torch.from_numpy(X_te).float().reshape(-1, 310, 1, 1)
    
    preds_list = []
    batch_size = 2048
    
    with torch.no_grad():
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_te_tensor), 
            batch_size=batch_size, shuffle=False
        )
        for (bx,) in dl:
            out = model(bx.to(device))
            preds_list.append(torch.softmax(out, dim=1).cpu().numpy())
            
    proba_test = np.concatenate(preds_list, axis=0)
    y_pred_win = np.argmax(proba_test, axis=1)
    
    # Accuracy
    acc = np.mean(y_pred_win == y_te)
    print(f"\nSeed {seed} Window Accuracy: {acc:.4f}")
    
    # Breakdown
    correct = np.sum(y_pred_win == y_te)
    total = len(y_te)
    print(f"Correct: {correct}/{total}")

if __name__ == "__main__":
    measure_window_acc(0)
