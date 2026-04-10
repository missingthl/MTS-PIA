# ... (imports unchanged)
import sys
import os
import torch
import numpy as np
import csv
sys.path.append(os.getcwd())

from datasets.adapters import Seed1Adapter
from models.spdnet import DeepSPDClassifier
from torch.utils.data import DataLoader

class TrialDataset(torch.utils.data.Dataset):
    def __init__(self, trials, labels, band_idx=4):
        self.trials = trials
        self.labels = labels
        self.band_idx = band_idx
    def __len__(self):
        return len(self.trials)
    def __getitem__(self, idx):
        t = self.trials[idx]
        t_reshaped = t.reshape(t.shape[0], 62, 5)
        x_band = t_reshaped[:, :, self.band_idx]
        x = x_band.transpose(1, 0)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long)

def run_probe():
    os.makedirs("experiments/phase8_rebaseline/reports/stats", exist_ok=True)
    out_csv = "experiments/phase8_rebaseline/reports/phase8_rank_diagnostics_raw.csv"
    
    # 1. Load Data
    print("Loading Data...")
    adapter = Seed1Adapter()
    
    # Init Adapter State
    print("Initializing Adapter State...")
    _ = adapter.get_spatial_folds_for_cnn(
        seed_de_mode="official", 
        seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s", 
        seed_de_var="de_LDS1"
    )
    
    # Get Manifold Folds (uses state set above)
    print("Getting Manifold Folds...")
    folds = adapter.get_manifold_trial_folds()
    fold = folds['fold1']
    
    print(f"Loaded {len(fold.trials_train)} trials.")
    
    train_dset = TrialDataset(fold.trials_train, fold.y_trial_train)
    loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    
    results = []
    
    seeds = [0, 1, 2] # Reduced from orig plan to just ensure we get data
    
    for seed in seeds:
        print(f"Probing Seed {seed}...")
        torch.manual_seed(seed)
        
        # 2. Init Model (Deep SPDNet A2 Config)
        model = DeepSPDClassifier(
            n_channels=62, 
            n_classes=3, 
            output_dim=32, 
            cov_eps=1e-4, 
            cov_alpha=0.01, 
            init_identity=True, 
            hidden_dim=96
        )
        model.double()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        
        # Run small number of steps
        steps = 50
        
        for i, (Xb, yb) in enumerate(loader):
            optimizer.zero_grad()
            
            # Enable Diagnostics
            model.diagnostics_enabled = True
            
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            # Constraint and Disable Diagnostics
            model.diagnostics_enabled = False
            
            with torch.no_grad():
                for module in model.manifold_layers:
                    if hasattr(module, 'W'):
                        Q, R = torch.linalg.qr(module.W)
                        module.W.data.copy_(Q)
            
            if i >= steps: break
            
        # Capture final stats
        last_stats = model.last_diagnostics
        last_stats['seed'] = seed
        results.append(last_stats)
        
    # Write aggregated
    if results:
        keys = results[0].keys()
        with open(out_csv, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
            
    print(f"Probe complete. Saved to {out_csv}")

if __name__ == "__main__":
    run_probe()
