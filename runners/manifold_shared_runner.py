
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from models.spdnet_shared import SharedMultiBandSPDNet

class MultiBandTrialDataset(Dataset):
    def __init__(self, X, y, info=None):
        """
        X: List of (62, T, 5) arrays (variable T)
        y: (N_trials,)
        info: List of dict info per trial
        """
        # If X is list, keep as list. If tensor, keep as tensor.
        self.X = [torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x for x in X]
        self.y = torch.tensor(y, dtype=torch.long)
        self.info = info if info is not None else [{}] * len(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.info[idx]

class SharedManifoldRunner:
    def __init__(self, seed, device="cuda"):
        self.seed = seed
        self.device = device
        self.model = None
        
    def train_and_evaluate(self, train_data, test_data, config):
        # Unpack Data
        # Expecting raw DE features grouped by trial: (B, 62, T, 5)
        X_tr, y_tr, info_tr = train_data
        X_te, y_te, info_te = test_data
        
        # Datasets
        ds_tr = MultiBandTrialDataset(X_tr, y_tr, info_tr)
        ds_te = MultiBandTrialDataset(X_te, y_te, info_te)
        
        dl_tr = DataLoader(ds_tr, batch_size=config['batch_size'], shuffle=True, drop_last=True) # Drop last to avoid singular cov
        dl_te = DataLoader(ds_te, batch_size=config['batch_size'], shuffle=False)
        
        # Model
        self.model = SharedMultiBandSPDNet(n_classes=3, dropout=config['dropout']).to(self.device)
        self.model.train() # Set clean state
        
        # Optimizer (Param Groups)
        # Backbone: 1e-4, Others: 1e-3
        backbone_params = list(map(id, self.model.backbone.parameters()))
        base_params = filter(lambda p: id(p) not in backbone_params, self.model.parameters())
        
        optimizer = optim.AdamW([
            {'params': self.model.backbone.parameters(), 'lr': config['lr_backbone']},
            {'params': base_params, 'lr': config['lr_head']}
        ], weight_decay=config['weight_decay'])
        
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        
        print(f"[Seed {self.seed}] Starting Training (Shared Backbone)...")
        
        for ep in range(config['epochs']):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for Xb, yb, _ in dl_tr:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                
                # Grad Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config['clip_grad_norm'])
                
                optimizer.step()
                
                total_loss += loss.item() * Xb.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += Xb.size(0)
                
            train_acc = correct / total if total > 0 else 0
            
            # Eval
            if (ep + 1) % 5 == 0 or (ep + 1) == config['epochs']:
                test_acc, _, _ = self._evaluate(dl_te)
                print(f"Ep {ep+1:02d} | Loss: {total_loss/total:.4f} | TrAcc: {train_acc:.4f} | TeAcc: {test_acc:.4f}")
                
        # Final Full Evaluation & Export
        print(f"[Seed {self.seed}] Training Complete.")
        
        # Eval Train (Full, no drop_last)
        dl_tr_eval = DataLoader(ds_tr, batch_size=config['batch_size'], shuffle=False)
        acc_tr, preds_tr, metrics_tr = self._evaluate(dl_tr_eval)
        acc_te, preds_te, metrics_te = self._evaluate(dl_te)
        
        # Gap
        gap = acc_tr - acc_te
        
        # Wrong Confidence
        wc_tr = metrics_tr['wrong_confidence']
        wc_te = metrics_te['wrong_confidence']
        
        print(f"RESULTS: Train={acc_tr:.4f}, Test={acc_te:.4f}, Gap={gap:.4f}")
        print(f"WRONG-CONF: Train={wc_tr:.4f}, Test={wc_te:.4f}")
        
        # Export
        out_dir = f"experiments/phase9_7_shared/seed{self.seed}"
        os.makedirs(out_dir, exist_ok=True)
        
        self._export_csv(preds_tr, f"{out_dir}/preds_train.csv")
        self._export_csv(preds_te, f"{out_dir}/preds_test.csv")
        
        metrics = {
            "seed": int(self.seed),
            "train_acc": float(acc_tr),
            "test_acc": float(acc_te),
            "gap": float(gap),
            "train_wrong_conf": float(wc_tr),
            "test_wrong_conf": float(wc_te)
        }
        with open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        return metrics

    def _evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_infos = [] # list of lists/tuples from dataloader batching
        
        with torch.no_grad():
            for Xb, yb, info_b in dataloader:
                Xb = Xb.to(self.device)
                logits = self.model(Xb)
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yb.numpy())
                # info_b is tuple of tuples usually (subject_batch, session_batch ...) or list of dicts?
                # Dataset returns info[idx] which is a dict.
                # DataLoader collates list of dicts into dict of lists.
                # e.g. {'trial_id': ['id1', 'id2'], ...}
                
                # We need to de-collate back to row-wise
                keys = info_b.keys()
                batch_size = len(yb)
                rows = []
                for i in range(batch_size):
                    row = {k: info_b[k][i] for k in keys}
                    rows.append(row)
                all_infos.extend(rows)

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        pred_labels = np.argmax(all_probs, axis=1)
        max_probs = np.max(all_probs, axis=1)
        
        acc = np.mean(pred_labels == all_labels)
        
        # Metrics
        wrong_mask = (pred_labels != all_labels)
        if np.any(wrong_mask):
            wrong_conf = np.mean(max_probs[wrong_mask])
        else:
            wrong_conf = 0.0
            
        metrics = {"wrong_confidence": wrong_conf}
        
        # Prepare Export Data
        export_data = []
        for i in range(len(all_labels)):
            row = all_infos[i].copy()
            row['true_label'] = int(all_labels[i])
            row['pred_label'] = int(pred_labels[i])
            row['max_prob'] = float(max_probs[i])
            row['p0'] = float(all_probs[i,0])
            row['p1'] = float(all_probs[i,1])
            row['p2'] = float(all_probs[i,2])
            export_data.append(row)
            
        return acc, export_data, metrics
        
    def _export_csv(self, data, path):
        df = pd.DataFrame(data)
        # Reorder handy columns
        first_cols = ['seed', 'subject', 'session', 'trial_id', 'split', 'true_label', 'pred_label', 'max_prob']
        # Intersect with existing
        first_cols = [c for c in first_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in first_cols]
        df = df[first_cols + other_cols]
        df.to_csv(path, index=False)
