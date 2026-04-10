
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset
from models.spdnet_multiband import MultiBandDeepSPDClassifier

class MultiBandTrialDataset(Dataset):
    def __init__(self, trials, labels, trial_ids=None):
        """
        trials: list of (T, 310) numpy arrays
        labels: list of int
        trial_ids: list of str (optional)
        """
        self.samples = []
        self.labels = []
        self.trial_ids = trial_ids if trial_ids is not None else [f"t{i}" for i in range(len(trials))]
        
        # Preprocess each trial
        for i, trial in enumerate(trials):
            # trial: (T, 310)
            T_total = trial.shape[0]
            
            # Reshape to (T, 62, 5)
            # Assuming 310 = 62 channels * 5 bands features
            try:
                t_reshaped = trial.reshape(T_total, 62, 5)
            except ValueError:
                # Handle edge cases or potential wrong dim
                print(f"Error reshaping trial {i} with shape {trial.shape} to (T, 62, 5)")
                raise
            
            # Transpose to (62, T, 5)
            # CovPool expects (C, T) usually, so we keep C=62, T=Time, Bands=5
            # We want (62, T, 5) so batching gives (B, 62, T, 5)
            sample = t_reshaped.transpose(1, 0, 2) 
            
            self.samples.append(sample)
            self.labels.append(labels[i])
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Return (62, T, 5)
        x = self.samples[idx]
        y = self.labels[idx]
        # We can't return string in default collate easily if mixed, 
        # but we can return idx to lookup ID later, or rely on sequential order (no shuffle).
        # To be safe, we return idx.
        return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long), idx

class MultiBandRunner:
    def __init__(self, args, num_classes=3):
        self.args = args
        self.num_classes = num_classes
        self.device = args.torch_device if hasattr(args, 'torch_device') and args.torch_device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model Parameters
        self.bands = 5
        self.output_dim = 32 # dim of BiMap output
        # Embedding dim = 32*33/2 = 528
        
        print(f"[MultiBandRunner] Device={self.device}")

    def fit_predict(self, fold, fold_name, seed):
        """
        Train on fold.X_train, Eval on fold.X_test.
        Export predictions for BOTH Train and Test.
        """
        # Group windows into trials
        X_train_list, y_train_list, id_train_list = self._group_into_trials(fold.X_train, fold.y_train, fold.trial_id_train)
        X_test_list, y_test_list, id_test_list = self._group_into_trials(fold.X_test, fold.y_test, fold.trial_id_test)
        
        train_ds = MultiBandTrialDataset(X_train_list, y_train_list, id_train_list)
        test_ds = MultiBandTrialDataset(X_test_list, y_test_list, id_test_list)
        
        # Loader
        # Shuffle Train
        batch_size = self.args.batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # Test & Inferno Loaders (BS=1 for stability)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        train_infer_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        
        # 2. Model
        model = MultiBandDeepSPDClassifier(
            num_classes=self.num_classes,
            bands=self.bands,
            output_dim=self.output_dim,
            dropout_p=0.5
        ).to(self.device).double() 
        
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # 3. Train
        epochs = self.args.epochs
        print(f"[{fold_name}] Starting Training: BS={batch_size}, Epochs={epochs}, Dev={self.device}")
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for X, y, idx in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X) # (B, 3)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X.size(0)
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += X.size(0)
                
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs-1:
                 # Print epoch 0, last, and every 10
                print(f"[{fold_name}] Ep {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")
                
        # 4. Inference & Export
        # Output Columns: seed, stream, subject, session, trial_id, trial_order_in_split, split, true_label, pred_label, max_prob, agg_method, n_windows
        
        out_dir = "experiments/phase9_5_multiband/preds"
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{out_dir}/manifold5_trial_preds_seed{seed}.csv"
        
        mode = 'w' 
        
        with open(fname, mode, newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed", "stream", "subject", "session", "trial_id", 
                "trial_order_in_split", "split", "true_label", "pred_label", 
                "prob_0", "prob_1", "prob_2", "max_prob", 
                "agg_method", "n_windows"
            ])
            
            self._evaluate_and_write(model, test_loader, writer, seed, "test", test_ds.trial_ids)
            self._evaluate_and_write(model, train_infer_loader, writer, seed, "train", train_ds.trial_ids)
            
        print(f"[{fold_name}] Exported predictions to {fname}")
        return {} 

    def _group_into_trials(self, X_flat, y_flat, id_flat):
        """
        Reconstruct trials from flat window arrays.
        X_flat: (N, 310)
        y_flat: (N,)
        id_flat: (N,) string IDs
        """
        unique_ids, indices = np.unique(id_flat, return_index=True)
        # np.unique sorts, which destroys order. 
        # We need to preserve appearance order?
        # Yes, for 'sanity' and 'reproducibility'.
        
        # Fast way to group preserving order:
        import pandas as pd
        df = pd.DataFrame({'id': id_flat, 'idx': range(len(id_flat))})
        # Group by ID
        # Note: trial IDs are unique across session? '1_s1_t0', '1_s1_t1'... yes.
        # But split into train/test beforehand. So uniqueness holds.
        
        groups = df.groupby('id', sort=False)['idx'].apply(list)
        
        X_trials = []
        y_trials = []
        id_trials = []
        
        for tid, idx_list in groups.items():
            # Idx list is indices for this trial
            # Assuming contiguous? 'official' data is contiguous.
            # But groupby handles non-contiguous too.
            indices = idx_list
            x_trial = X_flat[indices] # (T, 310)
            y_trial = y_flat[indices[0]] # Label should be same
            
            X_trials.append(x_trial)
            y_trials.append(y_trial)
            id_trials.append(tid)
            
        return X_trials, y_trials, id_trials
        
    def _evaluate_and_write(self, model, loader, writer, seed, split_name, trial_ids_list):
        model.eval()
        stream_name = "manifold5"
        
        with torch.no_grad():
            for i, (X, y, idx) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X) # (B=1, 3)
                probs = torch.softmax(outputs, dim=1)
                
                # Assuming Batch Size = 1 for output writing clarity
                for b in range(X.size(0)):
                    p_vec = probs[b].cpu().numpy()
                    pred_lbl = np.argmax(p_vec)
                    max_p = np.max(p_vec)
                    true_lbl = y[b].item()
                    
                    p0 = f"{p_vec[0]:.6f}"
                    p1 = f"{p_vec[1]:.6f}"
                    p2 = f"{p_vec[2]:.6f}"
                    
                    # Trial ID
                    d_idx = idx[b].item()
                    t_id = trial_ids_list[d_idx]
                    
                    # Metadata parsing
                    parts = t_id.split('_')
                    if len(parts) >= 2:
                        subj = parts[0]
                        sess = parts[1]
                    else:
                        subj = "u"
                        sess = "u"
                        
                    # Write row
                    writer.writerow([
                        seed, stream_name, subj, sess, t_id,
                        i, # trial_order_in_split
                        split_name, true_lbl, pred_lbl, 
                        p0, p1, p2, f"{max_p:.4f}",
                        "multiband_trial", "1" 
                    ])

