from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from aeon.classification.convolution_based import MultiRocketHydraClassifier

# Local imports
from core.resnet1d import ResNet1DClassifier

def build_model(n_kernels: int = 10000, random_state: int = 42, n_jobs: int = 1):
    # Using MultiRocketHydraClassifier for robustness
    return MultiRocketHydraClassifier(
        n_kernels=n_kernels,
        random_state=random_state,
        n_jobs=n_jobs
    )


def fit_eval_minirocket(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Basic fit and eval loop for MiniRocket on Raw MTS data."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro"))
    }


def fit_eval_resnet1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda",
) -> Dict[str, float]:
    """PyTorch training loop for ResNet1D."""
    
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    # 1. Prepare Data
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 2. Build Model
    in_channels = X_train.shape[1]
    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    model = ResNet1DClassifier(in_channels, num_classes).to(dev)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    # 4. Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(dev)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(by.numpy())
            
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro"))
    }
