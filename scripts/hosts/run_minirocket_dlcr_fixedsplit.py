#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from datasets.trial_dataset_factory import (  # noqa: E402
    load_trials_for_dataset,
    normalize_dataset_name,
)
from models.minirocket_dlcr_adapter import MiniRocketDLCRAdapter  # noqa: E402

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_json(path: str, obj: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _load_fixedsplit_raw(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    trials = load_trials_for_dataset(
        dataset=args.dataset,
        har_root=args.har_root,
        natops_root=args.natops_root,
        fingermovements_root=args.fingermovements_root,
        selfregulationscp1_root=args.selfregulationscp1_root,
        basicmotions_root=args.basicmotions_root,
        handmovementdirection_root=args.handmovementdirection_root,
        uwavegesturelibrary_root=args.uwavegesturelibrary_root,
        epilepsy_root=args.epilepsy_root,
        atrialfibrillation_root=args.atrialfibrillation_root,
        pendigits_root=args.pendigits_root,
        racketsports_root=args.racketsports_root,
        articularywordrecognition_root=args.articularywordrecognition_root,
        heartbeat_root=args.heartbeat_root,
        selfregulationscp2_root=args.selfregulationscp2_root,
        libras_root=args.libras_root,
        japanesevowels_root=args.japanesevowels_root,
        cricket_root=args.cricket_root,
        handwriting_root=args.handwriting_root,
        ering_root=args.ering_root,
        motorimagery_root=args.motorimagery_root,
        ethanolconcentration_root=args.ethanolconcentration_root,
    )
    train_trials = [t for t in trials if str(t.get("split", "")).lower() == "train"]
    test_trials = [t for t in trials if str(t.get("split", "")).lower() == "test"]
    if not train_trials or not test_trials:
        raise ValueError(f"dataset={args.dataset} does not expose fixed train/test splits")
    
    train_x = np.stack([np.asarray(t["x_trial"], dtype=np.float32) for t in train_trials], axis=0)
    test_x = np.stack([np.asarray(t["x_trial"], dtype=np.float32) for t in test_trials], axis=0)
    train_y = np.asarray([int(t["label"]) for t in train_trials], dtype=np.int64)
    test_y = np.asarray([int(t["label"]) for t in test_trials], dtype=np.int64)
    num_classes = int(max(train_y.max(initial=0), test_y.max(initial=0)) + 1)
    return train_x, test_x, train_y, test_y, num_classes

def _extract_minirocket_features(x: np.ndarray, n_kernels: int = 10000, seed: int = 1) -> np.ndarray:
    from aeon.transformations.collection.convolution_based import MiniRocket
    # Official padding for MiniRocket: minimum length 9
    if x.shape[2] < 9:
        pad_width = 9 - x.shape[2]
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    
    transformer = MiniRocket(n_kernels=int(n_kernels), random_state=int(seed))
    # aeon transformer expects [n_cases, n_channels, n_timepoints] which matches our x
    print(f"--- Extracting MiniRocket features (kernels={n_kernels}, shape={x.shape}) ---", flush=True)
    t0 = time.time()
    features = transformer.fit_transform(x)
    print(f"--- Extraction complete in {time.time() - t0:.2f}s, feature_dim={features.shape[1]} ---", flush=True)
    return features.astype(np.float32)

def _compute_fusion_alpha(args: argparse.Namespace, epoch: int) -> float:
    # Optional schedule, for now fixed at 1.0 or based on epoch
    return 1.0

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniRocket + DLCR fixed-split TSC runner.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-kernels", type=int, default=10000)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--test-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--init-beta", type=float, default=0.1)
    p.add_argument("--prototypes-per-class", type=int, default=4)
    p.add_argument("--routing-temperature", type=float, default=0.2) # Defaulting to peak tau
    p.add_argument("--closed-form-ridge", type=float, default=1e-2)
    p.add_argument("--closed-form-solve-mode", type=str, default="pinv")
    p.add_argument("--closed-form-probe", action="store_true")
    p.add_argument("--prototype-geometry-mode", type=str, default="center_subproto")
    p.add_argument("--out-root", type=str, default="out/minirocket_dlcr_bench")
    
    # Dataset roots
    p.add_argument("--har-root", type=str, default="data/HAR")
    p.add_argument("--natops-root", type=str, default="data/NATOPS")
    p.add_argument("--fingermovements_root", type=str, default="data/FingerMovements")
    p.add_argument("--selfregulationscp1_root", type=str, default="data/SelfRegulationSCP1")
    p.add_argument("--basicmotions_root", type=str, default="data/BasicMotions")
    p.add_argument("--handmovementdirection_root", type=str, default="data/HandMovementDirection")
    p.add_argument("--uwavegesturelibrary_root", type=str, default="data/UWaveGestureLibrary")
    p.add_argument("--epilepsy_root", type=str, default="data/Epilepsy")
    p.add_argument("--atrialfibrillation_root", type=str, default="data/AtrialFibrillation")
    p.add_argument("--pendigits_root", type=str, default="data/Pendigits")
    p.add_argument("--racketsports_root", type=str, default="data/RacketSports")
    p.add_argument("--articularywordrecognition_root", type=str, default="data/ArticularyWordRecognition")
    p.add_argument("--heartbeat_root", type=str, default="data/Heartbeat")
    p.add_argument("--selfregulationscp2_root", type=str, default="data/SelfRegulationSCP2")
    p.add_argument("--libras_root", type=str, default="data/Libras")
    p.add_argument("--japanesevowels_root", type=str, default="data/JapaneseVowels")
    p.add_argument("--cricket_root", type=str, default="data/Cricket")
    p.add_argument("--handwriting_root", type=str, default="data/Handwriting")
    p.add_argument("--ering_root", type=str, default="data/ERing")
    p.add_argument("--motorimagery_root", type=str, default="data/MotorImagery")
    p.add_argument("--ethanolconcentration_root", type=str, default="data/EthanolConcentration")
    return p

def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(args.seed)
    
    args.dataset = normalize_dataset_name(args.dataset)
    tag = f"minirocket_dlcr_tau{args.routing_temperature}_{args.dataset}_seed{args.seed}"
    run_dir = os.path.join(args.out_root, tag)
    _ensure_dir(run_dir)

    # 1. Load raw data and extract features (CPU-side)
    train_x_raw, test_x_raw, train_y, test_y, num_classes = _load_fixedsplit_raw(args)
    train_z = _extract_minirocket_features(train_x_raw, n_kernels=args.n_kernels, seed=args.seed)
    test_z = _extract_minirocket_features(test_x_raw, n_kernels=args.n_kernels, seed=args.seed)
    
    # 2. Setup GPU training
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model = MiniRocketDLCRAdapter(
        feature_dim=train_z.shape[1],
        num_classes=num_classes,
        prototypes_per_class=args.prototypes_per_class,
        routing_temperature=args.routing_temperature,
        ridge=args.closed_form_ridge,
        solve_mode=args.closed_form_solve_mode,
        enable_probe=args.closed_form_probe,
        init_beta=args.init_beta,
        prototype_geometry_mode=args.prototype_geometry_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Datasets
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(train_z), torch.from_numpy(train_y))
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(test_z), torch.from_numpy(test_y))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, n_train = 0.0, 0, 0
        fusion_alpha = _compute_fusion_alpha(args, epoch)
        
        for batch_z, batch_y in train_loader:
            batch_z, batch_y = batch_z.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_z, fusion_alpha=fusion_alpha)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_y.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            n_train += batch_y.size(0)

        # Eval
        model.eval()
        test_correct, n_test = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_z, batch_y in test_loader:
                batch_z, batch_y = batch_z.to(device), batch_y.to(device)
                logits = model(batch_z, fusion_alpha=fusion_alpha)
                pred = logits.argmax(dim=1)
                test_correct += (pred == batch_y).sum().item()
                n_test += batch_y.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")
        
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{args.dataset}] Epoch {epoch}/{args.epochs} train_loss={train_loss/n_train:.4f} train_acc={train_correct/n_train:.4f} test_acc={test_acc:.4f} beta={model.beta.item():.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss/n_train,
            "train_acc": train_correct/n_train,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "beta": model.beta.item()
        })

    _write_csv(os.path.join(run_dir, "history.csv"), history)
    
    # Final probe if enabled
    if args.closed_form_probe and hasattr(model.local_head, "probe_data"):
        _write_json(os.path.join(run_dir, "probe.json"), model.local_head.probe_data)

    print(f"[done][{args.dataset}][minirocket_dlcr] final_test_acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
