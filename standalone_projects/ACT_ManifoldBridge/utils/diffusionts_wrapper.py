import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# Add Diffusion-TS to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_TS_ROOT = PROJECT_ROOT / "external" / "DiffusionTS"
if str(DIFFUSION_TS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_TS_ROOT))

from engine.solver import Trainer
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from Models.interpretable_diffusion.classifier import Classifier
from Models.interpretable_diffusion.model_utils import cond_fn

class DiffusionTSNumpyDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, T, C)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.window = X.shape[1]
        self.var_num = X.shape[2]
        self.auto_norm = False # We handle normalization outside
        self.period = 'train'
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.period == 'test':
            return self.X[idx], self.y[idx]
        return self.X[idx]
    
    def shift_period(self, period):
        self.period = period

def fit_sample_diffusionts(X_train_ct, y_train, multiplier, seed, device="cpu", max_epochs=1000, batch_size=128):
    """
    Wrapper for Diffusion-TS class-conditional generation.
    X_train_ct: (N, C, T)
    """
    # 1. Preprocessing: Reshape and Scale
    N, C, T = X_train_ct.shape
    X_train_tc = np.transpose(X_train_ct, (0, 2, 1)) # (N, T, C)
    
    # Scale each channel
    scalers = []
    X_train_scaled = np.zeros_like(X_train_tc)
    for i in range(C):
        scaler = StandardScaler()
        X_train_scaled[:, :, i] = scaler.fit_transform(X_train_tc[:, :, i].reshape(-1, 1)).reshape(N, T)
        scalers.append(scaler)
    
    # 2. Setup Dataset and Dataloader
    dataset = DiffusionTSNumpyDataset(X_train_scaled, y_train)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, N), shuffle=True, drop_last=True)
    dl_info = {'dataloader': dataloader, 'dataset': dataset}
    
    # 3. Configure Model
    num_classes = len(np.unique(y_train))
    config = {
        'model': {
            'params': {
                'seq_length': T,
                'feature_size': C,
                'n_layer_enc': 3,
                'n_layer_dec': 2,
                'd_model': 64,
                'timesteps': 500,
                'sampling_timesteps': 100,
                'loss_type': 'l1',
                'beta_schedule': 'cosine',
                'n_heads': 4,
                'mlp_hidden_times': 4,
                'attn_pd': 0.0,
                'resid_pd': 0.0,
                'kernel_size': 1,
                'padding_size': 0
            }
        },
        'classifier': {
            'params': {
                'seq_length': T,
                'feature_size': C,
                'num_classes': num_classes,
                'n_layer_enc': 3,
                'n_embd': 64,
                'n_heads': 4,
                'mlp_hidden_times': 4,
                'attn_pd': 0.0,
                'resid_pd': 0.0,
                'max_len': T,
                'num_head_channels': 8
            }
        },
        'solver': {
            'base_lr': 1.0e-4,
            'max_epochs': max_epochs,
            'results_folder': './tmp_diffusionts',
            'gradient_accumulate_every': 2,
            'save_cycle': max_epochs + 1,
            'ema': {
                'decay': 0.995,
                'update_interval': 10
            },
            'scheduler': {
                'target': 'engine.lr_sch.ReduceLROnPlateauWithWarmup',
                'params': {
                    'factor': 0.5,
                    'patience': 3000,
                    'min_lr': 1.0e-5,
                    'threshold': 1.0e-1,
                    'threshold_mode': 'rel',
                    'warmup_lr': 8.0e-4,
                    'warmup': 500,
                    'verbose': False
                }
            }
        }
    }
    
    class SimpleArgs:
        def __init__(self, name, save_dir):
            self.name = name
            self.save_dir = save_dir
            
    args = SimpleArgs("diffusionts_aug", "./tmp_diffusionts")
    
    device_obj = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    
    model = Diffusion_TS(**config['model']['params']).to(device_obj)
    classifier = Classifier(**config['classifier']['params']).to(device_obj)
    
    trainer = Trainer(config=config, args=args, model=model, dataloader=dl_info)
    
    # 4. Train
    try:
        print(f"Training Diffusion-TS for {max_epochs} epochs...")
        trainer.train()
        
        print(f"Training Classifier Guidance for {max_epochs} epochs...")
        trainer.train_classfier(classifier)
    except Exception as e:
        print(f"Diffusion-TS training failed: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # 5. Sample
    try:
        print(f"Sampling augmented data (multiplier={multiplier})...")
        X_aug_list = []
        y_aug_list = []
        
        for cls in sorted(np.unique(y_train)):
            n_cls = np.sum(y_train == cls)
            num_to_sample = int(n_cls * multiplier)
            if num_to_sample == 0:
                continue
                
            model_kwargs = {
                'y': torch.full((num_to_sample,), cls, dtype=torch.long).to(device_obj)
            }
            
            # The cond_fn in DiffusionTS expects (x, t, classifier, y, classifier_scale)
            # But the sample_cond/fast_sample_cond calls cond_fn(x, t, **model_kwargs)
            # So we wrap it and slice y to match x's batch size
            def cond_fn_wrapper(x, t, y=None):
                if y is not None and len(y) > len(x):
                    y = y[:len(x)]
                return cond_fn(x, t, classifier=trainer.classifier, y=y, classifier_scale=1.0)
            
            samples = trainer.sample(num=num_to_sample, size_every=min(128, num_to_sample), 
                                     shape=[T, C], model_kwargs=model_kwargs, cond_fn=cond_fn_wrapper)
            samples = samples[:num_to_sample]
            
            # Unscale
            samples_unscaled = np.zeros_like(samples)
            for i in range(C):
                samples_unscaled[:, :, i] = scalers[i].inverse_transform(samples[:, :, i].reshape(-1, 1)).reshape(num_to_sample, T)
            
            # Transpose back to (N, C, T)
            samples_ct = np.transpose(samples_unscaled, (0, 2, 1))
            
            X_aug_list.append(samples_ct)
            y_aug_list.append(np.full(num_to_sample, cls))
            
        if not X_aug_list:
            return np.empty((0, C, T)), np.empty((0,))
            
        X_aug = np.concatenate(X_aug_list, axis=0)
        y_aug = np.concatenate(y_aug_list, axis=0)
        
        return X_aug, y_aug
    except Exception as e:
        print(f"Diffusion-TS sampling failed: {e}")
        import traceback
        traceback.print_exc()
        raise e
